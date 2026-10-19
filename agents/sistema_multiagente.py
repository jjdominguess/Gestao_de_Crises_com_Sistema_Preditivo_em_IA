import json
import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from pydantic import AliasChoices, BaseModel, ConfigDict, Field


RAIZ_PROJETO = Path(__file__).resolve().parent.parent
CAMINHO_MODELO = RAIZ_PROJETO / "agents" / "modelo_risco_cardiaco.pkl"
CAMINHO_HISTORICO = RAIZ_PROJETO / "resultados" / "historico_multiagente.json"

# A ordem das colunas deve ser igual à usada no treinamento do main.py.
FEATURES = [
    "idade_media", "carga_sistema", "tempo_espera", "fc_media",
    "spo2_media", "pa_sistolica", "temperatura", "densidade_eventos",
    "eventos_24h", "disponibilidade_recursos",
]

AVISO = "Demonstração acadêmica com dados sintéticos; não possui validação clínica."


class CasoPaciente(BaseModel):
    """Valida as dez entradas usadas pelo classificador."""

    model_config = ConfigDict(extra="forbid")

    idade_media: float = Field(ge=0, le=120)
    carga_sistema: float = Field(ge=0, le=100)
    tempo_espera: float = Field(ge=0)
    fc_media: float = Field(gt=0)
    spo2_media: float = Field(ge=0, le=100)
    pa_sistolica: float = Field(gt=0)
    temperatura: float = Field(ge=-50, le=70)
    densidade_eventos: int = Field(ge=0)
    eventos_24h: int = Field(ge=0)
    disponibilidade_recursos: float = Field(
        ge=0, le=100,
        validation_alias=AliasChoices("disponibilidade_recursos", "disponibilidade recursos"),
    )


class RespostaFinal(BaseModel):
    """Validação estrutural da resposta produzida pelo planejador."""

    model_config = ConfigDict(extra="forbid")

    probabilidade_risco: float = Field(ge=0, le=1)
    classe_predita: int = Field(ge=0, le=1)
    classificacao_risco: str
    protocolos_sugeridos: list[str]
    resumo_operacional: str
    aviso: str


@dataclass
class EstadoExecucao:
    """Contexto local compartilhado entre agents e tools do SDK."""

    paciente: CasoPaciente
    resultado_ml: dict | None = None
    protocolos: list[str] = field(default_factory=list)
    trilha: list[str] = field(default_factory=list)
    ferramentas_executadas: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def carregar_modelo():
    """Carrega somente o modelo gerado localmente pelo main.py do projeto."""
    if not CAMINHO_MODELO.is_file():
        raise FileNotFoundError(
            f"Modelo não encontrado: {CAMINHO_MODELO}. Rode 'python main.py' primeiro."
        )
    # Segurança: nunca use joblib.load com arquivos de origem desconhecida.
    modelo = joblib.load(CAMINHO_MODELO)
    if not hasattr(modelo, "predict_proba") or not hasattr(modelo, "predict"):
        raise TypeError("O arquivo não contém um classificador compatível.")
    return modelo


def calcular_risco(paciente: CasoPaciente) -> dict:
    """Inferência reproduzível: o LLM não inventa a probabilidade."""
    modelo = carregar_modelo()
    dataframe = pd.DataFrame([paciente.model_dump()])[FEATURES]
    probabilidade = float(modelo.predict_proba(dataframe)[0, 1])
    classe = int(modelo.predict(dataframe)[0])
    return {
        "probabilidade_risco": round(probabilidade, 4),
        "classe_predita": classe,
        "classificacao_risco": "PICO DE RISCO" if classe == 1 else "SEM PICO DE RISCO",
    }


def selecionar_protocolos(paciente: CasoPaciente, resultado_ml: dict) -> list[str]:
    """Aplica literalmente as dez condições simuladas do enunciado da FIAP."""
    regras = [
        (paciente.spo2_media < 88,
         "Se SPO2 < 88% → iniciar suporte respiratório imediato"),
        (paciente.carga_sistema > 85,
         "Se carga do sistema > 85% → abrir unidades temporárias"),
        (paciente.fc_media > 120,
         "Se FC > 120 → iniciar monitoramento intensivo"),
        (paciente.eventos_24h > 5,
         "Se eventos_24h > 5 → ativar equipe de resposta rápida"),
        (paciente.disponibilidade_recursos < 30,
         "Se disponibilidade_recursos < 30% → redistribuir insumos"),
        (paciente.tempo_espera > 90,
         "Se tempo_espera > 90 min → priorizar triagem emergencial"),
        (paciente.temperatura > 38,
         "Se temperatura > 38°C média → iniciar protocolo infeccioso"),
        (paciente.densidade_eventos > 6,
         "Se densidade_eventos > 6 → ativar plano de contingência"),
        (paciente.pa_sistolica < 90,
         "Se PA sistólica < 90 → iniciar protocolo de choque"),
        # Critério didático declarado: "risco sistêmico alto" = classe 1.
        (resultado_ml["classe_predita"] == 1,
         "Se risco sistêmico alto → ativar estado de emergência"),
    ]
    return [descricao for ativada, descricao in regras if ativada]


def criar_agentes():
    """Monta um fluxo com três handoffs reais e três ferramentas do SDK."""
    from agents import Agent, RunContextWrapper, function_tool, handoff

    modelo_llm = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    @function_tool
    def consultar_modelo_preditivo(ctx: RunContextWrapper[EstadoExecucao]) -> str:
        """Consulta o classificador Random Forest treinado para o caso sintético atual."""
        estado = ctx.context
        estado.resultado_ml = calcular_risco(estado.paciente)
        estado.ferramentas_executadas.append("consultar_modelo_preditivo")
        estado.trilha.append("[TOOL] consultar_modelo_preditivo executada")
        return json.dumps(estado.resultado_ml, ensure_ascii=False)

    @function_tool
    def consultar_protocolos(ctx: RunContextWrapper[EstadoExecucao]) -> str:
        """Consulta as dez regras simuladas e devolve apenas protocolos aplicáveis."""
        estado = ctx.context
        if estado.resultado_ml is None:
            raise RuntimeError("É obrigatório calcular o risco antes dos protocolos.")
        estado.protocolos = selecionar_protocolos(estado.paciente, estado.resultado_ml)
        estado.ferramentas_executadas.append("consultar_protocolos")
        estado.trilha.append("[TOOL] consultar_protocolos executada")
        return json.dumps({"protocolos_sugeridos": estado.protocolos}, ensure_ascii=False)

    @function_tool
    def consultar_resumo_validado(ctx: RunContextWrapper[EstadoExecucao]) -> str:
        """Devolve a previsão e os protocolos exatos para a resposta final."""
        estado = ctx.context
        if estado.resultado_ml is None or "consultar_protocolos" not in estado.ferramentas_executadas:
            raise RuntimeError("O analista de risco e os protocolos precisam executar primeiro.")
        estado.ferramentas_executadas.append("consultar_resumo_validado")
        estado.trilha.append("[TOOL] consultar_resumo_validado executada")
        return json.dumps(
            {**estado.resultado_ml, "protocolos_sugeridos": estado.protocolos, "aviso": AVISO},
            ensure_ascii=False,
        )

    def ao_encaminhar_analista(ctx: RunContextWrapper[EstadoExecucao]):
        ctx.context.trilha.append("[HANDOFF] Orquestrador → Analista de Risco")

    def ao_encaminhar_protocolos(ctx: RunContextWrapper[EstadoExecucao]):
        ctx.context.trilha.append("[HANDOFF] Analista de Risco → Especialista em Protocolos")

    def ao_encaminhar_planejador(ctx: RunContextWrapper[EstadoExecucao]):
        ctx.context.trilha.append("[HANDOFF] Especialista em Protocolos → Planejador")

    planejador = Agent[EstadoExecucao](
        name="Planejador de Resposta",
        model=modelo_llm,
        instructions=(
            "Você é o planejador do sistema acadêmico CardioIA. "
            "É OBRIGATÓRIO usar consultar_resumo_validado antes de responder. "
            "Devolva todos os campos da resposta estruturada. "
            "Copie SEM ALTERAR a probabilidade, a classe, a classificação, "
            "a lista COMPLETA de protocolos e o aviso recebidos da ferramenta. "
            "No campo resumo_operacional, explique brevemente o resultado do cenário "
            "simulado SEM inventar ações nem protocolos extras. "
            "Não ofereça aconselhamento médico para pacientes reais."
        ),
        tools=[consultar_resumo_validado],
        output_type=RespostaFinal,
    )

    especialista = Agent[EstadoExecucao](
        name="Especialista em Protocolos",
        model=modelo_llm,
        instructions=(
            "Você é o especialista em protocolos SIMULADOS da CardioIA. "
            "É OBRIGATÓRIO chamar consultar_protocolos exatamente para este caso. "
            "Não modifique os limiares nem crie protocolos novos. "
            "Depois da ferramenta, faça obrigatoriamente o handoff para "
            "Planejador de Resposta, sem emitir resposta final."
        ),
        tools=[consultar_protocolos],
        handoffs=[handoff(agent=planejador, on_handoff=ao_encaminhar_planejador)],
    )

    analista = Agent[EstadoExecucao](
        name="Analista de Risco",
        model=modelo_llm,
        instructions=(
            "Você é o analista de risco do caso sintético. "
            "É OBRIGATÓRIO chamar consultar_modelo_preditivo para calcular "
            "probabilidade e classe do Random Forest. Não estime números por conta própria. "
            "Depois, faça obrigatoriamente o handoff para Especialista em Protocolos. "
            "Não produza a resposta final."
        ),
        tools=[consultar_modelo_preditivo],
        handoffs=[handoff(agent=especialista, on_handoff=ao_encaminhar_protocolos)],
    )

    orquestrador = Agent[EstadoExecucao](
        name="Orquestrador",
        model=modelo_llm,
        instructions=(
            "Você coordena a CardioIA, um exercício acadêmico com dados sintéticos. "
            "Para cada caso, sua única tarefa é transferir imediatamente o controle "
            "por handoff para Analista de Risco. Não calcule risco nem dê resposta final."
        ),
        handoffs=[handoff(agent=analista, on_handoff=ao_encaminhar_analista)],
    )
    return orquestrador


def validar_resposta(resposta: RespostaFinal, estado: EstadoExecucao):
    """Rejeita qualquer divergência entre a resposta LLM e as tools determinísticas."""
    if estado.resultado_ml is None:
        raise ValueError("O agente de risco não consultou o modelo.")
    ferramentas_esperadas = [
        "consultar_modelo_preditivo", "consultar_protocolos", "consultar_resumo_validado"
    ]
    if estado.ferramentas_executadas != ferramentas_esperadas:
        raise ValueError(f"Fluxo incompleto ou fora de ordem: {estado.ferramentas_executadas}")
    if len([passo for passo in estado.trilha if passo.startswith("[HANDOFF]")]) != 3:
        raise ValueError("Era esperado um total de três handoffs reais do SDK.")
    if abs(resposta.probabilidade_risco - estado.resultado_ml["probabilidade_risco"]) > 0.0001:
        raise ValueError("Probabilidade na resposta difere do modelo.")
    if resposta.classe_predita != estado.resultado_ml["classe_predita"]:
        raise ValueError("Classe na resposta difere do modelo.")
    if resposta.classificacao_risco != estado.resultado_ml["classificacao_risco"]:
        raise ValueError("Classificação na resposta difere do modelo.")
    if len(resposta.protocolos_sugeridos) != len(estado.protocolos) or sorted(
        resposta.protocolos_sugeridos
    ) != sorted(estado.protocolos):
        raise ValueError("Protocolos na resposta diferem das regras simuladas.")
    if resposta.aviso != AVISO:
        raise ValueError("O aviso acadêmico deve ser preservado.")


def executar_caso(dados_paciente: dict) -> RespostaFinal:
    """Executa OpenAI Agents SDK e salva histórico/auditoria para o vídeo da FIAP."""
    from agents import Runner
    from dotenv import load_dotenv

    load_dotenv(RAIZ_PROJETO / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "Defina OPENAI_API_KEY na variável de ambiente ou no arquivo .env da raiz."
        )

    paciente = CasoPaciente.model_validate(dados_paciente)
    estado = EstadoExecucao(paciente=paciente)
    orquestrador = criar_agentes()
    mensagem = (
        "Execute a triagem acadêmica multiagente para o caso simulado a seguir. "
        "Use as ferramentas oficiais na ordem: modelo → protocolos → resposta. "
        "Faça os handoffs entre os agentes. Dados do caso:\n"
        + json.dumps(paciente.model_dump(), ensure_ascii=False)
    )
    resultado = Runner.run_sync(
        orquestrador, mensagem, context=estado, max_turns=20
    )
    if resultado.last_agent.name != "Planejador de Resposta":
        raise ValueError("O fluxo não terminou no Agente Planejador de Resposta.")
    resposta = RespostaFinal.model_validate(resultado.final_output)
    validar_resposta(resposta, estado)

    CAMINHO_HISTORICO.parent.mkdir(parents=True, exist_ok=True)
    CAMINHO_HISTORICO.write_text(
        json.dumps(
            {
                "caso_sintetico": paciente.model_dump(),
                "trilha_auditoria": estado.trilha,
                "historico_mensagens": resultado.to_input_list(),
                "resposta": resposta.model_dump(),
            },
            ensure_ascii=False, indent=2, default=str,
        ),
        encoding="utf-8",
    )
    print("\n=== HANDOFFS E TOOLS ===")
    for etapa in estado.trilha:
        print(etapa)
    return resposta


if __name__ == "__main__":
    # Caso sintético da FIAP (campo disponibilidade_recursos corrigido).
    novo_caso = {
        "idade_media": 62,
        "carga_sistema": 92,
        "tempo_espera": 75,
        "fc_media": 118,
        "spo2_media": 89,
        "pa_sistolica": 100,
        "temperatura": 37.5,
        "densidade_eventos": 7,
        "eventos_24h": 5,
        "disponibilidade_recursos": 25,
    }
    # Primeiro teste sem API: valida o modelo e as 10 regras localmente.
    if "--teste-local" in sys.argv:
        paciente = CasoPaciente.model_validate(novo_caso)
        risco = calcular_risco(paciente)
        protocolos = selecionar_protocolos(paciente, risco)
        print("=== TESTE LOCAL - SEM CHAMADAS A API ===")
        print(f"Probabilidade: {risco['probabilidade_risco']:.4f}")
        print(f"Classe: {risco['classe_predita']} - {risco['classificacao_risco']}")
        print(f"Protocolos aplicáveis: {len(protocolos)}")
        for regra in protocolos:
            print(f"- {regra}")
        print(AVISO)
        sys.exit(0)

    resposta_final = executar_caso(novo_caso)
    print("\n=== RESULTADO FINAL (CASO SIMULADO) ===")
    print(f"Probabilidade de pico de risco: {resposta_final.probabilidade_risco:.4f}")
    print(f"Classe: {resposta_final.classe_predita} — {resposta_final.classificacao_risco}")
    print("Protocolos sugeridos:")
    for protocolo in resposta_final.protocolos_sugeridos:
        print(f"- {protocolo}")
    print(f"Resumo: {resposta_final.resumo_operacional}")
    print(resposta_final.aviso)
    print(f"\nHistórico salvo em: {CAMINHO_HISTORICO}")
