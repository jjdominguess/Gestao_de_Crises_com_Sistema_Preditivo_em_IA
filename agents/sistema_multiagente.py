import joblib
import pandas as pd
from pathlib import Path

# ==============================================================================
# 1. FERRAMENTAS (TOOLS)
# ==============================================================================

def tool_consultar_modelo_preditivo(dados_paciente: dict) -> dict:
    """Tool que carrega o modelo treinado na Parte 1 e infere o risco."""
    dados = dict(dados_paciente)
    if "disponibilidade recursos" in dados:
        dados["disponibilidade_recursos"] = dados.pop("disponibilidade recursos")
        
    features = [
        "idade_media", "carga_sistema", "tempo_espera", "fc_media",
        "spo2_media", "pa_sistolica", "temperatura", "densidade_eventos",
        "eventos_24h", "disponibilidade_recursos"
    ]
    df_paciente = pd.DataFrame([dados])
    modelo_path = Path(__file__).resolve().with_name("modelo_risco_cardiaco.pkl")
    modelo = joblib.load(modelo_path)
    
    probabilidade = float(modelo.predict_proba(df_paciente[features])[:, 1][0])
    classe = int(modelo.predict(df_paciente[features])[0])
    
    return {
        "probabilidade_risco": round(probabilidade, 2),
        "classe_predita": classe,
        "classificacao": "Crítico / Iminente" if probabilidade >= 0.8 else "Controlado"
    }

def tool_consultar_protocolos(dados_paciente: dict, probabilidade: float) -> list:
    """Tool que avalia as 10 regras institucionais frente aos sinais vitais."""
    protocolos = []
    
    # Regra 1: Carga > 85%
    if dados_paciente.get("carga_sistema", 0) > 85:
        protocolos.append("Se carga do sistema > 85% → abrir unidades temporárias")
        
    # Regra 2: SpO2 < 88% (ou criticidade respiratória)
    spo2 = dados_paciente.get("spo2_media", 100)
    if spo2 < 88 or (spo2 <= 90 and probabilidade > 0.85):
        protocolos.append("Se SPO2 < 88% → iniciar suporte respiratório imediato")
        
    # Regra 3: Recursos < 30%
    recursos = dados_paciente.get("disponibilidade_recursos", dados_paciente.get("disponibilidade recursos", 100))
    if recursos < 30:
        protocolos.append("Se disponibilidade_recursos < 30% → redistribuir insumos")
        
    # Regra 4: Eventos 24h >= 5
    if dados_paciente.get("eventos_24h", 0) >= 5:
        protocolos.append("Se eventos_24h > 5 → ativar equipe de resposta rápida")
        
    # Regra 5: Densidade de eventos > 6
    if dados_paciente.get("densidade_eventos", 0) > 6:
        protocolos.append("Se densidade_eventos > 6 → ativar plano de contingência")
        
    # Regra 6: Risco sistêmico alto
    if probabilidade >= 0.70:
        protocolos.append("Se risco sistêmico alto → ativar estado de emergência")
        
    return protocolos


# ==============================================================================
# 2. OS AGENTES ESPECIALIZADOS
# ==============================================================================

class AgenteAnalistaRisco:
    def __init__(self):
        self.nome = "Agente Analista de Risco"

    def executar(self, dados_paciente: dict) -> dict:
        resultado_ml = tool_consultar_modelo_preditivo(dados_paciente)
        return {
            "agente": self.nome,
            "resultado_ml": resultado_ml,
            "mensagem": f"Modelo consultado. Risco calculado em {resultado_ml['probabilidade_risco']:.2f}."
        }

class AgenteEspecialistaProtocolos:
    def __init__(self):
        self.nome = "Agente Especialista em Protocolos"

    def executar(self, dados_paciente: dict, probabilidade: float) -> dict:
        protocolos_ativados = tool_consultar_protocolos(dados_paciente, probabilidade)
        return {
            "agente": self.nome,
            "protocolos": protocolos_ativados,
            "mensagem": f"{len(protocolos_ativados)} protocolo(s) institucional(is) ativado(s)."
        }

class AgentePlanejadorResposta:
    def __init__(self):
        self.nome = "Agente Planejador de Resposta"

    def formatar_resposta_final(self, risco: float, protocolos: list) -> str:
        linhas = "\n".join([f"- {p}" for p in protocolos])
        resposta = (
            f"RISCO PREVISTO: {risco:.2f}\n\n"
            f"PROTOCOLOS SUGERIDOS:\n"
            f"{linhas}"
        )
        return resposta


# ==============================================================================
# 3. O AGENTE ORQUESTRADOR (COORDENA OS HANDOFFS)
# ==============================================================================

class AgenteOrquestrador:
    def __init__(self):
        self.nome = "Agente Orquestrador"
        self.analista_risco = AgenteAnalistaRisco()
        self.especialista_protocolos = AgenteEspecialistaProtocolos()
        self.planejador = AgentePlanejadorResposta()
        self.historico = []

    def log(self, emissor: str, mensagem: str):
        self.historico.append(f"[{emissor}]: {mensagem}")

    def processar_paciente(self, paciente: dict) -> str:
        self.historico.clear()
        self.log(self.nome, "Paciente recebido. Iniciando triagem multiagente.")

        # --- HANDOFF 1: Orquestrador -> Analista de Risco ---
        self.log(self.nome, f"[HANDOFF] Transferindo paciente para '{self.analista_risco.nome}'.")
        resultado_risco = self.analista_risco.executar(paciente)
        self.log(self.analista_risco.nome, resultado_risco["mensagem"])

        probabilidade = resultado_risco["resultado_ml"]["probabilidade_risco"]

        # --- HANDOFF 2: Analista de Risco -> Especialista em Protocolos ---
        self.log(self.analista_risco.nome, f"[HANDOFF] Transferindo score ({probabilidade:.2f}) para '{self.especialista_protocolos.nome}'.")
        resultado_protocolos = self.especialista_protocolos.executar(paciente, probabilidade)
        self.log(self.especialista_protocolos.nome, resultado_protocolos["mensagem"])

        protocolos = resultado_protocolos["protocolos"]

        # --- HANDOFF 3: Especialista em Protocolos -> Planejador de Resposta ---
        self.log(self.especialista_protocolos.nome, f"[HANDOFF] Transferindo protocolos para '{self.planejador.nome}'.")
        resposta_final = self.planejador.formatar_resposta_final(probabilidade, protocolos)
        self.log(self.planejador.nome, "Resposta final consolidada com sucesso.")

        return resposta_final
    
if __name__ == "__main__":
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
        "disponibilidade recursos": 25
    }

    # Inicializa o Orquestrador e executa
    orquestrador = AgenteOrquestrador()
    resultado = orquestrador.processar_paciente(novo_caso)

    # Exibe a Trilha de Auditoria (Handoffs)
    print("=" * 60)
    print("TRILHA DE HANDOFFS ENTRE OS AGENTES:")
    print("=" * 60)
    for passo in orquestrador.historico:
        print(passo)

    # Exibe a Resposta Final Formatada
    print("\n" + "=" * 60)
    print(resultado)
    print("=" * 6)