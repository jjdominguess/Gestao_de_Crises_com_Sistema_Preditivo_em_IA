# FIAP - Faculdade de Informática e Administração Paulista

<p align="center">
  <a href="https://www.fiap.com.br/">
    <img src="assets/logo-fiap.png" alt="FIAP - Faculdade de Informática e Administração Paulista" width="40%">
  </a>
</p>

# Graduação ON em Inteligência Artificial
## CardioIA - Fase 6
### Gestão de Clusters, Segurança Quântica e Colaboração Multiagente

---

## Sobre este Repositório

Este repositório contém o projeto CardioIA, desenvolvido na Fase 6 da graduação em Inteligência Artificial da FIAP. A atividade envolve a classificação de picos de risco em dados sintéticos e a integração do modelo preditivo com um sistema de quatro agentes especializados.

O trabalho reúne o código de treinamento, o modelo salvo, o notebook de avaliação, a implementação multiagente e os documentos técnicos da entrega.

## Objetivo

Implementar um fluxo capaz de receber dados de um caso simulado, calcular a probabilidade de pico de risco e selecionar os protocolos correspondentes às condições identificadas. As etapas são:

- Gerar a base sintética e treinar um modelo supervisionado para a variável `pico_risco`.
- Avaliar o classificador em um conjunto de teste separado.
- Integrar o modelo ao OpenAI Agents SDK, com ferramentas e transferências de controle entre agentes.
- Consolidar a probabilidade estimada, a classe prevista e os protocolos sugeridos em uma resposta estruturada.

Os dados e os protocolos são simulados. O projeto não foi validado para utilização em decisões médicas reais.

## Integrantes

| Nome completo | RM |
|---|---|
| [Nome do integrante 1] | [RM] |
| [Nome do integrante 2] | [RM] |
| [Nome do integrante 3] | [RM] |
| [Nome do integrante 4] | [RM] |
| [Nome do integrante 5, se houver] | [RM] |

## Base de Dados e Modelo Preditivo

A base de treinamento contém **5.000 registros sintéticos**, gerados com NumPy e Pandas, utilizando `seed=42`. São consideradas dez variáveis:

`idade_media`, `carga_sistema`, `tempo_espera`, `fc_media`, `spo2_media`, `pa_sistolica`, `temperatura`, `densidade_eventos`, `eventos_24h` e `disponibilidade_recursos`.

O campo `pico_risco` é calculado a partir de seis condições estabelecidas na atividade. O valor é 1 quando pelo menos duas dessas condições ocorrem; nos demais registros, é 0.

O algoritmo utilizado é **Random Forest**, com 100 árvores, profundidade máxima de 10 e `min_samples_split=5`. A base foi dividida em 80% para treinamento e 20% para teste, mantendo a proporção das classes.

### Avaliação do modelo

| Indicador | Resultado |
|---|---:|
| Registros de treinamento | 4.000 |
| Registros de teste | 1.000 |
| Acurácia | 99,90% |
| Precisão (classe 1) | 100,00% |
| Recall (classe 1) | 99,62% |
| F1-score (classe 1) | 99,81% |

Matriz de confusão:

```text
             Previsto 0  Previsto 1
Real 0           734           0
Real 1             1         265
```

A elevada acurácia está associada à definição determinística do alvo na base sintética. Ela não demonstra capacidade de previsão em pacientes reais.

## Sistema Multiagente

A aplicação utiliza o OpenAI Agents SDK com os seguintes agentes:

| Agente | Responsabilidade |
|---|---|
| Orquestrador | Recebe o caso e encaminha a execução ao agente de risco. |
| Analista de Risco | Consulta o modelo salvo pela ferramenta `consultar_modelo_preditivo`. |
| Especialista em Protocolos | Verifica as condições da base por meio de `consultar_protocolos`. |
| Planejador de Resposta | Consolida os resultados usando `consultar_resumo_validado` e gera a resposta final. |

A execução utiliza três handoffs, histórico de mensagens e validação estruturada de entrada e saída. O histórico gerado é salvo em `resultados/historico_multiagente.json`.

### Resultado do caso simulado

Na execução de teste, o classificador retornou probabilidade **1,0000** e classe **1 - PICO DE RISCO**. Os protocolos selecionados foram:

- Abertura de unidades temporárias, devido à carga do sistema superior a 85%.
- Redistribuição de insumos, devido à disponibilidade de recursos inferior a 30%.
- Ativação do plano de contingência, devido à densidade de eventos superior a 6.
- Ativação do estado de emergência, em razão do risco sistêmico alto.

## Estrutura Macro do Repositório

```text
Gestao_de_Crises_com_Sistema_Preditivo_em_IA/
├── main.py
├── agents/
│   ├── modelo_risco_cardiaco.pkl
│   └── sistema_multiagente.py
├── jupyter/
│   └── main.ipynb
├── resultados/
│   ├── metricas_modelo.json
│   └── historico_multiagente.json
├── docs/
│   ├── Relatorio_Tecnico_CardioIA.pdf
│   └── Arquitetura_Multiagente_CardioIA.pdf
├── assets/
│   └── logo-fiap.png
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

O arquivo `historico_multiagente.json` é gerado ao executar o sistema com a API.

## Requisitos e Instalação

- Python 3.11
- Dependências do arquivo `requirements.txt`
- Chave de API da OpenAI para executar o fluxo completo dos agentes

Na raiz do projeto, crie e ative um ambiente virtual:

```bash
python -m venv .venv
```

No Windows, pelo Git Bash:

```bash
source .venv/Scripts/activate
```

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

Copie o arquivo `.env.example` para `.env` e configure a variável `OPENAI_API_KEY`. O arquivo `.env` não deve ser enviado ao repositório.

## Execução

**Treinamento, avaliação e salvamento do modelo:**

```bash
python main.py
```

**Teste local do modelo e dos protocolos, sem chamadas à API:**

```bash
python agents/sistema_multiagente.py --teste-local
```

**Execução completa dos quatro agentes:**

```bash
python agents/sistema_multiagente.py
```

A execução completa apresenta os handoffs, as ferramentas utilizadas, a probabilidade, a classe e os protocolos sugeridos.

## Documentação

- [Relatório técnico](docs/Relatorio_Tecnico_CardioIA.pdf) — preparação da base, treinamento, avaliação e limitações do modelo (2 páginas).
- [Arquitetura multiagente](docs/Arquitetura_Multiagente_CardioIA.pdf) — diagrama, responsabilidades dos agentes, handoffs, ferramentas e exemplo de entrada e saída (3 páginas).
- [Notebook Google Colab](jupyter/main.ipynb) — geração da base sintética, treinamento, métricas e simulação.

## Vídeo Demonstrativo

**YouTube (não listado):** [INSERIR_LINK_DO_VIDEO]

A demonstração apresenta o treinamento do modelo, a execução multiagente e o resultado do caso simulado, em até quatro minutos.

## Entrega

O repositório deve permanecer privado e ser compartilhado com o perfil `fiap-tutoria`. A entrega no Portal do Aluno será realizada com um único arquivo `.txt` contendo o link do repositório e o link do vídeo demonstrativo.

## Licença

A estrutura deste README foi adaptada do [MODELO GIT FIAP](https://github.com/SabrinaOtoni/TEMPLATE-FIAP-GRAD-ON-IA), disponibilizado sob [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Essa atribuição se refere ao modelo de documentação e não estabelece, por si só, uma licença para o código da CardioIA.
