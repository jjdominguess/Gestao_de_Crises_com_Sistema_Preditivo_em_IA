

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


RAIZ_PROJETO = Path(__file__).resolve().parent
CAMINHO_MODELO = RAIZ_PROJETO / "agents" / "modelo_risco_cardiaco.pkl"
CAMINHO_METRICAS = RAIZ_PROJETO / "resultados" / "metricas_modelo.json"

SEMENTE = 42
TAMANHO_BASE = 5000
FEATURES = [
    "idade_media",
    "carga_sistema",
    "tempo_espera",
    "fc_media",
    "spo2_media",
    "pa_sistolica",
    "temperatura",
    "densidade_eventos",
    "eventos_24h",
    "disponibilidade_recursos",
]


def gerar_base_sintetica() -> pd.DataFrame:
    """Reproduz os 5.000 registros e as 6 regras do enunciado da FIAP."""
    np.random.seed(SEMENTE)
    n = TAMANHO_BASE

    df = pd.DataFrame(
        {
            "idade_media": np.random.normal(55, 12, n).clip(18, 90),
            "carga_sistema": np.random.uniform(30, 100, n),
            "tempo_espera": np.random.normal(45, 20, n).clip(5, 180),
            "fc_media": np.random.normal(85, 15, n).clip(50, 150),
            "spo2_media": np.random.normal(95, 3, n).clip(70, 100),
            "pa_sistolica": np.random.normal(125, 20, n).clip(80, 200),
            "temperatura": np.random.normal(24, 5, n).clip(10, 40),
            "densidade_eventos": np.random.poisson(3, n),
            "eventos_24h": np.random.poisson(2, n),
            "disponibilidade_recursos": np.random.uniform(20, 100, n),
        }
    )

    pontos_risco = (
        (df["carga_sistema"] > 80).astype(int)
        + (df["tempo_espera"] > 60).astype(int)
        + (df["spo2_media"] < 92).astype(int)
        + (df["fc_media"] > 110).astype(int)
        + (df["eventos_24h"] > 4).astype(int)
        + (df["disponibilidade_recursos"] < 40).astype(int)
    )
    df["pico_risco"] = (pontos_risco >= 2).astype(int)
    return df


def treinar_modelo(df: pd.DataFrame) -> RandomForestClassifier:
    """Treina, avalia e salva o mesmo classificador utilizado no projeto."""
    if df[FEATURES + ["pico_risco"]].isna().any().any():
        raise ValueError("A base possui valores ausentes em campos obrigatorios.")

    X = df[FEATURES]
    y = df["pico_risco"]
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.20, random_state=SEMENTE, stratify=y
    )

    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=SEMENTE,
    )
    modelo.fit(X_treino, y_treino)

    classes_previstas = modelo.predict(X_teste)
    probabilidades = modelo.predict_proba(X_teste)[:, 1]
    matriz = confusion_matrix(y_teste, classes_previstas, labels=[0, 1])
    metricas = {
        "total_registros": int(len(df)),
        "registros_treino": int(len(X_treino)),
        "registros_teste": int(len(X_teste)),
        "acuracia": float(accuracy_score(y_teste, classes_previstas)),
        "precisao": float(precision_score(y_teste, classes_previstas, zero_division=0)),
        "recall": float(recall_score(y_teste, classes_previstas, zero_division=0)),
        "f1_score": float(f1_score(y_teste, classes_previstas, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_teste, probabilidades)),
        "matriz_confusao": matriz.tolist(),
    }

    # A etapa seguinte (agentes) busca o modelo exatamente neste endereco.
    CAMINHO_MODELO.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, CAMINHO_MODELO)

    CAMINHO_METRICAS.parent.mkdir(parents=True, exist_ok=True)
    CAMINHO_METRICAS.write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nBase gerada: {len(df)} casos sintéticos")
    print(f"Treino: {len(X_treino)} | Teste: {len(X_teste)}")
    print("\n=== AVALIACAO DO MODELO ===")
    for nome in ("acuracia", "precisao", "recall", "f1_score", "roc_auc"):
        print(f"{nome}: {metricas[nome]:.4f}")
    print("\nMatriz de confusao [classes 0, 1]:")
    print(matriz)
    print("\nRelatorio por classe:")
    print(classification_report(y_teste, classes_previstas, zero_division=0))
    print(f"Modelo salvo em: {CAMINHO_MODELO}")
    print(f"Metricas salvas em: {CAMINHO_METRICAS}")
    return modelo


def testar_novo_caso() -> None:
    """Testa o caso do enunciado usando o modelo recarregado do disco."""
    novo_caso = pd.DataFrame(
        [
            {
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
        ]
    )

    # Carregue apenas modelos .pkl produzidos por voce ou de origem confiavel.
    modelo = joblib.load(CAMINHO_MODELO)
    probabilidade = float(modelo.predict_proba(novo_caso[FEATURES])[0, 1])
    classe = int(modelo.predict(novo_caso[FEATURES])[0])
    classificacao = "PICO DE RISCO" if classe == 1 else "SEM PICO DE RISCO"

    print("\n=== NOVO CASO (DADOS SINTETICOS) ===")
    print(f"Probabilidade prevista: {probabilidade:.4f} ({probabilidade:.2%})")
    print(f"Classe prevista: {classe} - {classificacao}")
    print("AVISO: resultado academico, nao validado para uso clinico.")


if __name__ == "__main__":
    base = gerar_base_sintetica()
    treinar_modelo(base)
    testar_novo_caso()
