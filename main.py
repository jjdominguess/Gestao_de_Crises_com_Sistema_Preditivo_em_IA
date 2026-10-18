import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score, 
    confusion_matrix,
    classification_report
    )


# Definir a semente para que os números gerados sejam sempre os mesmos
np.random.seed(42)
n = 5000

#Criação do DataFrame 
df = pd.DataFrame({
    "idade_media": np.random.normal(55, 12, n).clip(18, 90),
    "carga_sistema": np.random.uniform(30, 100, n),
    "tempo_espera": np.random.normal(45, 20, n).clip(5, 180),
    "fc_media": np.random.normal(85, 15, n).clip(50, 150),
    "spo2_media": np.random.normal(95, 3, n).clip(70, 100),
    "pa_sistolica": np.random.normal(125, 20, n).clip(80, 200),
    "temperatura": np.random.normal(24, 5, n).clip(10, 40),
    "densidade_eventos": np.random.poisson(3, n),
    "eventos_24h": np.random.poisson(2, n),
    "disponibilidade_recursos": np.random.uniform(20, 100, n)
})
# 2. Calcular a pontuação de risco com as 6 regras exatas da aula
risk_score = (
    (df["carga_sistema"] > 80).astype(int) +
    (df["tempo_espera"] > 60).astype(int) +
    (df["spo2_media"] < 92).astype(int) +
    (df["fc_media"] > 110).astype(int) +
    (df["eventos_24h"] > 4).astype(int) +
    (df["disponibilidade_recursos"] < 40).astype(int)
)
df["pico_risco"] = (risk_score >= 2).astype(int)

print(df["pico_risco"].value_counts())
print("\nPercentual:")
print(df["pico_risco"].value_counts(normalize=True) * 100)
print(df.isnull().sum())

# 3. Lista de features atualizada com as 10 colunas
features = [
    "idade_media", "carga_sistema", "tempo_espera", "fc_media",
    "spo2_media", "pa_sistolica", "temperatura", "densidade_eventos",
    "eventos_24h", "disponibilidade_recursos"
]

X = df[features]
y = df["pico_risco"]

# Divisão do data frame em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Total de amostras de Treino: {len(X_train)}")
print(f"Total de amostras de Teste: {len(X_test)}")

print("Proporção de treino: ")
print(y_train.value_counts(normalize=True) * 100)

print("Proporção de teste: ")
print(y_test.value_counts(normalize=True) * 100)


modelo = RandomForestClassifier(
    n_estimators=100,    # Cria 100 árvores de decisão
    max_depth=10,        # Limita a profundidade máxima das árvores para não deorar ruído
    min_samples_split=5, # Mínimo de amostras para criar uma nova bifurcação
    random_state=42      # Semente fixa para garantir reprodutibilidade 
    )

# Treinamento 
modelo.fit(X_train, y_train)

print("Modelo treinado com sucesso!")

# 1. A classe prevista (0 ou 1)
y_pred = modelo.predict(X_test)

# 2. A probabilidade prevista da classe 1 (ex.: 0.91, 0.15, 0.84)
y_proba = modelo.predict_proba(X_test)[:, 1]

  
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)

print("=" * 45)
print(f"Acurácia: {acc:.4f} ({acc*100:.2f}%)")
print(f"Precisão: {prec:.4f} ({prec*100:.2f}%)")
print(f"Revocação: {rec:.4f} ({rec*100:.2f}%)")
print(f"F1-Score: {f1:.4f}")
print(f"ROC AUC: {roc:.4f}")
print("=" * 45)

print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))


joblib.dump(modelo, "modelo_risco_cardiaco.pkl")
print("\nModelo salvo com sucesso no arquivo 'modelo_risco_cardiaco.pkl'!")

novo_caso = pd.DataFrame([{
    "idade_media": 62,
    "carga_sistema": 92,
    "tempo_espera": 75,
    "fc_media": 118,
    "spo2_media": 89,
    "pa_sistolica": 100,
    "temperatura": 37.5,
    "densidade_eventos": 7,
    "eventos_24h": 5,
    "disponibilidade_recursos": 25
}])

modelo_carregado = joblib.load("modelo_risco_cardiaco.pkl")
risco_previsto = modelo_carregado.predict_proba(novo_caso[features])[:,1][0]
classe_prevista = modelo_carregado.predict(novo_caso[features])[0]

print("\n" + "=" * 45)
print("TESTE DO NOVO PACIENTE (CASO DO ENUNCIADO):")
print("=" * 45)
print(f"Probabilidade de Pico de Risco: {risco_previsto:.2f}")
print(f"Classificação: {'PICO DE RISCO (1)' if classe_prevista == 1 else 'SEM RISCO (0)'}")
print("=" * 45)

  