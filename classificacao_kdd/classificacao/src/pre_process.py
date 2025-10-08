import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def tratar_dados(df: pd.DataFrame, coluna_alvo: str):
    """Limpa e transforma o dataset."""
    # Remover colunas irrelevantes
    df = df.dropna(subset=[coluna_alvo])
    df = df.drop(columns=[col for col in df.columns if 'id' in col.lower()], errors='ignore')

    # Tratar valores nulos
    df = df.fillna(df.median(numeric_only=True))

    # Separar X e y
    X = df.drop(columns=[coluna_alvo])
    y = df[coluna_alvo]

    # Label encoding se alvo for categórico
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # One-hot encoding para variáveis categóricas
    X = pd.get_dummies(X, drop_first=True)

    # Padronização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
