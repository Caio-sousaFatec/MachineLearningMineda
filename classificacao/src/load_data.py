import pandas as pd

def carregar_dados(caminho: str) -> pd.DataFrame:
    """Carrega os dados CSV do Kaggle."""
    df = pd.read_csv(caminho)
    print(f"Base carregada: {df.shape[0]} linhas e {df.shape[1]} colunas.")
    return df
