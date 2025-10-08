from src.load_data import carregar_dados
from src.preprocess import tratar_dados
from src.train import treinar_modelo
from src.evaluate import avaliar_modelo

def main():
    df = carregar_dados("data/raw/dataset.csv")
    X, y = tratar_dados(df, coluna_alvo="target")

    model, X_test, y_test = treinar_modelo(X, y, "models/best_model.pkl")
    avaliar_modelo(model, X_test, y_test)

if __name__ == "__main__":
    main()
