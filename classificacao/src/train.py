from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

def treinar_modelo(X, y, caminho_modelo: str):
    """Treina e salva um modelo de classificação."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    dump(model, caminho_modelo)
    return model, X_test, y_test
