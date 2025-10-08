from sklearn.metrics import classification_report, confusion_matrix

def avaliar_modelo(model, X_test, y_test):
    """Avalia o desempenho do modelo."""
    y_pred = model.predict(X_test)
    print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))
    print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))
