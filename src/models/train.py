import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path


class TitanicTrainer:
    """
    Clase para entrenar, evaluar y guardar un modelo ML con Titanic.
    """

    def __init__(self, model=None, test_size=0.2, random_state=42):
        self.model = model if model else LogisticRegression(max_iter=500)
        self.test_size = test_size
        self.random_state = random_state
        self.metrics = {}

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Entrena el modelo con train/test split"""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_val)

        # Guardamos métricas
        self.metrics["accuracy"] = accuracy_score(y_val, y_pred)
        self.metrics["report"] = classification_report(y_val, y_pred, output_dict=True)

        return self.model, self.metrics

    def get_metrics(self):
        """Devuelve métricas después del entrenamiento"""
        return self.metrics

    def save_model(self, path: str = "models/trained_model.pkl"):
        """Guarda el modelo entrenado en disco"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)  # crea carpeta si no existe
        joblib.dump(self.model, path)
        print(f"✅ Modelo guardado en {path}")

    @staticmethod
    def load_model(path: str = "models/trained_model.pkl"):
        """Carga un modelo entrenado desde disco"""
        return joblib.load(path)
