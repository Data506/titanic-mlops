import pandas as pd
from typing import List


class Preprocessor:
    """
    Clase para preprocesar datos del dataset Titanic.
    Incluye manejo de nulos y encoding de variables categóricas.
    """

    def __init__(self, features: List[str]):
        self.features = features
        self.mean_age = None

    def fit(self, df: pd.DataFrame):
        """Calcula estadísticos necesarios para transformar datos"""
        self.mean_age = df["Age"].mean()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforma el dataset aplicando limpieza y encoding"""
        X = df[self.features].copy()

        # Encoding: convertir "Sex" a numérico
        if "Sex" in X.columns:
            X["Sex"] = X["Sex"].map({"male": 0, "female": 1})

        # Rellenar nulos en Age
        if "Age" in X.columns:
            X["Age"] = X["Age"].fillna(self.mean_age)

        return X

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Atajo para fit + transform"""
        self.fit(df)
        return self.transform(df)
