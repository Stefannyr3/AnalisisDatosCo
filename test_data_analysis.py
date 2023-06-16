#Punto4

import unittest
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
import numpy as np

class TestDataAnalysis(unittest.TestCase):

    def setUp(self):
        # Configurar los datos de prueba
        self.data = pd.read_csv("CovidData.csv", delimiter=";")
        self.X = self.data.drop("DATE_DIED", axis=1)
        self.y = pd.to_datetime(self.data["DATE_DIED"], format="%d/%m/%Y", errors="coerce").dt.day

    def test_data_loading(self):
        # Verificar que los datos se carguen correctamente desde el archivo CSV
        self.assertIsNotNone(self.data)
        self.assertEqual(self.data.shape[0], 1048575)  # Ajustar el valor según el tamaño esperado de los datos

    def test_data_preprocessing(self):
        # Verificar que el preprocesamiento de datos se realice correctamente
        data = self.data.copy()
        data["DATE_DIED"] = pd.to_datetime(data["DATE_DIED"], format="%d/%m/%Y", errors="coerce")
        data = data.dropna(subset=["DATE_DIED"])

        # Agrega aquí las pruebas específicas de preprocesamiento que desees realizar
        self.assertIsNotNone(data)
        self.assertEqual(data.shape[0], 76942)  # Ajustar el valor según el tamaño esperado de los datos

        # Verificar que la columna "DATE_DIED" sea de tipo datetime
        self.assertEqual(data["DATE_DIED"].dtype, "datetime64[ns]")

    def test_model_training(self):
        # Verificar que el entrenamiento del modelo se realice correctamente
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),  # Imputación de datos
            ("scaler", StandardScaler()),
            ("regression", LinearRegression())
        ])
        X = self.X.dropna()  # Eliminar filas con valores NaN en las características
        y = self.y[X.index]  # Obtener los valores correspondientes en y

        imputer = SimpleImputer(strategy="mean")  # Imputador para la variable objetivo
        y = imputer.fit_transform(y.values.reshape(-1, 1)).flatten()

        pipeline.fit(X, y)
        self.assertIsNotNone(pipeline.named_steps['regression'].coef_)

    def test_predictions(self):
        # Verificar que las predicciones sean coherentes y tengan el formato esperado
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),  # Imputación de datos
            ("scaler", StandardScaler()),
            ("regression", LinearRegression())
        ])
        X = self.X.dropna()  # Eliminar filas con valores NaN en las características
        y = self.y[X.index]  # Obtener los valores correspondientes en y

        imputer = SimpleImputer(strategy="mean")  # Imputador para la variable objetivo
        y = imputer.fit_transform(y.values.reshape(-1, 1)).flatten()

        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)

        self.assertIsNotNone(y_pred)
        self.assertEqual(len(y_pred), len(y))

if __name__ == '__main__':
    unittest.main()
