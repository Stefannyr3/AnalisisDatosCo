import pandas as pd  # Importar la biblioteca pandas para el manejo de datos tabulares
from sklearn.pipeline import Pipeline  # Importar la clase Pipeline de scikit-learn para crear una tubería de procesamiento
from sklearn.preprocessing import StandardScaler  # Importar la clase StandardScaler para realizar el escalado de características
from sklearn.linear_model import LinearRegression  # Importar la clase LinearRegression para crear un modelo de regresión lineal
from sklearn.metrics import mean_squared_error, r2_score  # Importar las métricas para evaluar el rendimiento del modelo
from sklearn.model_selection import cross_val_score  # Importar la función cross_val_score para realizar validación cruzada
import matplotlib.pyplot as plt  # Importar la biblioteca matplotlib para visualización de gráficos

# Cargar los datos desde un archivo CSV
data = pd.read_csv("CovidData.csv", delimiter=";")

# Convertir la columna "DATE_DIED" a tipo datetime, ignorando los valores incorrectos
data["DATE_DIED"] = pd.to_datetime(data["DATE_DIED"], format="%d/%m/%Y", errors="coerce")

# Eliminar las filas con fechas inválidas
data = data.dropna(subset=["DATE_DIED"])

# Dividir los datos en características (X) y variable objetivo (y)
X = data.drop("DATE_DIED", axis=1)
y = data["DATE_DIED"].dt.day  # Utilizar solo el día como variable objetivo

# 1) Crear una tubería de datos
pipeline = Pipeline([
    ("scaler", StandardScaler()),  # Preprocesamiento de los datos con escalado estándar
    ("regression", LinearRegression())  # Modelo de regresión lineal
])

# Entrenar la tubería de datos (crear el modelo de datos)
pipeline.fit(X, y)

# Verificar los coeficientes del modelo
coeficientes = pipeline.named_steps['regression'].coef_
print("Coeficientes del modelo:", coeficientes)

# 4) Diccionario de datos
data_dict = {
    "AGE": "Edad del paciente",
    "SEX": "Sexo del paciente (1: masculino, 2: femenino)",
    "BMI": "Índice de masa corporal",
    "SMOKER": "Indicador de si el paciente es fumador (0: no fumador, 1: fumador)",
}

# Imprimir el diccionario de datos
print("Diccionario de datos:")
for feature, description in data_dict.items():
    print(f"{feature}: {description}")

# 2) Ejecutar controles de calidad de los datos para asegurar que la tubería funcionó como se esperaba
# Realizar predicciones en los datos de entrenamiento
y_pred = pipeline.predict(X)

# Calcular el error cuadrático medio
mse = mean_squared_error(y, y_pred)
print("Error cuadrático medio (MSE):", mse)

# Calcular el coeficiente de determinación
r2 = r2_score(y, y_pred)
print("Coeficiente de determinación (R²):", r2)

# 5) Realizar validación cruzada con 5 divisiones
scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
mean_mse = -scores.mean()
print("Error cuadrático medio promedio (validación cruzada):", mean_mse)

# Graficar valores reales vs. predicciones
plt.scatter(y, y_pred)
plt.plot([min(y), max(y)], [min(y), max(y)], 'k--', lw=2)
plt.xlabel('Valores reales')
plt.ylabel('Predicciones')
plt.show()



