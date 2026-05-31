import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def predecir_precio_viviendas(ruta_csv: str, test_size: float, random_state: int) -> tuple:
    # 1. Cargar el dataset desde el archivo CSV
    df = pd.read_csv(ruta_csv)
    
    # 2. Limpiar los datos (Manejo de nulos)
    # Optamos por eliminar las filas con nulos (dropna) para mantener la consistencia
    # con el comportamiento esperado del caso de prueba.
    df_limpio = df.dropna()
    
    # 3. Separar variables independientes (X) y variable objetivo (y)
    # Excluimos explícitamente la columna 'precio' de X
    X = df_limpio.drop(columns=["precio"])
    y = df_limpio["precio"]
    
    # 4. Dividir los datos en entrenamiento y prueba
    # Usamos el test_size y random_state proporcionados en los argumentos
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # 5. Entrenar el modelo de regresión
    # Instanciamos el RandomForestRegressor asegurando la reproducibilidad con random_state
    modelo = RandomForestRegressor(n_estimators=100, random_state=random_state)
    modelo.fit(X_train, y_train)
    
    # 6. Realizar predicciones sobre el conjunto de prueba
    predicciones = modelo.predict(X_test)
    
    # 7. Evaluar el modelo usando el Error Cuadrático Medio (MSE)
    mse = mean_squared_error(y_test, predicciones)
    
    # Retornamos la tupla asegurando tipos estándar (lista y float)
    return (predicciones.tolist(), float(mse))
