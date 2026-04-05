import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def generar_caso_de_uso():
    # Componente aleatorio: cantidad de datos y nivel de contaminación
    n_rows = np.random.randint(300, 1000)
    contaminacion_rnd = round(np.random.uniform(0.02, 0.12), 3)
    
    # Generar datos normales
    datos_normales = np.random.randn(n_rows, 4) * 2 + 10
    
    # Inyectar anomalías manualmente en la matriz para que el modelo las detecte
    n_anomalias = int(n_rows * contaminacion_rnd)
    indices_anomalos = np.random.choice(n_rows, n_anomalias, replace=False)
    datos_normales[indices_anomalos] = np.random.randn(n_anomalias, 4) * 20 + 50
    
    columnas = ['monto', 'velocidad', 'distancia', 'frecuencia']
    df_aleatorio = pd.DataFrame(datos_normales, columns=columnas)
    
    # 1. Definir el input
    input_dict = {
        'df': df_aleatorio,
        'contamination': contaminacion_rnd
    }
    
    # 2. Calcular el output esperado
    X = df_aleatorio.to_numpy()
    modelo_iso = IsolationForest(contamination=contaminacion_rnd, random_state=42)
    predicciones = modelo_iso.fit_predict(X)
    
    # Filtrar solo los datos "sanos" (1)
    df_sano = df_aleatorio[predicciones == 1]
    
    # Calcular promedios por columna
    medias_esperadas = df_sano.mean().to_numpy()
    
    return input_dict, medias_esperadas
