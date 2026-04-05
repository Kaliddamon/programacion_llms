import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import OPTICS

def generar_caso_de_uso():
    # 1. Componente aleatorio: dimensiones de la matriz y el parámetro min_muestras
    n_rows = np.random.randint(100, 300)
    n_cols = np.random.randint(2, 6)
    min_muestras_rnd = np.random.randint(3, 10)

    # Generamos la matriz X aleatoria
    X = np.random.rand(n_rows, n_cols) * 100

    # Diccionario de entrada
    input_dict = {
        'X': X,
        'min_muestras': min_muestras_rnd
    }

    # 2. Calcular el output esperado
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    optics = OPTICS(min_samples=min_muestras_rnd)
    optics.fit(X_scaled)
    
    # Extraer las distancias de alcanzabilidad
    output_esperado = optics.reachability_ # np.ndarray unidimensional

    return input_dict, output_esperado
