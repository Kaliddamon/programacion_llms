import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

def generar_caso_de_uso():
    # 1. Componente aleatorio: cantidad de datos de entrenamiento/prueba y características
    n_train = np.random.randint(50, 150)
    n_test = np.random.randint(20, 50)
    n_features = np.random.randint(1, 5)

    # Generamos matrices aleatorias
    X_train = np.random.rand(n_train, n_features) * 10
    # y_train como una función no lineal con algo de ruido
    y_train = np.sin(X_train[:, 0]) + np.random.normal(0, 0.1, n_train) 
    X_test = np.random.rand(n_test, n_features) * 10

    # Diccionario de entrada
    input_dict = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test
    }

    # 2. Calcular el output esperado
    gpr = GaussianProcessRegressor(random_state=42)
    gpr.fit(X_train, y_train)
    
    # Predecir pidiendo la desviación estándar
    predicciones, std = gpr.predict(X_test, return_std=True)
    output_esperado = std # np.ndarray unidimensional

    return input_dict, output_esperado
