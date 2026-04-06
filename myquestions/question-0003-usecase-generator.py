import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

def generar_caso_de_uso_log_verosimilitud_promedio_gmm():
    # 1. Componente aleatorio: tamaño del conjunto de datos
    n_rows = np.random.randint(200, 600)
    n_cols = np.random.randint(2, 5)

    # Generamos datos aleatorios simulando agrupaciones densas
    data = np.random.randn(n_rows, n_cols) * np.random.randint(1, 5) + np.random.randint(-10, 10, size=n_cols)
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_cols)])

    # Diccionario de entrada
    input_dict = {
        'df': df
    }

    # 2. Calcular el output esperado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(X_scaled)
    
    # El método score calcula la log-verosimilitud promedio
    output_esperado = float(gmm.score(X_scaled))

    return input_dict, output_esperado
