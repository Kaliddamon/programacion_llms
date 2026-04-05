import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def generar_caso_de_uso():
    # Componente aleatorio: dimensiones y componentes
    n_rows = np.random.randint(100, 300)
    n_sensores = np.random.randint(5, 12)
    n_comp_rnd = np.random.randint(2, n_sensores - 1)
    
    columnas_sensores = [f'sensor_{i}' for i in range(n_sensores)]
    
    # Generación de datos
    df_aleatorio = pd.DataFrame(
        np.random.rand(n_rows, n_sensores) * 50, 
        columns=columnas_sensores
    )
    # Agregar una columna irrelevante
    df_aleatorio['fecha'] = pd.date_range(start='1/1/2026', periods=n_rows)
    
    # 1. Definir el input
    input_dict = {
        'df': df_aleatorio,
        'columnas_features': columnas_sensores,
        'n_components': n_comp_rnd
    }
    
    # 2. Calcular el output esperado
    X = df_aleatorio[columnas_sensores]
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=n_comp_rnd, random_state=42)
    pca.fit(X_scaled)
    varianza_esperada = np.cumsum(pca.explained_variance_ratio_)
    
    return input_dict, varianza_esperada
