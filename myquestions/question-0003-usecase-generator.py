import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def generar_caso_de_uso():
    # Componente aleatorio: filas, clusters y datos
    n_rows = np.random.randint(150, 400)
    k_rnd = np.random.randint(2, 6)
    
    comportamiento_cols = ['gasto_total', 'visitas_mes', 'tiempo_sesion']
    
    # Simular datos agrupados usando distribuciones normales desplazadas
    data = np.vstack([
        np.random.randn(n_rows//2, 3) * 5 + 10,
        np.random.randn(n_rows - n_rows//2, 3) * 5 + 50
    ])
    
    df_aleatorio = pd.DataFrame(data, columns=comportamiento_cols)
    df_aleatorio['id_cliente'] = range(n_rows) # Columna a ignorar
    
    # Inyectar NaNs para forzar limpieza
    nulos_idx = np.random.choice(df_aleatorio.index, 8, replace=False)
    df_aleatorio.loc[nulos_idx, 'visitas_mes'] = np.nan
    
    # 1. Definir el input
    input_dict = {
        'df': df_aleatorio,
        'columnas_comportamiento': comportamiento_cols,
        'n_clusters': k_rnd
    }
    
    # 2. Calcular el output esperado
    df_filtrado = df_aleatorio[comportamiento_cols].dropna()
    X = df_filtrado.to_numpy()
    
    kmeans = KMeans(n_clusters=k_rnd, random_state=42, n_init='auto')
    etiquetas = kmeans.fit_predict(X)
    score_esperado = float(silhouette_score(X, etiquetas))
    
    return input_dict, score_esperado
