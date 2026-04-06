import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.covariance import EllipticEnvelope

def generar_caso_de_uso_conteo_anomalias_elipticas():
    # 1. Componente aleatorio: número de registros, columnas y nivel de contaminación
    n_rows = np.random.randint(100, 500)
    n_cols = np.random.randint(2, 6)
    contaminacion_rnd = round(np.random.uniform(0.01, 0.15), 3)

    # Generamos datos normales (Gaussianos)
    data = np.random.normal(loc=0, scale=1, size=(n_rows, n_cols))
    df = pd.DataFrame(data, columns=[f'var_{i}' for i in range(n_cols)])

    # Inyectamos valores nulos de forma aleatoria para forzar la imputación
    num_nulos = int(n_rows * 0.05)
    for _ in range(num_nulos):
        fila = np.random.randint(0, n_rows)
        col = np.random.randint(0, n_cols)
        df.iloc[fila, col] = np.nan

    # Diccionario de entrada
    input_dict = {
        'df': df,
        'contaminacion': contaminacion_rnd
    }

    # 2. Calcular el output esperado según la lógica del problema
    imputer = SimpleImputer(strategy='mean')
    df_imputado = imputer.fit_transform(df)

    modelo = EllipticEnvelope(contamination=contaminacion_rnd, random_state=42)
    modelo.fit(df_imputado)
    predicciones = modelo.predict(df_imputado)
    
    # Contar las anomalías (-1)
    output_esperado = int(np.sum(predicciones == -1))

    return input_dict, output_esperado
