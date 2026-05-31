import pandas as pd
import numpy as np
import random

def calcular_puntualidad_ferroviaria(df, linea_col, retraso_col, umbral_min):
    # 1. Eliminar las filas donde retraso_col sea nula
    df_clean = df.dropna(subset=[retraso_col]).copy()
    
    # 2. Crear columna booleana 'puntual' (estrictamente menor al umbral)
    df_clean["puntual"] = df_clean[retraso_col] < umbral_min
    
    # 3. Agrupar por linea_col y calcular tasa de puntualidad y retraso promedio
    # Al sacar la media ('mean') de una columna booleana, Pandas calcula automáticamente la proporción (0.0 a 1.0)
    grouped = df_clean.groupby(linea_col).agg(
        tasa_puntualidad=("puntual", "mean"),
        retraso_promedio=(retraso_col, "mean")
    ).reset_index()
    
    # 4. Usar numpy para calcular la mediana global y evaluar si se supera
    mediana_global = np.median(df_clean[retraso_col].values)
    grouped["supera_mediana"] = grouped["retraso_promedio"] > mediana_global
    
    # 5. Ordenar de mayor a menor tasa_puntualidad y filtrar las columnas requeridas
    columnas_salida = [linea_col, "tasa_puntualidad", "retraso_promedio", "supera_mediana"]
    df_resultado = (
        grouped[columnas_salida]
        .sort_values(by="tasa_puntualidad", ascending=False)
        .reset_index(drop=True)
    )
    
    return df_resultado
