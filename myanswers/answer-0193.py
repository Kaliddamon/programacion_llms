import pandas as pd
import numpy as np

def resumir_por_segmento(df, col_segmento, col_cliente, col_monto):
    # 1. Calcular el monto total por cliente
    totales_por_cliente = df.groupby(col_cliente)[col_monto].sum().reset_index(name='monto_total_cliente')
    
    # 2. Filtrar clientes cuyo total supere la mediana global de todos los totales
    mediana_global = totales_por_cliente['monto_total_cliente'].median()
    clientes_validos = totales_por_cliente[totales_por_cliente['monto_total_cliente'] > mediana_global][col_cliente]
    
    # Filtrar el DataFrame original para quedarnos solo con las transacciones de los clientes válidos
    df_filtrado = df[df[col_cliente].isin(clientes_validos)]
    
    # Manejo de caso extremo: si ningún cliente supera la mediana (ej. todos tienen el mismo monto)
    if df_filtrado.empty:
        return pd.DataFrame()
    
    # 3. Agrupar por segmento y calcular estadísticas
    # Calculamos también la suma total por segmento temporalmente para el paso 4
    resumen = df_filtrado.groupby(col_segmento).agg(
        total_transacciones=(col_monto, 'count'),
        monto_promedio=(col_monto, 'mean'),
        monto_maximo=(col_monto, 'max'),
        clientes_unicos=(col_cliente, 'nunique'),
        suma_segmento=(col_monto, 'sum') # Columna auxiliar para el porcentaje
    ).reset_index()
    
    # 4. Agregar columna 'pct_del_total' sobre la suma global de los clientes filtrados
    suma_global_filtrada = resumen['suma_segmento'].sum()
    resumen['pct_del_total'] = (resumen['suma_segmento'] / suma_global_filtrada) * 100
    
    # Eliminamos la columna auxiliar ya que no fue solicitada en la salida final
    resumen = resumen.drop(columns=['suma_segmento'])
    
    # 5. Ordenar de mayor a menor por monto_promedio
    resumen = resumen.sort_values(by='monto_promedio', ascending=False).reset_index(drop=True)
    
    return resumen
