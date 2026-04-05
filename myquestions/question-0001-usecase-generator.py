import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

def generar_caso_de_uso():
    # Componente aleatorio: número de filas y tamaño de prueba
    n_rows = np.random.randint(100, 500)
    test_size_rnd = round(np.random.uniform(0.15, 0.4), 2)
    
    # Generación de datos aleatorios
    df_aleatorio = pd.DataFrame({
        'num_1': np.random.rand(n_rows) * 100,
        'num_2': np.random.randn(n_rows) * 10,
        'cat_1': np.random.choice(['A', 'B', 'C'], n_rows), # Columna no numérica para filtrar
        'target_price': np.random.rand(n_rows) * 5000 + 1000
    })
    
    # Inyectar algunos valores nulos aleatoriamente en una columna numérica
    indices_nulos = np.random.choice(df_aleatorio.index, size=int(n_rows*0.05), replace=False)
    df_aleatorio.loc[indices_nulos, 'num_1'] = np.nan
    
    # 1. Definir el input
    input_dict = {
        'df': df_aleatorio,
        'target_col': 'target_price',
        'test_size': test_size_rnd
    }
    
    # 2. Calcular el output esperado según las reglas del problema
    df_num = df_aleatorio.select_dtypes(include=[np.number])
    df_clean = df_num.dropna()
    X = df_clean.drop(columns=['target_price'])
    y = df_clean['target_price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_rnd, random_state=42)
    
    modelo = DecisionTreeRegressor(random_state=42)
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)
    rmse_esperado = float(np.sqrt(mean_squared_error(y_test, predicciones)))
    
    return input_dict, rmse_esperado
