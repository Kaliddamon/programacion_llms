import numpy as np

def evaluar_clasificacion(y_true, y_pred):
    # Asegurarnos de que las entradas sean arreglos de NumPy
    # Esto nos permite usar operaciones lógicas vectorizadas (&)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 1. Calcular los componentes de la matriz de confusión
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    # 2. Calcular métricas con manejo de división por cero
    denom_precision = TP + FP
    precision = TP / denom_precision if denom_precision > 0 else 0.0
    
    denom_recall = TP + FN
    recall = TP / denom_recall if denom_recall > 0 else 0.0
    
    denom_f1 = precision + recall
    f1_score = 2 * (precision * recall) / denom_f1 if denom_f1 > 0 else 0.0
    
    # 3. Retornar el diccionario con los tipos de datos nativos de Python (int/float)
    return {
        'matriz_confusion': [[int(TN), int(FP)], [int(FN), int(TP)]],
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score)
    }
