"""
Script auxiliar para inspeccionar datos CSV.
"""

import pandas as pd
import numpy as np
import os
import sys

def inspect_csv(file_path):
    """
    Inspecciona un archivo CSV para entender su estructura.
    """
    if not os.path.exists(file_path):
        print(f"Error: No se encuentra el archivo {file_path}")
        return
        
    print(f"Inspeccionando archivo: {file_path}")
    
    try:
        # Cargar el CSV
        df = pd.read_csv(file_path, low_memory=False)
        
        # Información básica
        print("\n=== INFORMACIÓN BÁSICA ===")
        print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
        print(f"Uso de memoria: {df.memory_usage().sum() / 1024**2:.2f} MB")
        
        # Nombres de columnas
        print("\n=== NOMBRES DE COLUMNAS ===")
        print(f"Primeras 10 columnas: {list(df.columns[:10])}")
        
        # Columnas específicas importantes
        important_cols = ['C208', 'C303', 'C207', 'INGTRABW', 'ingtrabw']
        print("\n=== VERIFICACIÓN DE COLUMNAS IMPORTANTES ===")
        for col in important_cols:
            if col in df.columns:
                print(f"✓ '{col}' existe")
                # Mostrar algunos valores
                values = df[col].dropna().values[:5]
                print(f"  Ejemplos: {values}")
            else:
                print(f"✗ '{col}' NO existe")
        
        # Búsqueda de columnas similares a INGTRABW
        print("\n=== BÚSQUEDA DE COLUMNA DE INGRESOS ===")
        income_cols = [col for col in df.columns 
                       if 'ing' in col.lower() or 'trab' in col.lower()]
        if income_cols:
            print(f"Posibles columnas de ingresos: {income_cols}")
            for col in income_cols:
                print(f"- {col}: {df[col].dropna().values[:5]}")
        
        # Estadísticas de valores nulos
        print("\n=== ANÁLISIS DE VALORES NULOS ===")
        null_counts = df.isna().sum()
        high_null_cols = null_counts[null_counts > df.shape[0] * 0.5].index.tolist()
        print(f"Columnas con >50% nulos: {high_null_cols}")
        
    except Exception as e:
        print(f"Error durante la inspección: {str(e)}")

if __name__ == "__main__":
    # Si se pasa un archivo como argumento, lo usa; si no, usa un valor por defecto
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/employ25_3-4-5.csv"
    inspect_csv(file_path)