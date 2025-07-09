import numpy as np
import scipy.sparse as sp

# Cargar la matriz dispersa
sparse_matrix = sp.load_npz('C:\\Users\\angel\\Downloads\\ML_burns-feature-employ\\data\\processed.npz')

# Información básica sobre la matriz
print(f"Tipo de matriz: {type(sparse_matrix)}")
print(f"Dimensiones: {sparse_matrix.shape}")
print(f"Número de elementos no-cero: {sparse_matrix.nnz}")
print(f"Densidad: {sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1]):.6f}")

# Ver primeras filas de la matriz convertida a formato denso
# (Cuidado si es muy grande)
dense_sample = sparse_matrix[:5].toarray()
print("\nPrimeras 5 filas:")
print(dense_sample)

# Si quieres ver las características específicas, necesitas cargar metadatos adicionales
# Si hay un archivo de metadatos, puedes cargarlo así:
try:
    with open('C:\\Users\\angel\\Downloads\\ML_burns-feature-employ\\data\\processed_meta.txt', 'r') as f:
        metadata = f.read()
        print("\nMetadatos:")
        print(metadata)
except FileNotFoundError:
    print("\nNo se encontró archivo de metadatos.")

# También puedes guardar la matriz en formato CSV si prefieres verla así
# (Cuidado, esto puede ser muy grande si tu matriz es grande)
import pandas as pd
sample_df = pd.DataFrame(sparse_matrix[:100].toarray())  # Tomar solo las primeras 100 filas para muestra
sample_df.to_csv('C:\\Users\\angel\\Downloads\\ML_burns-feature-employ\\data\\processed_sample.csv', index=False)
print("\nSe ha guardado una muestra de 100 filas en processed_sample.csv")