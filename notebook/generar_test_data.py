import pandas as pd

# 1. Ruta del archivo
archivo_origen = r'..\data\employ25_3-4-5.csv'  # Ajusta si es necesario

# 2. Variables exactas a tomar
variables_finales = [
    'C207', 'C208', 'C366', 'C377', 'C303',
    'C310', 'C312', 'C317', 'C338',  # ✅ incluye C338
    'ingtrabw', 'C333', 'C330', 'C335',
    'C375_1', 'C375_2', 'C375_3', 'C375_4', 'C375_5', 'C375_6'
]


# 3. Cargar el CSV original
df = pd.read_csv(archivo_origen, low_memory=False)
df.columns = df.columns.str.strip().str.lower()  # Normaliza columnas a minúsculas

# 4. Convertir lista también a minúsculas para hacer match
variables_existentes = [col.lower() for col in variables_finales if col.lower() in df.columns]
df_filtrado = df[variables_existentes].copy()

# 5. Eliminar filas totalmente vacías
df_filtrado = df_filtrado.dropna(how='all')

# 6. Tomar 30 filas aleatorias
df_muestra = df_filtrado.sample(n=150, random_state=42)

# 7. Renombrar 'ingtrabw' a 'INGTRABW'
if 'ingtrabw' in df_muestra.columns:
    df_muestra.rename(columns={'ingtrabw': 'INGTRABW'}, inplace=True)

# 8. Guardar CSV
df_muestra.to_csv('test_data_temp.csv', index=False)
print("✅ test_data.csv generado correctamente con 30 registros.")
