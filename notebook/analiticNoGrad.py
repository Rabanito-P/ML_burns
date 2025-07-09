import pandas as pd
import numpy as np

# Cargar archivo
df = pd.read_csv('..\data\employ25_3-4-5.csv', low_memory=False)

# Lista de columnas
columnas_preprocesamiento = [
    'C207', 'C208', 'C366', 'C377',
    'C303', 'C310', 'C312', 'C317',
    'INGTRABW', 'C338',
    'C333', 'C335',
    'C375_1', 'C375_2', 'C375_3', 'C375_4', 'C375_5', 'C375_6',
    'ESTRATO', 'REGION',
    'C205', 'C330',
    'C342', 'C334', 'C313'
]

# Limpiar nombres de columnas
df.columns = df.columns.str.strip().str.upper()
columnas_preprocesamiento = [col.upper() for col in columnas_preprocesamiento]

# Filtrar columnas existentes
cols_existentes = [col for col in columnas_preprocesamiento if col in df.columns]
cols_faltantes = list(set(columnas_preprocesamiento) - set(cols_existentes))

print(f"✅ Columnas encontradas: {len(cols_existentes)}")
print(f"❌ Columnas faltantes: {cols_faltantes}")

df_sub = df[cols_existentes].copy()

# Reemplazar valores vacíos y codificados como nulos
valores_missing = [99, 999, 9999, 999999, 999998, 998, '99', '9999', '999999']
df_sub.replace(r'^\s*$', np.nan, regex=True, inplace=True)
df_sub.replace(valores_missing, np.nan, inplace=True)

# Reporte de nulos
print("\n📉 Nulos por columna:")
nulos = df_sub.isnull().sum()
porcentaje = (nulos / len(df_sub) * 100).round(2)
nulos_df = pd.DataFrame({
    'Nulos': nulos,
    'Porcentaje (%)': porcentaje
})
print(nulos_df.sort_values(by='Porcentaje (%)', ascending=False))

# Reporte de tipo de datos y valores únicos
print("\n🔎 Tipos de datos y ejemplos únicos:")
for col in cols_existentes:
    tipo = df_sub[col].dtype
    unicos = df_sub[col].dropna().unique()
    print(f"{col}: {tipo}, {len(unicos)} únicos → {unicos[:5]}")
