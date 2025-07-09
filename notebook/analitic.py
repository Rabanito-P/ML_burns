import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar tus datos (ajusta el path si lo necesitas)
df = pd.read_csv('employ25_3-4-5.csv')

# Variables finales que deseas explorar
variables_finales = [
    'C207', 'C208', 'C366', 'C377',
    'C303', 'C310', 'C312', 'C313', 'C317',
    'ingtrabw', 'C338', 'C342',
    'C333', 'C334', 'C335',
    'C375_1', 'C375_2', 'C375_3', 'C375_4', 'C375_5', 'C375_6',
    'C205', 'C330'
]

# Filtrar solo esas columnas
df_sub = df[variables_finales]

# Cantidad total de registros
print("✅ Total de registros:", len(df_sub))

# Información general del DataFrame
print("\n🧾 Información del DataFrame:")
print(df_sub.info())

# Conteo de valores nulos
print("\n🔎 Valores nulos por variable:")
print(df_sub.isnull().sum())

# Porcentaje de valores nulos
print("\n📉 Porcentaje de nulos:")
print((df_sub.isnull().mean() * 100).round(2))

# Valores únicos por variable
print("\n📊 Valores únicos por variable:")
for col in variables_finales:
    print(f"{col}: {df_sub[col].nunique()} únicos → {df_sub[col].unique()}")

# Distribución de variables numéricas
numericas = df_sub.select_dtypes(include=['float64', 'int64']).columns
df_sub[numericas].hist(figsize=(15, 10), bins=20)
plt.suptitle("Distribuciones de variables numéricas", fontsize=16)
plt.show()

# Distribución de variables categóricas
categoricas = df_sub.select_dtypes(include=['object', 'category']).columns.tolist()
categoricas += [col for col in df_sub.columns if df_sub[col].nunique() <= 10 and col not in numericas]

for col in categoricas:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df_sub, x=col)
    plt.title(f"Distribución de: {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
