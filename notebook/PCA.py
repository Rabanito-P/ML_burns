import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn import __version__ as sklearn_version

# --- Cargar datos ---
df = pd.read_csv('../data/employ25_3-4-5.csv', low_memory=False)
df.columns = df.columns.str.strip().str.upper()

# --- Variables relevantes (excluyendo REGION, C205, ESTRATO) ---
columnas_preprocesamiento = [
    'C207', 'C208', 'C366', 'C377',
    'C303', 'C310', 'C312', 'C317',
    'INGTRABW', 'C338',
    'C333', 'C335',
    'C375_1', 'C375_2', 'C375_3', 'C375_4', 'C375_5', 'C375_6',
    'C330'
]

# --- Limpieza ---
df = df.replace(r'^\s*$', np.nan, regex=True)
df = df.replace([99, 999, 9999, 999999, 999998, 998, '99', '9999', '999999'], np.nan)
df = df[[col for col in columnas_preprocesamiento if col in df.columns]].copy()

# --- Tipos de variables ---
categ = [col for col in df.columns if df[col].nunique() <= 10 and df[col].dtype in [np.int64, np.float64]]
num = [col for col in df.columns if col not in categ]

# --- OneHotEncoder versión compatible ---
major = int(sklearn_version.split('.')[0])
minor = int(sklearn_version.split('.')[1])
if major >= 1 and minor >= 2:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
else:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

# --- Pipelines de preprocesamiento ---
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', encoder)
])
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, num),
    ('cat', categorical_pipeline, categ)
])

# --- Transformar datos ---
X = preprocessor.fit_transform(df)

# --- PCA: todos los componentes ---
pca = PCA()
X_pca = pca.fit_transform(X)

# --- Varianza explicada acumulada ---
var_acumulada = np.cumsum(pca.explained_variance_ratio_)
n_componentes = np.arange(1, len(var_acumulada)+1)

# --- Crear tabla
tabla_pca = pd.DataFrame({
    'Número de componentes': n_componentes,
    'Varianza explicada acumulada': var_acumulada.round(4)
})

print(tabla_pca)

# --- Graficar curva
plt.figure(figsize=(8, 5))
plt.plot(n_componentes, var_acumulada, marker='o')
plt.title('PCA: Varianza explicada acumulada')
plt.xlabel('Número de componentes')
plt.ylabel('Varianza acumulada')
plt.grid(True)
plt.tight_layout()
plt.show()
