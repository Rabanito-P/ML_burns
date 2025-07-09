import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import __version__ as sklearn_version
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- Configurar advertencias opcionalmente ---
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Cargar archivo corregido (usa ruta raw o slashes normales) ---
df = pd.read_csv('../data/employ25_3-4-5.csv', low_memory=False)
df.columns = df.columns.str.strip().str.upper()

# --- Variables a procesar (sin REGION, ESTRATO ni C205) ---
columnas_preprocesamiento = [
    'C207', 'C208', 'C366', 'C377',
    'C303', 'C310', 'C312', 'C317',
    'INGTRABW', 'C338',
    'C333', 'C335',
    'C375_1', 'C375_2', 'C375_3', 'C375_4', 'C375_5', 'C375_6',
    'C330',
    # Omitimos: 'ESTRATO', 'REGION', 'C205'
]

# --- Limpieza básica ---
df = df.replace(r'^\s*$', np.nan, regex=True)
df = df.replace([99, 999, 9999, 999999, 999998, 998, '99', '9999', '999999'], np.nan)
df = df[[col for col in columnas_preprocesamiento if col in df.columns]].copy()

# --- Identificar tipos de variables ---
categ = [col for col in df.columns if df[col].nunique() <= 10 and df[col].dtype in [np.int64, np.float64]]
num = [col for col in df.columns if col not in categ]

# --- OneHotEncoder compatibilidad de versión ---
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

# --- Calcular inercia y silhouette para distintos k ---
inertias = []
silhouettes = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X, labels))

# --- Graficar ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, marker='o')
plt.title('Método del Codo (Inercia)')
plt.xlabel('Número de clústeres (k)')
plt.ylabel('Inercia')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouettes, marker='o', color='green')
plt.title('Silhouette Score')
plt.xlabel('Número de clústeres (k)')
plt.ylabel('Score')
plt.grid()

plt.tight_layout()
plt.show()
