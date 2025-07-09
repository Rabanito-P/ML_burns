import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn import __version__ as sklearn_version

# 🔇 Ignorar advertencias futuras
warnings.simplefilter(action='ignore', category=FutureWarning)

# 📥 Cargar datos
df = pd.read_csv('../data/employ25_3-4-5.csv', low_memory=False)
df.columns = df.columns.str.strip().str.upper()

# 🔧 Reemplazar valores vacíos y codificados como nulos
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
df.replace([99, 999, 9999, 999999, 999998, 998, '99', '9999', '999999'], np.nan, inplace=True)

# 🧹 Eliminar columnas constantes (sin variabilidad)
df = df.loc[:, df.nunique(dropna=True) > 1]

# 🧠 Detectar columnas categóricas y numéricas automáticamente
categ = [col for col in df.columns if df[col].nunique(dropna=True) <= 10 and df[col].dtype in [np.int64, np.float64, object]]
num = [col for col in df.columns if col not in categ]

# 🛠️ Compatibilidad con OneHotEncoder según versión de sklearn
major = int(sklearn_version.split('.')[0])
minor = int(sklearn_version.split('.')[1])

if major >= 1 and minor >= 2:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
else:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

# 🔄 Preprocesamiento
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

# ✅ Transformar los datos
X = preprocessor.fit_transform(df)

# 📉 PCA completo
pca = PCA()
X_pca = pca.fit_transform(X)
var_acumulada = np.cumsum(pca.explained_variance_ratio_)

# 🔢 Limitar a 129 componentes como máximo
limite = min(129, len(var_acumulada))
n_componentes = np.arange(1, limite + 1)
var_recortada = var_acumulada[:limite]

# 📋 Mostrar tabla
tabla_pca = pd.DataFrame({
    'Número de componentes': n_componentes,
    'Varianza explicada acumulada': var_recortada.round(4)
})
print(tabla_pca)

# 📈 Graficar
plt.figure(figsize=(8, 5))
plt.plot(n_componentes, var_recortada, marker='o', label='Varianza acumulada')
plt.axhline(y=0.8, color='r', linestyle='--', label='80% de varianza')
plt.axhline(y=0.9, color='g', linestyle='--', label='90% de varianza')
plt.title('PCA: Varianza explicada acumulada (máx 129 componentes)')
plt.xlabel('Número de componentes')
plt.ylabel('Varianza acumulada')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
