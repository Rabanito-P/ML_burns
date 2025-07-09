# src/pipeline.py
"""
Construye el pipeline completo de preprocesamiento + clustering
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, 
    OneHotEncoder,
    FunctionTransformer
)
from sklearn.cluster import KMeans
from .preprocess import (
    ColumnDropper,
    ConditionalEmploymentImputer,
    FeatureEngineer,
    SimpleImputerCustom
)

def build_processing_pipeline():
    """Construye el pipeline de preprocesamiento"""
    # 1. Columnas a eliminar (sin variabilidad o alto % nulos)
    columns_to_drop = [
        'C342', 'C334', 'C313',  # Alto % nulos
        'REGION', 'ESTRATO', 'C205'  # Sin variabilidad
    ]
    
    # 2. Estrategias de imputación simple
    simple_impute_strategies = {
        'C208': 'median'  # Edad: imputar con mediana
    }
    
    # 3. Definir transformers para tipos específicos
    preprocessor = ColumnTransformer([
        # Variables numéricas (escalar)
        ('num', StandardScaler(), ['C208', 'LOG_INGTRABW']),
        
        # Variables binarias (pasar sin cambios)
        ('bin', 'passthrough', ['C303', 'C207', 'TIENE_DISCAPACIDAD']),
        
        # Variables categóricas (one-hot encoding)
        ('cat', OneHotEncoder(handle_unknown='ignore'), 
         ['C310', 'C312', 'C317', 'C333', 'C330', 'C338', 'C335'])
    ])
    
    # Pipeline completo de preprocesamiento
    processing_pipeline = Pipeline([
        ('drop_columns', ColumnDropper(columns_to_drop)),
        ('conditional_imputer', ConditionalEmploymentImputer()),
        ('feature_engineer', FeatureEngineer()),
        ('simple_imputer', SimpleImputerCustom(simple_impute_strategies)),
        ('preprocessor', preprocessor)
    ])
    
    return processing_pipeline

def build_full_pipeline(n_clusters=4):
    """Construye pipeline completo (preprocesamiento + clustering)"""
    processing = build_processing_pipeline()
    
    full_pipeline = Pipeline([
        ('processing', processing),
        ('clustering', KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        ))
    ])
    
    return full_pipeline