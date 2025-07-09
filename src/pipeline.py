"""
Construye el pipeline completo de preprocesamiento + clustering.
Define la estructura del flujo de datos desde su entrada hasta el modelo final.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, 
    OneHotEncoder,
    FunctionTransformer
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
import logging
import pandas as pd
from .preprocess import (
    ColumnDropper,
    ConditionalEmploymentImputer,
    FeatureEngineer,
    SimpleImputerCustom
)

# Configurar logger para el módulo
logger = logging.getLogger(__name__)


class ColumnValidator(BaseEstimator, TransformerMixin):
    """Valida y ajusta las columnas para asegurar compatibilidad"""
    def __init__(self, config):
        self.config = config
        self.column_mapping = {}  # Mapeo de nombres de columnas originales a normalizados
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        """
        Identifica columnas necesarias sin modificar los nombres originales.
        """
        # Crear diccionario case-insensitive para buscar columnas
        column_dict = {col.upper(): col for col in X.columns}
        X_copy = X.copy()
        
        # Actualizar configuración con columnas reales
        
        # 1. Columnas numéricas
        real_numeric_cols = []
        for col in ['C208']:
            if col.upper() in column_dict:
                real_numeric_cols.append(column_dict[col.upper()])
        
        # Buscar columna de ingresos (ingtrabw)
        income_variations = ['INGTRABW', 'INGTRAB', 'INGRESO']
        for var in income_variations:
            if var.upper() in column_dict:
                income_col = column_dict[var.upper()]
                logger.info(f"Encontrada columna de ingresos: {income_col}")
                real_numeric_cols.append(income_col)
                break
        
        # 2. Columnas binarias
        real_binary_cols = []
        for col in ['C303', 'C207', 'TIENE_DISCAPACIDAD']:
            if col.upper() in column_dict:
                real_binary_cols.append(column_dict[col.upper()])
        
        # 3. Columnas categóricas
        real_categorical_cols = []
        for col in ['C310', 'C312', 'C317', 'C333', 'C330', 'C338', 'C335']:
            if col.upper() in column_dict:
                real_categorical_cols.append(column_dict[col.upper()])
        
        # Actualizar configuración
        self.config['real_numeric_cols'] = real_numeric_cols
        self.config['real_binary_cols'] = real_binary_cols
        self.config['real_categorical_cols'] = real_categorical_cols
        
        logger.info(f"Columnas numéricas reales: {real_numeric_cols}")
        logger.info(f"Columnas binarias reales: {real_binary_cols}")
        logger.info(f"Columnas categóricas reales: {real_categorical_cols}")
        
        return X_copy


# Definir DynamicColumnTransformer como clase de nivel superior
class DynamicColumnTransformer(BaseEstimator, TransformerMixin):
    """Transformador de columnas que usa los nombres reales identificados"""
    def __init__(self, config):
        self.config = config
        self.column_transformer_ = None
        
    def fit(self, X, y=None):
        # Obtener las columnas reales a partir del validator
        if not hasattr(self.config, 'real_numeric_cols'):
            # Si no se ejecutó el validator, usar columnas por defecto
            self.column_transformer_ = ColumnTransformer([
                ('num', StandardScaler(), ['C208']),
                ('bin', 'passthrough', ['C303', 'C207']),
                ('cat', OneHotEncoder(handle_unknown='ignore'), 
                 ['C310', 'C312', 'C317', 'C333', 'C330', 'C338', 'C335'])
            ], remainder='drop')
        else:
            # Usar columnas identificadas por el validator
            transformers = []
            
            if self.config['real_numeric_cols']:
                transformers.append(
                    ('num', StandardScaler(), self.config['real_numeric_cols'])
                )
            
            if self.config['real_binary_cols']:
                transformers.append(
                    ('bin', 'passthrough', self.config['real_binary_cols'])
                )
            
            if self.config['real_categorical_cols']:
                transformers.append(
                    ('cat', OneHotEncoder(handle_unknown='ignore'), 
                     self.config['real_categorical_cols'])
                )
            
            self.column_transformer_ = ColumnTransformer(
                transformers, remainder='drop'
            )
            
        return self.column_transformer_.fit(X, y)
        
    def transform(self, X):
        return self.column_transformer_.transform(X)


def build_processing_pipeline(config=None):
    """
    Construye el pipeline de preprocesamiento de datos.
    
    Parameters
    ----------
    config : dict, optional
        Configuración personalizada para el pipeline.
        
    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline completo de preprocesamiento.
    """
    # Configuración por defecto
    default_config = {
        'columns_to_drop': [
            'C342', 'C334', 'C313',  # Alto % nulos
            'REGION', 'ESTRATO', 'C205'  # Sin variabilidad
        ],
        'imputation_strategies': {
            'C208': 'median'  # Edad: imputar con mediana
        }
    }
    
    # Combinar con configuración personalizada
    if config:
        for key, value in config.items():
            default_config[key] = value
    
    cfg = default_config
    
    # Construir pipeline de preprocesamiento con validación previa
    logger.info("Construyendo pipeline de preprocesamiento")
    
    # Pipeline inicial de preprocesamiento (antes de columnas específicas)
    pre_processing_pipeline = Pipeline([
        ('validator', ColumnValidator(cfg)),
        ('drop_columns', ColumnDropper(cfg['columns_to_drop'])),
        ('conditional_imputer', ConditionalEmploymentImputer()),
        ('feature_engineer', FeatureEngineer()),
        ('simple_imputer', SimpleImputerCustom(cfg['imputation_strategies']))
    ])
    
    # Pipeline completo de preprocesamiento
    processing_pipeline = Pipeline([
        ('pre_processing', pre_processing_pipeline),
        ('column_transformer', DynamicColumnTransformer(cfg))
    ])
    
    return processing_pipeline


def build_full_pipeline(n_clusters=4, random_state=42, config=None):
    """
    Construye pipeline completo (preprocesamiento + clustering).
    
    Parameters
    ----------
    n_clusters : int, default=4
        Número de clusters para KMeans.
    random_state : int, default=42
        Semilla aleatoria para reproducibilidad.
    config : dict, optional
        Configuración personalizada para el pipeline.
        
    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline completo de preprocesamiento + clustering.
    """
    logger.info(f"Construyendo pipeline completo con {n_clusters} clusters")
    
    # Obtener pipeline de preprocesamiento
    processing = build_processing_pipeline(config)
    
    # Construir pipeline completo
    full_pipeline = Pipeline([
        ('processing', processing),
        ('clustering', KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        ))
    ])
    
    return full_pipeline