# src/preprocess.py
"""
Módulo de preprocesamiento para datos de empleo
Incluye limpieza, imputación condicional y transformaciones específicas
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnDropper(BaseEstimator, TransformerMixin):
    """Elimina columnas especificadas sin variabilidad o con alto porcentaje de nulos"""
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors='ignore')

class ConditionalEmploymentImputer(BaseEstimator, TransformerMixin):
    """Imputación inteligente basada en situación laboral (ocupado/desempleado)"""
    def __init__(self, employment_col='C303'):
        self.employment_col = employment_col
        self.impute_values_ = {}
        
    def fit(self, X, y=None):
        # Variables que dependen del estado laboral
        employment_vars = ['C310', 'C312', 'C317', 'C333', 'C330', 'C338', 'INGTRABW', 'C335']
        
        # Calcular valores de imputación solo para ocupados (C303=1)
        employed_data = X[X[self.employment_col] == 1]
        
        for col in employment_vars:
            if col in employed_data.columns:
                # Para numéricas: mediana, para categóricas: moda
                if pd.api.types.is_numeric_dtype(employed_data[col]):
                    self.impute_values_[col] = employed_data[col].median()
                else:
                    self.impute_values_[col] = employed_data[col].mode()[0]
        return self
    
    def transform(self, X):
        X = X.copy()
        for col, value in self.impute_values_.items():
            # Desempleados: asignar 0 (numérico) o '0' (categórico)
            mask_unemployed = X[self.employment_col] == 2
            replacement = 0 if pd.api.types.is_numeric_dtype(X[col]) else '0'
            X.loc[mask_unemployed, col] = replacement
            
            # Ocupados con nulos: imputar con valor calculado
            mask_employed_na = (X[self.employment_col] == 1) & X[col].isna()
            X.loc[mask_employed_na, col] = value
        return X

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Crea características compuestas y transformaciones especiales"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # 1. Combinar variables de discapacidad en una sola
        disc_cols = [f'C375_{i}' for i in range(1, 7)]
        if all(col in X.columns for col in disc_cols):
            X['TIENE_DISCAPACIDAD'] = X[disc_cols].max(axis=1)
            X['TIENE_DISCAPACIDAD'] = X['TIENE_DISCAPACIDAD'].replace({2: 0})
            X = X.drop(columns=disc_cols)
        
        # 2. Transformación logarítmica de ingresos
        if 'INGTRABW' in X.columns:
            # Convertir a numérico y manejar errores
            X['INGTRABW'] = pd.to_numeric(X['INGTRABW'], errors='coerce')
            X['LOG_INGTRABW'] = np.log1p(X['INGTRABW'])
            X = X.drop(columns='INGTRABW')
        
        return X

class SimpleImputerCustom(BaseEstimator, TransformerMixin):
    """Imputación simple con estrategias personalizadas por columna"""
    def __init__(self, strategies):
        """
        strategies: dict {col: strategy}
        Estrategias: 'median', 'mode', o valor fijo
        """
        self.strategies = strategies
        
    def fit(self, X, y=None):
        self.impute_values_ = {}
        for col, strategy in self.strategies.items():
            if strategy == 'median':
                self.impute_values_[col] = X[col].median()
            elif strategy == 'mode':
                self.impute_values_[col] = X[col].mode()[0]
            else:  # Valor fijo
                self.impute_values_[col] = strategy
        return self
    
    def transform(self, X):
        for col, value in self.impute_values_.items():
            X[col] = X[col].fillna(value)
        return X

# Función de utilidad para cargar datos
def load_data(file_path):
    """Carga datos desde un archivo CSV"""
    return pd.read_csv(file_path)

# Función de utilidad para guardar datos procesados
def save_processed_data(df, output_path):
    """Guarda datos procesados como CSV"""
    df.to_csv(output_path, index=False)