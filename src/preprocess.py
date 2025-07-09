"""
Módulo de preprocesamiento para datos de empleo.
Incluye limpieza, imputación condicional y transformaciones específicas.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import logging
import os
import scipy.sparse as sp

# Configurar logger para el módulo
logger = logging.getLogger(__name__)


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Elimina columnas especificadas sin variabilidad o con alto porcentaje de nulos.
    
    Parameters
    ----------
    columns_to_drop : list
        Lista de nombres de columnas a eliminar.
    """
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        # Filtrar solo las columnas existentes
        cols_to_drop = [col for col in self.columns_to_drop if col in X_copy.columns]
        logger.info(f"Eliminando {len(cols_to_drop)} columnas: {cols_to_drop}")
        return X_copy.drop(columns=cols_to_drop, errors='ignore')


class ConditionalEmploymentImputer(BaseEstimator, TransformerMixin):
    """
    Imputación inteligente basada en situación laboral (ocupado/desempleado).
    
    Parameters
    ----------
    employment_col : str, default='C303'
        Columna que indica el estado laboral (1=ocupado, 2=desempleado).
    employment_vars : list, optional
        Lista de variables que dependen del estado laboral.
    """
    def __init__(self, employment_col='C303', employment_vars=None):
        self.employment_col = employment_col
        self.employment_vars = employment_vars or [
            'C310', 'C312', 'C317', 'C333', 'C330', 
            'C338', 'INGTRABW', 'C335'
        ]
        self.impute_values_ = {}
        
    def fit(self, X, y=None):
        """
        Aprende los valores de imputación a partir de los datos.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Datos de entrenamiento.
        y : Ignored
            No se usa, presente por compatibilidad con API.
            
        Returns
        -------
        self : object
            Retorna el objeto mismo.
        """
        # Verificar que la columna de empleo existe
        if self.employment_col not in X.columns:
            logger.warning(f"Columna de empleo {self.employment_col} no encontrada, saltando imputación condicional")
            return self
            
        # Normalizar nombres de columnas
        X_cols = [col.upper() for col in X.columns]
        X_upper = X.copy()
        X_upper.columns = X_cols
            
        # Filtrar solo variables existentes
        vars_to_impute = [col for col in self.employment_vars 
                          if col.upper() in X_upper.columns]
        
        if not vars_to_impute:
            logger.warning("No se encontraron variables de empleo válidas para imputar")
            return self
        
        # Calcular valores de imputación solo para ocupados (C303=1)
        employed_data = X_upper[X_upper[self.employment_col.upper()] == 1]
        
        for col in vars_to_impute:
            col_upper = col.upper()
            if col_upper in employed_data.columns:
                # Para numéricas: mediana, para categóricas: moda
                if pd.api.types.is_numeric_dtype(employed_data[col_upper]):
                    self.impute_values_[col_upper] = employed_data[col_upper].median()
                else:
                    if not employed_data[col_upper].empty:
                        self.impute_values_[col_upper] = employed_data[col_upper].mode()[0]
        
        logger.info(f"Valores de imputación calculados para {len(self.impute_values_)} variables")
        return self
    
    def transform(self, X):
        """
        Aplica la imputación condicional a los datos.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Datos a transformar.
            
        Returns
        -------
        pandas.DataFrame
            Datos con valores imputados.
        """
        X_copy = X.copy()
        
        # Normalizar nombres de columnas
        X_cols = [col.upper() for col in X_copy.columns]
        X_copy.columns = X_cols
        
        # Verificar que la columna de empleo existe
        if self.employment_col.upper() not in X_copy.columns:
            logger.warning(f"Columna de empleo {self.employment_col} no encontrada en transform")
            return X_copy
            
        for col, value in self.impute_values_.items():
            if col not in X_copy.columns:
                continue
                
            # Desempleados: asignar 0 (numérico) o '0' (categórico)
            mask_unemployed = X_copy[self.employment_col.upper()] == 2
            replacement = 0 if pd.api.types.is_numeric_dtype(X_copy[col]) else '0'
            X_copy.loc[mask_unemployed, col] = replacement
            
            # Ocupados con nulos: imputar con valor calculado
            mask_employed_na = (X_copy[self.employment_col.upper()] == 1) & X_copy[col].isna()
            X_copy.loc[mask_employed_na, col] = value
            
        return X_copy


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Crea características compuestas y transformaciones especiales.
    
    Parameters
    ----------
    create_disability_feature : bool, default=True
        Si se debe crear la variable TIENE_DISCAPACIDAD combinando C375_1 a C375_6.
    log_transform_income : bool, default=True
        Si se debe transformar logarítmicamente la variable ingtrabw.
    """
    def __init__(self, create_disability_feature=True, log_transform_income=True):
        self.create_disability_feature = create_disability_feature
        self.log_transform_income = log_transform_income
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Crear diccionario case-insensitive de columnas
        column_dict = {col.upper(): col for col in X_copy.columns}
        
        # 1. Combinar variables de discapacidad en una sola
        if self.create_disability_feature:
            # Buscar columnas C375_1 a C375_6 en cualquier caso
            disc_cols = []
            for i in range(1, 7):
                disc_key = f'C375_{i}'.upper()
                if disc_key in column_dict:
                    disc_cols.append(column_dict[disc_key])
            
            if disc_cols:
                logger.info(f"Creando variable TIENE_DISCAPACIDAD con {len(disc_cols)} columnas")
                X_copy['TIENE_DISCAPACIDAD'] = X_copy[disc_cols].max(axis=1)
                X_copy['TIENE_DISCAPACIDAD'] = X_copy['TIENE_DISCAPACIDAD'].replace({2: 0})
                X_copy = X_copy.drop(columns=disc_cols)
        
        # 2. Transformación logarítmica de ingresos
        if self.log_transform_income:
            income_variations = ['INGTRABW', 'ingtrabw', 'INGTOT', 'INGRESO']
            income_col_found = False
            
            for var in income_variations:
                if var.upper() in column_dict:
                    income_col = column_dict[var.upper()]
                    logger.info(f"Aplicando transformación logarítmica a {income_col}")
                    # Convertir a numérico y manejar errores
                    X_copy[income_col] = pd.to_numeric(X_copy[income_col], errors='coerce')
                    # Evitar log(0) usando log1p
                    X_copy['LOG_INGTRABW'] = np.log1p(X_copy[income_col])
                    # No eliminar la columna original en este caso
                    # X_copy = X_copy.drop(columns=income_col)
                    income_col_found = True
                    break
            
            if not income_col_found:
                logger.warning("No se encontró ninguna columna de ingresos para transformación logarítmica")
        
        return X_copy


class SimpleImputerCustom(BaseEstimator, TransformerMixin):
    """
    Imputación simple con estrategias personalizadas por columna.
    
    Parameters
    ----------
    strategies : dict
        Diccionario {columna: estrategia} donde estrategia puede ser 'median', 'mode', 
        o un valor específico para imputar.
    """
    def __init__(self, strategies=None):
        self.strategies = strategies or {}
        self.impute_values_ = {}
        
    def fit(self, X, y=None):
        for col, strategy in self.strategies.items():
            if col not in X.columns:
                logger.warning(f"Columna {col} no encontrada, saltando imputación")
                continue
                
            if strategy == 'median':
                self.impute_values_[col] = X[col].median()
            elif strategy == 'mode':
                if not X[col].empty:
                    self.impute_values_[col] = X[col].mode()[0]
            else:  # Valor fijo
                self.impute_values_[col] = strategy
                
        logger.info(f"Valores de imputación calculados para {len(self.impute_values_)} variables")
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col, value in self.impute_values_.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].fillna(value)
        return X_copy


def load_data(file_path, missing_values=None):
    """
    Carga datos desde un archivo CSV y realiza limpieza básica.
    
    Parameters
    ----------
    file_path : str
        Ruta al archivo CSV.
    missing_values : list, optional
        Lista de valores a considerar como nulos.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame con los datos cargados y limpios.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
        
    logger.info(f"Cargando datos desde {file_path}")
    
    # Valores por defecto para missing values
    missing_values = missing_values or [99, 999, 9999, 999999, 999998, 998, '99', '9999', '999999']
    
    # Cargar datos
    try:
        df = pd.read_csv(file_path, low_memory=False)
        
        # Normalizar nombres de columnas (mayúsculas)
        df.columns = df.columns.str.strip().str.upper()
        
        # Limpiar valores vacíos
        df = df.replace(r'^\s*$', np.nan, regex=True)
        
        # Reemplazar códigos de missing por NaN
        df = df.replace(missing_values, np.nan)
        
        logger.info(f"Datos cargados correctamente: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
        
    except Exception as e:
        logger.error(f"Error al cargar datos: {str(e)}")
        raise


def save_processed_data(data, output_path):
    """
    Guarda datos procesados como CSV o NPY dependiendo del tipo.
    
    Parameters
    ----------
    data : pandas.DataFrame, numpy.ndarray, or scipy.sparse.csr_matrix
        Datos a guardar.
    output_path : str
        Ruta de salida para el archivo.
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Verificar el tipo de dato
        if isinstance(data, pd.DataFrame):
            # Si es un DataFrame, guardarlo como CSV
            data.to_csv(output_path, index=False)
            logger.info(f"DataFrame guardado en {output_path}")
            
        elif sp.issparse(data):
            # Si es una matriz dispersa, guardarla como archivo NPY
            npy_path = output_path.replace('.csv', '.npz')
            sp.save_npz(npy_path, data)
            logger.info(f"Matriz dispersa guardada en {npy_path}")
            
            # También guardar un archivo de metadatos con información básica
            meta_path = output_path.replace('.csv', '_meta.txt')
            with open(meta_path, 'w') as f:
                f.write(f"Tipo: {type(data)}\n")
                f.write(f"Forma: {data.shape}\n")
                f.write(f"Densidad: {data.nnz / (data.shape[0] * data.shape[1]):.4f}\n")
            logger.info(f"Metadatos guardados en {meta_path}")
            
        elif isinstance(data, np.ndarray):
            # Si es un array de NumPy, guardarlo como NPY
            npy_path = output_path.replace('.csv', '.npy')
            np.save(npy_path, data)
            logger.info(f"Array de NumPy guardado en {npy_path}")
            
        else:
            # Para otros tipos, intentar convertir a DataFrame
            logger.warning(f"Tipo de datos no reconocido: {type(data)}. Intentando convertir a DataFrame.")
            
            # Para matrices densas o arrays convertibles
            try:
                if hasattr(data, 'toarray'):
                    df = pd.DataFrame(data.toarray())
                else:
                    df = pd.DataFrame(data)
                df.to_csv(output_path, index=False)
                logger.info(f"Datos convertidos y guardados en {output_path}")
            except Exception as e:
                logger.error(f"No se pudo convertir a DataFrame: {str(e)}")
                # Guardar como pickle como último recurso
                import pickle
                pickle_path = output_path.replace('.csv', '.pkl')
                with open(pickle_path, 'wb') as f:
                    pickle.dump(data, f)
                logger.info(f"Datos guardados como pickle en {pickle_path}")
                
    except Exception as e:
        logger.error(f"Error al guardar datos procesados: {str(e)}")
        raise


def verify_columns(file_path):
    """
    Verifica qué columnas existen en el archivo CSV y su tipo de datos.
    """
    try:
        import pandas as pd
        df = pd.read_csv(file_path, nrows=5)
        
        # Imprimir información sobre columnas
        print("\nColumnas disponibles en el archivo CSV:")
        for col in df.columns:
            print(f"- '{col}' (tipo: {df[col].dtype})")
            
        # Verificar columnas específicas importantes
        important_cols = ['ingtrabw', 'C208', 'C303', 'C207']
        print("\nVerificación de columnas importantes:")
        for col in important_cols:
            col_upper = col.upper()
            found = False
            for df_col in df.columns:
                if df_col.upper() == col_upper:
                    print(f"+ '{df_col}' existe")
                    found = True
                    break
            
            if not found:
                print(f"- '{col}' NO existe")
                
                # Buscar columnas similares
                similar_cols = [c for c in df.columns if col.lower() in c.lower()]
                if similar_cols:
                    print(f"  Posibles alternativas: {similar_cols}")
    
    except Exception as e:
        print(f"Error al verificar columnas: {str(e)}")