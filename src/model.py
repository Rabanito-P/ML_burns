"""
Entrenamiento y guardado del modelo final.
Maneja la creación, entrenamiento y persistencia de modelos.
"""

import joblib
import os
import logging
from .pipeline import build_full_pipeline
from .preprocess import load_data

# Configurar logger para el módulo
logger = logging.getLogger(__name__)


def train_and_save_model(data_path, model_path, n_clusters=4, config=None):
    """
    Entrena un modelo de clustering y lo guarda en disco.
    
    Parameters
    ----------
    data_path : str
        Ruta a datos crudos.
    model_path : str
        Ruta para guardar modelo.
    n_clusters : int, default=4
        Número de clusters para KMeans.
    config : dict, optional
        Configuración personalizada para el pipeline.
        
    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline entrenado.
    """
    # 1. Cargar datos
    try:
        logger.info(f"Cargando datos para entrenamiento desde {data_path}")
        df = load_data(data_path)
    except Exception as e:
        logger.error(f"Error al cargar datos: {str(e)}")
        raise
    
    # 2. Construir pipeline
    try:
        logger.info(f"Construyendo pipeline con {n_clusters} clusters")
        pipeline = build_full_pipeline(n_clusters=n_clusters, config=config)
    except Exception as e:
        logger.error(f"Error al construir pipeline: {str(e)}")
        raise
    
    # 3. Entrenar modelo
    try:
        logger.info("Entrenando modelo...")
        pipeline.fit(df)
        logger.info("Entrenamiento completado")
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {str(e)}")
        raise
    
    # 4. Guardar modelo
    try:
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        joblib.dump(pipeline, model_path)
        logger.info(f"Modelo guardado exitosamente en {model_path}")
    except Exception as e:
        logger.error(f"Error al guardar modelo: {str(e)}")
        raise
    
    return pipeline


def load_model(model_path):
    """
    Carga un modelo previamente entrenado.
    
    Parameters
    ----------
    model_path : str
        Ruta al modelo guardado.
        
    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline cargado.
    """
    try:
        logger.info(f"Cargando modelo desde {model_path}")
        pipeline = joblib.load(model_path)
        return pipeline
    except FileNotFoundError:
        logger.error(f"No se encontró el archivo de modelo en {model_path}")
        raise
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {str(e)}")
        raise