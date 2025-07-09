#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML_burns-feature-employ: Sistema de clustering para datos de empleo.

Script principal que orquesta todo el flujo:
1. Preprocesamiento
2. Entrenamiento
3. Evaluación
4. Generación de reportes
"""

import os
import logging
import sys
import argparse
import io
from datetime import datetime
from src.model import train_and_save_model
from src.evaluate import find_optimal_clusters, generate_cluster_report
from src.preprocess import load_data, save_processed_data
from src.pipeline import build_processing_pipeline
from src.predict import predict_clusters, summarize_clusters

# Solución para problemas de codificación
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configuración de logging global
def setup_logging(log_level=logging.INFO):
    """Configura el sistema de logging"""
    # Crear directorio de logs si no existe
    os.makedirs('logs', exist_ok=True)
    
    # Nombre de archivo con timestamp
    log_filename = f"logs/ml_employ_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configurar handler para archivo
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(log_level)
    
    # Configurar handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Formato de logs
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configurar logger raíz
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Limpiar handlers existentes
    if root_logger.handlers:
        root_logger.handlers.clear()
        
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

# Configuración global del logger
logger = setup_logging()

def verify_csv_structure(file_path):
    """Verifica la estructura del CSV y muestra información útil"""
    try:
        import pandas as pd
        
        logger.info(f"Verificando estructura del archivo: {file_path}")
        df = pd.read_csv(file_path, nrows=5)
        
        logger.info(f"Dimensiones: {df.shape}")
        logger.info(f"Columnas (primeras 10): {df.columns.tolist()[:10]}...")
        
        # Verificar columnas importantes - sin usar símbolos Unicode
        for col in ['C208', 'C303', 'C207', 'ingtrabw', 'INGTRABW']:
            if col in df.columns:
                logger.info(f"[ENCONTRADA] Columna '{col}'")
            else:
                logger.info(f"[NO ENCONTRADA] Columna '{col}'")
        
        # Buscar columnas de ingresos
        income_cols = [col for col in df.columns 
                      if 'ing' in col.lower() or 'trab' in col.lower()]
        if income_cols:
            logger.info(f"Columnas de ingresos encontradas: {income_cols}")
            
        return df
    
    except Exception as e:
        logger.error(f"Error al verificar CSV: {str(e)}")
        return None

def parse_args():
    """Parsea argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='ML_burns-feature-employ: Clustering para datos de empleo'
    )
    
    parser.add_argument(
        '--data', 
        default='data/employ25_3-4-5.csv',
        help='Ruta al archivo de datos (default: data/employ25_3-4-5.csv)'
    )
    
    parser.add_argument(
        '--clusters', 
        type=int, 
        help='Número de clusters (si no se especifica, se busca el óptimo)'
    )
    
    parser.add_argument(
        '--skip-evaluation', 
        action='store_true',
        help='Omitir evaluación de número óptimo de clusters'
    )
    
    parser.add_argument(
        '--output-dir', 
        default='output',
        help='Directorio para archivos de salida (default: output)'
    )
    
    return parser.parse_args()


def ensure_directories():
    """Crea directorios necesarios si no existen"""
    dirs = ['data', 'models', 'reports', 'output', 'logs']
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def main():
    # No necesitamos configurar el logging de nuevo aquí, ya está configurado globalmente
    logger.info("Iniciando ML_burns-feature-employ")
    
    # Parsear argumentos
    args = parse_args()
    
    # Crear directorios necesarios
    ensure_directories()
    
    # Configuración de rutas
    DATA_DIR = 'data'
    RAW_DATA = args.data
    PROCESSED_DATA = os.path.join(DATA_DIR, 'processed.csv')
    MODEL_PATH = 'models/kmeans_model.pkl'
    REPORT_PATH = 'reports/clusters_summary.md'
    
    try:
        # ---------------------------
        # 0. Verificación preliminar
        # ---------------------------
        logger.info("Verificando estructura del CSV...")
        df_preview = verify_csv_structure(RAW_DATA)
        if df_preview is None:
            logger.error("Verificación del CSV falló. Abortando proceso.")
            return 1
            
        # ---------------------------
        # 1. Preprocesamiento
        # ---------------------------
        logger.info("Paso 1: Preprocesando datos...")

        # Cargar datos
        df = load_data(RAW_DATA)
        
        # Pipeline de preprocesamiento
        processing_pipe = build_processing_pipeline()
        
        # Aplicar pipeline
        logger.info("Aplicando pipeline de preprocesamiento...")
        processed_data = processing_pipe.fit_transform(df)
        
        # Guardar datos procesados
        save_processed_data(processed_data, PROCESSED_DATA)
        logger.info(f"Datos procesados guardados en {PROCESSED_DATA}")
        
        # ---------------------------
        # 2. Selección de hiperparámetros (opcional)
        # ---------------------------
        optimal_k = 4  # Valor por defecto
        
        if args.clusters:
            # Usar el valor especificado por el usuario
            optimal_k = args.clusters
            logger.info(f"Usando número de clusters especificado: k={optimal_k}")
        elif not args.skip_evaluation:
            # Buscar el número óptimo de clusters
            logger.info("Paso 2: Buscando número óptimo de clusters...")
            optimal_k = find_optimal_clusters(
                RAW_DATA, 
                output_dir='reports'
            )
            logger.info(f"Número óptimo de clusters encontrado: k={optimal_k}")
        
        # ---------------------------
        # 3. Entrenamiento del modelo
        # ---------------------------
        logger.info(f"Paso 3: Entrenando modelo con k={optimal_k}...")
        model = train_and_save_model(RAW_DATA, MODEL_PATH, n_clusters=optimal_k)
        logger.info(f"Modelo entrenado y guardado en {MODEL_PATH}")
        
        # ---------------------------
        # 4. Generación de reportes
        # ---------------------------
        logger.info("Paso 4: Generando reporte de clusters...")
        
        # Obtener nombres de características
        try:
            # Intentar extraer los nombres de características del pipeline
            if hasattr(model, 'named_steps') and 'processing' in model.named_steps:
                if hasattr(model.named_steps['processing'], 'named_steps'):
                    feature_names = []
                    
                    # Nombres de características numéricas
                    feature_names.extend(['C208', 'LOG_INGTRABW'])
                    
                    # Variables binarias
                    feature_names.extend(['C303', 'C207', 'TIENE_DISCAPACIDAD'])
                    
                    # Variables categóricas one-hot encoded
                    if ('column_transformer' in model.named_steps['processing'].named_steps and 
                        hasattr(model.named_steps['processing'].named_steps['column_transformer'], 'transformers_')):
                        
                        for name, transformer, columns in model.named_steps['processing'].named_steps['column_transformer'].transformers_:
                            if name == 'cat' and hasattr(transformer, 'get_feature_names_out'):
                                cat_features = transformer.get_feature_names_out()
                                feature_names.extend(list(cat_features))
                                
            else:
                # Si no podemos obtener los nombres, usamos valores genéricos
                logger.warning("No se pudieron obtener nombres de características del modelo")
                feature_names = [f"Feature_{i}" for i in range(20)]
                
        except Exception as e:
            logger.warning(f"Error al obtener nombres de características: {str(e)}")
            feature_names = None
        
        # Generar reporte
        generate_cluster_report(model, feature_names, REPORT_PATH)
        logger.info(f"Reporte de clusters generado en {REPORT_PATH}")
        
        # ---------------------------
        # 5. Predicciones de ejemplo
        # ---------------------------
        logger.info("Paso 5: Realizando predicciones de ejemplo...")
        
        # Predecir usando los mismos datos de entrenamiento (ejemplo)
        predictions_path = os.path.join(args.output_dir, 'predictions.csv')
        predictions = predict_clusters(
            MODEL_PATH, 
            RAW_DATA,
            output_path=predictions_path,
            visualize=True
        )
        logger.info(f"Predicciones guardadas en {predictions_path}")
        
        # Resumir clusters
        summary_path = os.path.join(args.output_dir, 'cluster_summary.csv')
        summarize_clusters(
            predictions,
            output_path=summary_path
        )
        logger.info(f"Resumen de clusters guardado en {summary_path}")
        
        logger.info("Proceso completado exitosamente!")
        
    except Exception as e:
        logger.error(f"Error durante la ejecución: {str(e)}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())