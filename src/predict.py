"""
Módulo para realizar predicciones con modelos entrenados.
Permite aplicar el modelo a nuevos datos y visualizar resultados.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
from .preprocess import load_data
from .model import load_model

# Configurar logger para el módulo
logger = logging.getLogger(__name__)


def predict_clusters(model_path, data_path, output_path=None, visualize=False):
    """
    Aplica un modelo entrenado a nuevos datos para asignar clusters.
    
    Parameters
    ----------
    model_path : str
        Ruta al modelo guardado.
    data_path : str
        Ruta a los datos a predecir.
    output_path : str, optional
        Ruta para guardar resultados. Si None, no se guarda.
    visualize : bool, default=False
        Si se debe visualizar la distribución de clusters.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame con datos originales y asignación de cluster.
    """
    try:
        # 1. Cargar datos y modelo
        logger.info("Cargando datos y modelo para predicción")
        df = load_data(data_path)
        model = load_model(model_path)
        
        # 2. Realizar predicciones
        logger.info("Aplicando modelo para asignar clusters")
        cluster_labels = model.predict(df)
        
        # 3. Añadir etiquetas al DataFrame original
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = cluster_labels
        
        # 4. Visualizar distribución de clusters
        if visualize:
            plt.figure(figsize=(10, 6))
            plt.hist(cluster_labels, bins=range(max(cluster_labels) + 2), alpha=0.7)
            plt.title('Distribución de Clusters')
            plt.xlabel('Cluster')
            plt.ylabel('Frecuencia')
            plt.xticks(range(max(cluster_labels) + 1))
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Guardar visualización si se especificó un directorio de salida
            if output_path:
                output_dir = os.path.dirname(output_path)
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, 'cluster_distribution.png'))
                
            plt.show()
        
        # 5. Guardar resultados si se especificó una ruta
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_with_clusters.to_csv(output_path, index=False)
            logger.info(f"Resultados guardados en {output_path}")
        
        return df_with_clusters
        
    except Exception as e:
        logger.error(f"Error durante la predicción: {str(e)}")
        raise


def summarize_clusters(df_with_clusters, key_features=None, output_path=None):
    """
    Genera resumen estadístico de cada cluster.
    
    Parameters
    ----------
    df_with_clusters : pandas.DataFrame
        DataFrame con columna 'cluster'.
    key_features : list, optional
        Lista de características clave a incluir en el resumen.
        Si None, se usan todas las columnas numéricas.
    output_path : str, optional
        Ruta para guardar resumen. Si None, no se guarda.
        
    Returns
    -------
    dict
        Diccionario con resumen estadístico por cluster.
    """
    try:
        if 'cluster' not in df_with_clusters.columns:
            raise ValueError("El DataFrame no contiene la columna 'cluster'")
        
        # Determinar características a resumir
        if key_features is None:
            # Usar columnas numéricas excepto 'cluster'
            key_features = [col for col in df_with_clusters.select_dtypes(
                include=['number']).columns if col != 'cluster']
        
        # Filtrar solo las características disponibles
        key_features = [col for col in key_features if col in df_with_clusters.columns]
        
        if not key_features:
            raise ValueError("No se encontraron características válidas para resumir")
        
        # Generar resumen por cluster
        clusters = df_with_clusters['cluster'].unique()
        summary = {}
        
        for cluster in sorted(clusters):
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster]
            cluster_size = len(cluster_data)
            cluster_pct = (cluster_size / len(df_with_clusters)) * 100
            
            # Estadísticas básicas para cada característica
            feature_stats = {}
            for feature in key_features:
                if feature in cluster_data.columns:
                    stats = cluster_data[feature].describe()
                    feature_stats[feature] = {
                        'mean': stats['mean'],
                        'median': cluster_data[feature].median(),
                        'std': stats['std'],
                        'min': stats['min'],
                        'max': stats['max']
                    }
            
            # Guardar en diccionario de resumen
            summary[f"cluster_{cluster}"] = {
                'size': cluster_size,
                'percentage': cluster_pct,
                'features': feature_stats
            }
        
        # Guardar resumen en formato CSV si se especificó una ruta
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Crear DataFrame para formato tabular
            rows = []
            for cluster_name, cluster_data in summary.items():
                for feature_name, feature_stats in cluster_data['features'].items():
                    row = {
                        'cluster': cluster_name.split('_')[1],
                        'feature': feature_name,
                        'mean': feature_stats['mean'],
                        'median': feature_stats['median'],
                        'std': feature_stats['std'],
                        'min': feature_stats['min'],
                        'max': feature_stats['max']
                    }
                    rows.append(row)
            
            summary_df = pd.DataFrame(rows)
            summary_df.to_csv(output_path, index=False)
            logger.info(f"Resumen de clusters guardado en {output_path}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error durante la generación del resumen: {str(e)}")
        raise