"""
Evaluación del modelo y selección de hiperparámetros.
Incluye herramientas para encontrar el número óptimo de clusters
y generar reportes sobre los clusters encontrados.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import logging
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from .preprocess import load_data
from .pipeline import build_processing_pipeline

# Configurar logger para el módulo
logger = logging.getLogger(__name__)


def find_optimal_clusters(data_path, min_clusters=2, max_clusters=10, output_dir=None):
    """
    Encuentra el número óptimo de clusters usando método del codo y silueta.
    
    Parameters
    ----------
    data_path : str
        Ruta a datos para evaluación.
    min_clusters : int, default=2
        Mínimo número de clusters a evaluar.
    max_clusters : int, default=10
        Máximo número de clusters a evaluar.
    output_dir : str, optional
        Directorio para guardar visualizaciones. Si es None, no se guardan.
        
    Returns
    -------
    int
        Número óptimo de clusters según puntuación de silueta.
    """
    try:
        # 1. Cargar y preprocesar datos
        logger.info(f"Evaluando número óptimo de clusters ({min_clusters}-{max_clusters})")
        df = load_data(data_path)
        processing_pipe = build_processing_pipeline()
        X_processed = processing_pipe.fit_transform(df)
        
        # 2. Evaluar diferentes valores de k
        inertias = []
        silhouette_scores = []
        k_values = range(min_clusters, max_clusters + 1)
        
        for k in k_values:
            logger.info(f"Evaluando k={k}")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_processed)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_processed, cluster_labels))
        
        # 3. Graficar resultados
        plt.figure(figsize=(12, 5))
        
        # Método del codo
        plt.subplot(1, 2, 1)
        plt.plot(k_values, inertias, 'bo-')
        plt.xlabel('Número de clusters (k)')
        plt.ylabel('Inercia')
        plt.title('Método del Codo')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Puntuación de silueta
        plt.subplot(1, 2, 2)
        plt.plot(k_values, silhouette_scores, 'go-')
        plt.xlabel('Número de clusters (k)')
        plt.ylabel('Puntuación de Silueta')
        plt.title('Puntuación de Silueta')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Guardar gráfico si se especificó un directorio
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'elbow_silhouette.png')
            plt.savefig(output_path)
            logger.info(f"Gráfico guardado en {output_path}")
        
        plt.close()
        
        # 4. Recomendar mejor k
        optimal_k = k_values[np.argmax(silhouette_scores)]
        best_score = max(silhouette_scores)
        logger.info(f"K óptimo recomendado: {optimal_k} (silueta={best_score:.3f})")
        
        # Crear un DataFrame con los resultados
        results = pd.DataFrame({
            'k': list(k_values),
            'inertia': inertias,
            'silhouette': silhouette_scores
        })
        
        # Guardar resultados si se especificó un directorio
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            results_path = os.path.join(output_dir, 'cluster_evaluation.csv')
            results.to_csv(results_path, index=False)
            logger.info(f"Resultados de evaluación guardados en {results_path}")
        
        return optimal_k
        
    except Exception as e:
        logger.error(f"Error durante la búsqueda de clusters óptimos: {str(e)}")
        # En caso de error, retornar un valor razonable
        return 4


def generate_cluster_report(model, feature_names=None, report_path=None):
    """
    Genera reporte detallado de interpretación de clusters.
    
    Parameters
    ----------
    model : Pipeline
        Modelo entrenado (pipeline completo).
    feature_names : list, optional
        Lista de nombres de características.
    report_path : str, optional
        Ruta para guardar el reporte. Si None, no se guarda.
        
    Returns
    -------
    str
        Contenido del reporte.
    """
    try:
        # Extraer información de clusters
        if hasattr(model, 'named_steps') and 'clustering' in model.named_steps:
            cluster_centers = model.named_steps['clustering'].cluster_centers_
            n_clusters = len(cluster_centers)
            
            # Si no se especificaron nombres de características, usar índices
            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(cluster_centers.shape[1])]
            
            # Asegurar que la longitud de feature_names coincida con cluster_centers
            if len(feature_names) != cluster_centers.shape[1]:
                logger.warning(
                    f"Longitud de feature_names ({len(feature_names)}) no coincide "
                    f"con dimensión de cluster_centers ({cluster_centers.shape[1]})"
                )
                # Usar solo la cantidad correcta o añadir extras
                if len(feature_names) < cluster_centers.shape[1]:
                    feature_names = feature_names + [
                        f"Feature_{i}" for i in range(len(feature_names), cluster_centers.shape[1])
                    ]
                else:
                    feature_names = feature_names[:cluster_centers.shape[1]]
            
            # Generar reporte
            report = "# Reporte de Clusters\n\n"
            report += f"Análisis de {n_clusters} clusters identificados.\n\n"
            
            # Añadir información sobre tamaño de clusters si está disponible
            if hasattr(model.named_steps['clustering'], 'labels_'):
                labels = model.named_steps['clustering'].labels_
                sizes = np.bincount(labels)
                total = sum(sizes)
                
                report += "## Distribución de clusters\n\n"
                report += "| Cluster | Tamaño | Porcentaje |\n"
                report += "|---------|--------|------------|\n"
                
                for i, size in enumerate(sizes):
                    percentage = (size / total) * 100
                    report += f"| {i} | {size} | {percentage:.2f}% |\n"
                
                report += "\n"
            
            # Describir cada cluster
            report += "## Características de clusters\n\n"
            
            for i, center in enumerate(cluster_centers):
                report += f"### Cluster {i}\n\n"
                
                # Características más importantes (valores altos positivos)
                top_positive = sorted(
                    zip(feature_names, center),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                # Características menos importantes (valores altos negativos)
                top_negative = sorted(
                    zip(feature_names, center),
                    key=lambda x: x[1]
                )[:5]
                
                # Mostrar características positivas
                report += "#### Características distintivas (valores altos)\n\n"
                for feature, value in top_positive:
                    report += f"- **{feature}**: {value:.3f}\n"
                
                # Mostrar características negativas
                report += "\n#### Características distintivas (valores bajos)\n\n"
                for feature, value in top_negative:
                    report += f"- **{feature}**: {value:.3f}\n"
                
                report += "\n"
            
            # Guardar reporte si se especificó una ruta
            if report_path:
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                with open(report_path, 'w') as f:
                    f.write(report)
                logger.info(f"Reporte guardado en {report_path}")
            
            return report
            
        else:
            logger.error("El modelo proporcionado no tiene un paso 'clustering'")
            return "Error: Modelo inválido, no contiene clustering."
            
    except Exception as e:
        logger.error(f"Error al generar reporte de clusters: {str(e)}")
        return f"Error durante la generación del reporte: {str(e)}"