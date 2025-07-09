Hello
ESTRATO: object, 1 únicos → ['1']
REGION: int64, 1 únicos → [1]
C205: object, 1 únicos → ['2']
---

El corazón de **ML\_EMPLOY** es el manejo cuidadoso y sistemático de los datos. Todo comienza en la carpeta `data/`, donde se almacena el archivo bruto `employ25_3-4-5.csv` —la Encuesta Permanente de Empleo tal cual la provee el INEI—y su diccionario de variables (`diccionario.pdf`). A partir de este origen, el script de preprocesamiento (`src/preprocess.py`) elimina valores constantes, reemplaza los códigos de “missing” (99, 9999, etc.) por `NaN`, realiza la imputación de medias o modas, codifica las variables categóricas con OneHotEncoder y escala las numéricas con StandardScaler. El resultado intermedio se guarda como `data/processed.csv`, listo para modelado.

Una vez limpios y normalizados, se construye un **pipeline** modular en `src/pipeline.py` que integra todas las transformaciones de columnas (ColumnTransformer) y culmina en un objeto KMeans. Esta arquitectura facilita mantener separadas las etapas de limpieza y clustering, a la vez que garantiza reproducibilidad: cualquier dato nuevo que pase por el pipeline sufre exactamente las mismas transformaciones antes de asignarse a un clúster.

La selección del hiperparámetro clave (el número de clústeres, k) se gestiona en `src/evaluate.py`. Allí, el sistema ejecuta KMeans para distintos valores de k (por ejemplo, de 2 a 10), calcula la inercia (método del codo) y el Silhouette Score, y genera dos gráficos que se guardan en `reports/silhouette_codo.png`. De este análisis, el valor óptimo (k = 4) se determina de forma objetiva y se documenta para el entrenamiento final.

Con k ya establecido, `src/model.py` carga `processed.csv`, aplica el mismo pipeline y ajusta el modelo KMeans con el número de clústeres escogido. Luego serializa el pipeline completo (preprocesamiento + KMeans) en `models/kmeans_model.pkl` usando joblib, de modo que cualquier aplicación posterior pueda cargar directamente un único objeto y predecir clústeres sin redefinir ninguna transformación.

Por último, la carpeta `src/predict.py` contiene la rutina para aplicar el modelo guardado a nuevos registros: lee datos nuevos, ejecuta el pipeline y devuelve la etiqueta de clúster asignada. Los perfiles resultantes se describen en `reports/clusters_summary.md`, donde se analizan las características promedio de cada grupo (edad, ingresos, tipo de empleo, etc.). De esta forma, todo el flujo —desde datos crudos hasta resultados interpretables— está organizado, documentado y automatizado, listo para que cualquier colaborador clone el repositorio, instale dependencias (`pip install -r requirements.txt`) y ejecute `python main.py` para reproducir el proyecto completo.
