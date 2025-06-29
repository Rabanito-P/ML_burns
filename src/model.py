import tensorflow as tf
import tensorflow_hub as hub

def load_yolo_model():
    """Carga un modelo preentrenado de YOLOv5 desde TensorFlow Hub."""
    yolo_model_url = "https://www.kaggle.com/models/kaggle/yolo-v5/"  # URL de un modelo preentrenado
    yolo_model = hub.load(yolo_model_url)
    return yolo_model

def build_model(input_shape=(416, 416, 3)):
    """Función para construir el modelo de YOLO."""
    model = load_yolo_model()
    return model
