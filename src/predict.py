import tensorflow as tf
from model import build_model
from PIL import Image
import numpy as np

def predict_image(image_path):
    model = tf.keras.models.load_model('models/trained_yolo_model.h5')
    
    # Cargar y preprocesar la imagen
    img = Image.open(image_path)
    img = img.resize((416, 416))  # Redimensionar imagen a 416x416
    img = np.array(img) / 255.0  # Normalizar
    img = np.expand_dims(img, axis=0)  # Agregar batch dimension
    
    # Hacer la predicción
    predictions = model.predict(img)
    return predictions

if __name__ == "__main__":
    image_path = 'path_to_image.jpg'  # Ruta de la imagen
    result = predict_image(image_path)
    print(result)
