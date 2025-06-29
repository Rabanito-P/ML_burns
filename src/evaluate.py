import tensorflow as tf
from sklearn.metrics import classification_report
import numpy as np
from preprocess import preprocess_data

def evaluate_model():
    # Cargar el modelo entrenado
    model = tf.keras.models.load_model('models/trained_yolo_model.h5')
    
    # Cargar datos de prueba
    test_images, test_labels = preprocess_data('data/test/images/', 'data/test/labels/')
    
    # Evaluar el modelo
    predictions = model.predict(np.array(test_images))
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels, axis=1)
    
    # Mostrar el reporte de clasificación
    print(classification_report(true_classes, predicted_classes))

if __name__ == "__main__":
    evaluate_model()
