import tensorflow as tf
from model import build_model
from preprocess import preprocess_data
import numpy as np

def train_model():
    # Cargar datos de entrenamiento y validación
    train_images, train_labels = preprocess_data('data/train/images/', 'data/train/labels/')
    val_images, val_labels = preprocess_data('data/val/images/', 'data/val/labels/')
    
    # Construir el modelo
    model = build_model(input_shape=(416, 416, 3))
    model.compile(optimizer=tf.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Entrenar el modelo
    model.fit(np.array(train_images), np.array(train_labels), validation_data=(np.array(val_images), np.array(val_labels)), epochs=10)
    
    # Guardar el modelo entrenado
    model.save('models/trained_yolo_model.h5')

if __name__ == "__main__":
    train_model()
