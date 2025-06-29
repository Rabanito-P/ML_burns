import os
import shutil
import random

# Configura los directorios de destino
def split_data(image_dir, label_dir, train_dir, val_dir, test_dir, split_ratio=(0.7, 0.15, 0.15)):
    # Crear las carpetas de destino si no existen
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'labels'), exist_ok=True)
    
    # Listar todas las imágenes en el directorio
    all_images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]  # Ajusta el filtro según tu tipo de imagen
    random.shuffle(all_images)  # Mezclar las imágenes de manera aleatoria
    
    # Calcular el tamaño de cada conjunto
    total_images = len(all_images)
    train_size = int(total_images * split_ratio[0])
    val_size = int(total_images * split_ratio[1])
    
    # Dividir las imágenes
    train_images = all_images[:train_size]
    val_images = all_images[train_size:train_size+val_size]
    test_images = all_images[train_size+val_size:]
    
    # Función para mover las imágenes y las etiquetas
    def move_files(images, source_image_dir, source_label_dir, dest_image_dir, dest_label_dir):
        for image_name in images:
            # Mover imagen
            shutil.move(os.path.join(source_image_dir, image_name), os.path.join(dest_image_dir, image_name))
            # Mover archivo de etiquetas
            label_name = image_name.replace('.jpg', '.txt')  # Asumiendo que las imágenes son .jpg
            shutil.move(os.path.join(source_label_dir, label_name), os.path.join(dest_label_dir, label_name))
    
    # Mover las imágenes y etiquetas a las carpetas correspondientes
    move_files(train_images, image_dir, label_dir, os.path.join(train_dir, 'images'), os.path.join(train_dir, 'labels'))
    move_files(val_images, image_dir, label_dir, os.path.join(val_dir, 'images'), os.path.join(val_dir, 'labels'))
    move_files(test_images, image_dir, label_dir, os.path.join(test_dir, 'images'), os.path.join(test_dir, 'labels'))
    
    print(f"Datos divididos: {len(train_images)} para entrenamiento, {len(val_images)} para validación, {len(test_images)} para prueba.")

# Ejecutar la función de división de datos
image_dir = '../data/raw/images/'
label_dir = '../data/raw/labels/'

train_dir = '../data/train/'
val_dir = '../data/val/'
test_dir = '../data/test/'

split_data(image_dir, label_dir, train_dir, val_dir, test_dir)
