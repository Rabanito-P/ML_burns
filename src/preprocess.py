from PIL import Image
import os

# Redimensionar la imagen y sobrescribirla en la misma carpeta
def resize_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img.save(image_path)  # Sobrescribe la imagen original
    return img

# Ajustar las coordenadas de los bounding boxes en formato YOLO
def adjust_bounding_boxes_yolo(bboxes, width_scale, height_scale):
    adjusted_bboxes = []
    for bbox in bboxes:
        class_id, cx, cy, w, h = bbox  # Formato: class cx cy w h
        cx *= width_scale
        cy *= height_scale
        w *= width_scale
        h *= height_scale
        adjusted_bboxes.append([int(class_id), cx, cy, w, h])
    return adjusted_bboxes

# Preprocesar las imágenes y las etiquetas
def preprocess_data(image_dir, label_dir, target_size=(224, 224)):
    images = []
    labels = []
    
    for image_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, image_name)

        # Obtener nombre del archivo de etiqueta correspondiente
        label_filename = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(label_dir, label_filename)

        # Saltar si no existe el archivo .txt correspondiente
        if not os.path.exists(label_path):
            print(f"⚠️  No se encontró etiqueta para: {image_name}, se omite.")
            continue

        # Redimensionar la imagen y sobrescribirla
        img = resize_image(img_path, target_size)

        # Leer las etiquetas del bounding box y ajustarlas
        with open(label_path, 'r') as f:
            bboxes = [list(map(float, line.strip().split())) for line in f.readlines() if line.strip()]

        # Calcular las proporciones de escala para ajustar los bounding boxes
        width_scale = target_size[0] / img.width
        height_scale = target_size[1] / img.height
        adjusted_bboxes = adjust_bounding_boxes_yolo(bboxes, width_scale, height_scale)

        # Sobrescribir las nuevas etiquetas en el mismo archivo
        with open(label_path, 'w') as f:
            for bbox in adjusted_bboxes:
                f.write(' '.join(map(str, bbox)) + '\n')

        images.append(img)
        labels.append(adjusted_bboxes)
    
    return images, labels

# Ejecutar el preprocesamiento
if __name__ == "__main__":
    # Directorios de imágenes y etiquetas originales
    image_dir = '../data/raw/images/'  # Directorio de imágenes originales
    label_dir = '../data/raw/labels/'  # Directorio de etiquetas originales

    # Preprocesar los datos
    images, labels = preprocess_data(image_dir, label_dir)
    print(f"✅ Imágenes redimensionadas y sobrescritas en {image_dir}")
    print(f"✅ Etiquetas ajustadas y sobrescritas en {label_dir}")
