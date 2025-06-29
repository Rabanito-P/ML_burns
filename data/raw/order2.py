import os

def verify_image_label_pairs(base_path):
    images_dir = os.path.join(base_path, "images")
    labels_dir = os.path.join(base_path, "labels")

    image_exts = [".jpg", ".jpeg"]

    image_files = [f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in image_exts]
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    image_basenames = set(os.path.splitext(f)[0] for f in image_files)
    label_basenames = set(os.path.splitext(f)[0] for f in label_files)

    only_images = image_basenames - label_basenames
    only_labels = label_basenames - image_basenames

    # Archivos inválidos en cada carpeta
    invalid_images = [f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() not in image_exts]
    invalid_labels = [f for f in os.listdir(labels_dir) if not f.endswith('.txt')]

    print("📊 Verificación de consistencia:")
    print(f"🖼️  Imágenes válidas: {len(image_files)}")
    print(f"📝 Etiquetas válidas: {len(label_files)}")

    if only_images:
        print("\n⚠️  Imágenes sin etiqueta correspondiente:")
        for f in only_images:
            print(f"  - {f}")
    if only_labels:
        print("\n⚠️  Etiquetas sin imagen correspondiente:")
        for f in only_labels:
            print(f"  - {f}")

    if invalid_images:
        print("\n❌ Archivos no válidos en carpeta 'images/':")
        for f in invalid_images:
            print(f"  - {f}")
    if invalid_labels:
        print("\n❌ Archivos no válidos en carpeta 'labels/':")
        for f in invalid_labels:
            print(f"  - {f}")

    if not only_images and not only_labels and not invalid_images and not invalid_labels:
        print("\n✅ Todo está en orden: imágenes y etiquetas coinciden perfectamente.")

if __name__ == "__main__":
    base_folder = "./"  # Cambiar si es necesario
    verify_image_label_pairs(base_folder)
