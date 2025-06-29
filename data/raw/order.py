import os
import shutil

def organize_files(folder_path):
    images_dir = os.path.join(folder_path, "images")
    labels_dir = os.path.join(folder_path, "labels")

    # Crear carpetas si no existen
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.jpg', '.jpeg']:
                shutil.move(file_path, os.path.join(images_dir, file))
            elif ext == '.txt':
                shutil.move(file_path, os.path.join(labels_dir, file))

    print("✅ Archivos organizados en 'images/' y 'labels/'.")

if __name__ == "__main__":
    folder = "./"  # Cambia esto si tu carpeta está en otra ubicación
    organize_files(folder)
