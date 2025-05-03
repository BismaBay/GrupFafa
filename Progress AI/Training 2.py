import cv2
from mtcnn import MTCNN
import os

# Inisialisasi detektor wajah
detector = MTCNN()

# Folder input dan output
input_folder = "D:\SEMESTER 6\9. Desain Proyek\DATASET\DATASHEET"       # Folder dengan subfolder per nama
output_folder = "D:\SEMESTER 6\9. Desain Proyek\DATASET\PreProcessed"   # Folder hasil preprocessing
os.makedirs(output_folder, exist_ok=True)

def preprocess_image(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Gagal membaca gambar: {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)

    if len(faces) > 0:
        x, y, width, height = faces[0]['box']
        face = img_rgb[y:y+height, x:x+width]

        # Resize wajah ke 160x160 piksel
        face_resized = cv2.resize(face, (160, 160))

        # Simpan hasil
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR))
        print(f"Processed: {output_path}")
    else:
        print(f"Wajah tidak terdeteksi: {image_path}")

# Telusuri seluruh subfolder dan file gambar
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(input_path, input_folder)
            output_path = os.path.join(output_folder, relative_path)

            preprocess_image(input_path, output_path)
