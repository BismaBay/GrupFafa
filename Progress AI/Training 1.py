import cv2
import os
from mtcnn import MTCNN

# Inisialisasi MTCNN
detector = MTCNN()

# Folder input (ubah sesuai kebutuhan)
input_folder = "D:\SEMESTER 6\9. Desain Proyek\DATASET\DATASHEET"

# Statistik
total_files = 0
readable_files = 0
face_detected = 0

print("Mulai pengecekan struktur dan isi folder...\n")

for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            total_files += 1
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"[ERROR] Gagal membaca: {img_path}")
                continue
            else:
                readable_files += 1

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(img_rgb)
            if len(faces) > 0:
                face_detected += 1
            else:
                print(f"[NO FACE] Tidak ada wajah di: {img_path}")

print("\n--- RINGKASAN ---")
print(f"Total file gambar ditemukan     : {total_files}")
print(f"File berhasil dibaca (cv2)      : {readable_files}")
print(f"Wajah berhasil terdeteksi       : {face_detected}")
print(f"File tanpa wajah Terlihat     : {readable_files - face_detected}")
