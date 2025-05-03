import numpy as np
import cv2
import os
from keras_facenet import FaceNet

# Load model FaceNet
embedder = FaceNet()

# Folder dataset hasil preprocessing
input_folder = "D:\SEMESTER 6\9. Desain Proyek\DATASET\PreProcessed"
output_folder = "D:\SEMESTER 6\9. Desain Proyek\DATASET\Hasil Embadding Face"

# Pastikan folder output ada
os.makedirs(output_folder, exist_ok=True)

# Proses setiap folder dalam folder Preprocessed
for person_folder in os.listdir(input_folder):
    person_path = os.path.join(input_folder, person_folder)

    # Cek apakah itu folder (karena setiap folder berisi gambar untuk satu orang)
    if not os.path.isdir(person_path):
        continue

    # Buat folder output untuk setiap orang
    output_person_folder = os.path.join(output_folder, person_folder)
    os.makedirs(output_person_folder, exist_ok=True)

    # Proses setiap gambar dalam folder orang
    for filename in os.listdir(person_path):
        img_path = os.path.join(person_path, filename)

        # Cek apakah file gambar ada
        if not os.path.isfile(img_path):
            print(f"File tidak ditemukan: {img_path}")
            continue

        img = cv2.imread(img_path)

        # Pastikan gambar berhasil dibaca
        if img is None:
            print(f"Gagal membaca gambar: {img_path}")
            continue

        # Convert ke RGB dan resize (FaceNet input: 160x160)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (160, 160))

        # Ekstraksi embedding
        embedding = embedder.embeddings([img_rgb])[0]

        # Tentukan path file output untuk menyimpan embedding
        embedding_file = os.path.join(output_person_folder, f"{os.path.splitext(filename)[0]}_embedding.npy")

        # Simpan embedding dalam file .npy
        np.save(embedding_file, embedding)

        print(f"Embedding untuk {filename} disimpan di {embedding_file}")

print("Ekstraksi fitur selesai!")

