import cv2
import numpy as np
import joblib
from keras_facenet import FaceNet
from mtcnn import MTCNN
import time

# Load model dan tools
embedder = FaceNet()
detector = MTCNN()
model_path = r"D:\SEMESTER 6\9. Desain Proyek\DATASET\svm_multiclass_model.pkl"
clf = joblib.load(model_path)

# Mulai capture video dari webcam
cap = cv2.VideoCapture(0)

# Variabel untuk menghitung FPS
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal menangkap frame dari kamera")
        break

    faces = detector.detect_faces(frame)
    for face in faces:
        x, y, w, h = face['box']
        x, y = abs(x), abs(y)  # Pastikan koordinat positif
        face_crop = frame[y:y+h, x:x+w]

        try:
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (180, 180))
            embedding = embedder.embeddings([face_resized])[0]

            pred = clf.predict([embedding])[0]
            proba = clf.predict_proba([embedding])[0].max()

            label = f"{pred} ({proba:.2f})"

            # Gambar kotak dan label pada frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        except Exception as e:
            print(f"Error proses wajah: {e}")

    # Hitung FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # Tampilkan FPS di pojok kiri atas
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Recognition Real-Time", frame)

    # Tekan tombol 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
