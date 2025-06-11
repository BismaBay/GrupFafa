from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import joblib
from keras_facenet import FaceNet
from mtcnn import MTCNN
import time
import datetime
import csv
import os
import base64

# Initialize the Flask App
app = Flask(__name__)

# --- MODEL & TOOLS INITIALIZATION ---
embedder = FaceNet()
detector = MTCNN()
model_path = r"D:\SEMESTER 6\9. Desain Proyek\svm_multiclass_model.pkl"
clf = joblib.load(model_path)
camera = cv2.VideoCapture(0)

# --- HISTORY & CSV RELATED VARIABLES/FUNCTIONS ---
HISTORI_FILE = 'histori_absen.csv'
absen_tercatat = {}

def load_absen_tercatat_hari_ini():
    """Memuat data yang sudah absen hari ini dari CSV untuk mencegah duplikat."""
    tanggal_hari_ini = datetime.datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(HISTORI_FILE):
        return
    with open(HISTORI_FILE, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        try:
            next(reader) # Lewati header
            for row in reader:
                if row and row[1] == tanggal_hari_ini:
                    absen_tercatat[row[0]] = row[1]
        except StopIteration:
            pass
    print(f"Data absen hari ini yang sudah dimuat: {absen_tercatat}")

def simpan_ke_csv(nama):
    """Saves name, date, and time to the CSV file."""
    sekarang = datetime.datetime.now()
    tanggal = sekarang.strftime("%Y-%m-%d")
    jam = sekarang.strftime("%H:%M:%S")

    if nama in absen_tercatat and absen_tercatat[nama] == tanggal:
        return False

    absen_tercatat[nama] = tanggal
    
    with open(HISTORI_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['Nama', 'Tanggal', 'Jam'])
        writer.writerow([nama, tanggal, jam])
    return True

def generate_frames():
    """Video stream generator with real-time face labeling, FPS, and timestamp."""
    prev_time = 0
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            try:
                faces = detector.detect_faces(frame)
                for face in faces:
                    x, y, w, h = face['box']
                    x, y = abs(x), abs(y)
                    face_crop = frame[y:y+h, x:x+w]
                    
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    face_resized = cv2.resize(face_rgb, (160, 160))
                    embedding = embedder.embeddings([face_resized])[0]
                    
                    pred = clf.predict([embedding])[0]
                    proba = clf.predict_proba([embedding])[0].max()
                    
                    label = f"{pred} ({proba:.2f})"
                    
                    color = (0, 0, 255)
                    if proba >= 0.8:
                        color = (0, 255, 0)
                    elif proba >= 0.5:
                        color = (0, 255, 255)
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            except Exception:
                pass
            
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            current_time_str = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            cv2.putText(frame, current_time_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    # ... (fungsi ini tetap sama)
    histori_data = []
    if os.path.exists(HISTORI_FILE):
        with open(HISTORI_FILE, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                histori_data.append(row)
    histori_data.reverse()
    return render_template('index.html', histori=histori_data)

@app.route('/api/histori')
def api_histori():
    # ... (fungsi ini tetap sama)
    histori_data = []
    if os.path.exists(HISTORI_FILE):
        with open(HISTORI_FILE, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                histori_data.append(row)
    histori_data.reverse()
    response = jsonify(histori_data)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/capture', methods=['POST'])
def capture():
    # ... (fungsi ini tetap sama)
    data = request.get_json()
    image_data_url = data['image']
    header, encoded = image_data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faces = detector.detect_faces(frame)
    if not faces:
        return jsonify({'status': 'error', 'message': 'Tidak ada wajah yang terdeteksi.'})
    face = faces[0]
    x, y, w, h = face['box']
    x, y = abs(x), abs(y)
    face_crop = frame[y:y+h, x:x+w]
    try:
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (160, 160))
        embedding = embedder.embeddings([face_resized])[0]
        pred = clf.predict([embedding])[0]
        proba = clf.predict_proba([embedding])[0].max()
        if proba >= 0.8:
            if simpan_ke_csv(pred):
                return jsonify({'status': 'success', 'message': f'Absensi untuk {pred} berhasil dicatat!'})
            else:
                return jsonify({'status': 'info', 'message': f'{pred} sudah melakukan absensi hari ini.'})
        else:
            return jsonify({'status': 'error', 'message': f'Wajah terdeteksi sebagai {pred}, tapi tingkat keyakinan terlalu rendah untuk absen.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': 'Terjadi kesalahan saat memproses gambar.'})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- FUNGSI BARU UNTUK MERESET HISTORI ---
@app.route('/reset_history', methods=['POST'])
def reset_history():
    """Menghapus file histori dan mereset data di memori."""
    try:
        if os.path.exists(HISTORI_FILE):
            os.remove(HISTORI_FILE)
        
        # Mereset dictionary di memori agar bisa absen lagi
        absen_tercatat.clear()
        
        print("Histori berhasil direset.")
        return jsonify({'status': 'success', 'message': 'Semua riwayat absensi telah berhasil dihapus.'})
    except Exception as e:
        print(f"Error saat mereset histori: {e}")
        return jsonify({'status': 'error', 'message': 'Gagal menghapus riwayat absensi.'})

if __name__ == '__main__':
    load_absen_tercatat_hari_ini()
    app.run(debug=True)
