# GrupFafa
Desain Proyek (Absensi Face Recognition)

## 🛠️Metode Yang Digunakan
Metodologi pengembangan yang dipilih adalah SCRUM

## 🧑‍🤝‍🧑Peran Dalam Tim SCRUM
Team Leader :

Bisma Bayu Kresna 

Supervisior :

Joko Febrianto

🧠 Kelompok 1 - Team AI :

Fokus: Mengembangkan dan menguji model Face Recognition menggunakan machine learning/deep learning.

Anggota :

M.Meddy Athallah,
Devanda Syaputra,
Rhenuka Ayusha,
Fariz Hadi P.

Tugas-Tugas :

Tugas-tugas:

Mengumpulkan dan menyiapkan data wajah (preprocessing, augmentasi).

Melatih model face recognition menggunakan algoritma (misalnya, CNN, FaceNet, atau OpenCV LBPH).

Melakukan validasi dan evaluasi model (akurasi, precision, recall).

Mengintegrasikan model dengan sistem backend agar bisa dipanggil dari website.

🌐 Kelompok 2 - Team Website :

Fokus: Mendesain dan membangun antarmuka pengguna serta backend untuk menghubungkan pengguna dengan sistem Face Recognition.

Anggota :

Shafli,
M.Hakim Fahad,
M.Ikhsan,
Achmad Husni.

Tugas - tugas :

Membangun tampilan web (front-end) untuk login/registrasi dengan face recognition.

Menghubungkan front-end ke server/model AI (misalnya lewat API Flask/FastAPI).

Membuat sistem upload/capture foto wajah via webcam atau file input.

Menampilkan hasil pengenalan wajah (berhasil/gagal, nama pengguna, dsb).

Mengelola database pengguna dan autentikasi.

Melakukan testing antarmuka agar mudah digunakan dan responsif.

## 🎯 Client Request:

Jalan Lokal di laptop

Proses Absensi tidak perlu gerakan khusus

Database semua anak 1 kelas + Dosen.

## 🎯 Fitur Sistem Face Recognition :

🔍 Fitur Kecerdasan Buatan (AI):

Bounding Box Dinamis Berwarna
Menampilkan kotak pembatas (bounding box) dengan warna berbeda sesuai tingkat kepercayaan (confidence level) hasil deteksi wajah, sehingga memudahkan identifikasi akurasi sistem secara visual.

Rekapan Tanggal dan Waktu Otomatis
Sistem secara otomatis mencatat dan merekam data waktu (tanggal dan jam) setiap kali proses pengenalan wajah terjadi, sebagai bagian dari histori kehadiran.

Pengenalan Wajah Multi-User Secara Real-Time
Mampu mengenali dan memproses beberapa wajah secara bersamaan dalam satu frame, disesuaikan dengan kapasitas hardware yang digunakan.

Tampilan Kecepatan Pemrosesan (FPS)
Menyediakan informasi jumlah frame per detik (FPS) secara real-time sebagai indikator performa sistem dalam memproses gambar/video secara langsung.

🖥️ Fitur Antarmuka dan Sistem:
Halaman Beranda (Home)
Tampilan utama yang menyajikan informasi ringkas dan akses cepat ke fitur inti sistem.

Fitur Presensi (Absen)
Menu untuk melakukan absensi secara otomatis menggunakan teknologi pengenalan wajah, tanpa perlu input manual.

Riwayat Absensi (History Absen)
Menyediakan akses ke histori kehadiran pengguna berdasarkan waktu, termasuk pencatatan detail nama, tanggal, dan waktu kehadiran.

Integrasi Perangkat Tunggal (1 Device)
Sistem dirancang untuk berjalan optimal pada satu perangkat terpusat, yang menggabungkan fungsi kamera, pemrosesan AI, dan antarmuka pengguna dalam satu unit.
