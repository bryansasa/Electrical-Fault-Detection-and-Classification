# Electrical Fault Detection and Classification (Three-Phase System)
### Implementasi Edge-to-Cloud: NVIDIA Jetson Nano x Render Cloud

[![Platform](https://img.shields.io/badge/Edge%20Device-NVIDIA%20Jetson%20Nano-76B900?logo=nvidia)](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
[![Framework](https://img.shields.io/badge/ML%20Framework-Scikit--Learn-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![Web Engine](https://img.shields.io/badge/Backend-Flask%20%7C%20Ngrok-000000?logo=flask)](https://flask.palletsprojects.com/)
[![Deployment](https://img.shields.io/badge/Cloud%20Hosting-Render-46E3B7?logo=render)](https://phase-ai-y1l9.onrender.com/)

---

## 📌 Ringkasan Proyek
Proyek ini mengimplementasikan sistem klasifikasi gangguan kelistrikan 3 fasa ($I_a, I_b, I_c, V_a, V_b, V_c$) secara real-time berbasis **Edge-to-Cloud Hybrid Architecture**.
- **Edge Inference Engine (NVIDIA Jetson Nano):** Berfungsi sebagai relai proteksi cerdas (*Intelligent Electronic Device / IED*) di gardu induk transmisi dengan latensi inferensi ultra-rendah (< 5 ms).
- **Cloud Dashboard (Render):** Portal antarmuka pengguna interaktif (PHASE/AI) yang memvisualisasikan kondisi transmisi dan topologi gangguan.
- **Secure Encrypted Tunnel (Ngrok):** Menghubungkan cloud frontend ke hardware Jetson Nano fisik secara permanen melalui domain publik HTTPS berenkripsi SSL/TLS.

---

## 📂 Struktur Direktori Repositori

```text
Electrical-Fault-Detection-and-Classification/
├── data/                             # Dataset, partisi split, dan data sampel
│   ├── classData.csv                 # Dataset benchmark 6 kelas
│   ├── detect_dataset.csv            # Dataset pendukung deteksi biner
│   ├── split_manifest.csv            # Partisi data (Train, Test Random, Test Block)
│   ├── contoh_input.csv              # Sampel uji per kelas untuk pengujian CLI
│   └── contoh_label_referensi.csv    # Label acuan validasi
├── docs/                             # Laporan resmi & evaluasi performa
│   ├── DOKUMENTASI_DEPLOYMENT_JETSON_NANO.md  # Laporan lengkap edge deployment
│   ├── DOKUMENTASI_DEPLOYMENT_JETSON_NANO.pdf # Laporan resmi PDF A4 5 Halaman
│   ├── ringkasan_hasil.txt           # Rangkuman evaluasi komparasi
│   └── evaluation/                   # Grafik visualisasi & metrik evaluasi
│       ├── confusion_matrix.png
│       ├── perbandingan_model.png
│       ├── korelasi_train.png
│       └── *.csv (metrik holdout, cv, hasil tuning)
├── notebooks/                        # Eksperimen Google Colab
│   └── TugasColab_Klasifikasi_Gangguan_Listrik.ipynb
├── phase-ai/                         # Web Interface & REST API Engine
│   ├── app.py                        # Server Flask + CORS + Live Terminal Log
│   ├── requirements.txt              # Dependensi web
│   └── templates/index.html          # Dashboard interaktif fasa transmisi
├── main_deploy.py                    # Script inferensi mandiri CLI
├── retrain_on_jetson.py              # Script auto-retrain mandiri di Jetson Nano
├── benchmark_jetson.py               # Script pengukur latensi & throughput edge
├── model_pipeline.joblib             # Pipeline ML (KNN k=15)
├── metadata.json                     # Kontrak input & metadata model
└── requirements.txt                  # Dependensi Python
```

---

## 🚀 Panduan Menjalankan

### 1. Inferensi Mandiri via Baris Perintah (CLI)
Pastikan dependensi terpasang (`pip install -r requirements.txt`).
- **Menggunakan 6 nilai arus & tegangan:**
  ```bash
  python main_deploy.py --values -151.2 -9.6 85.8 0.40 -0.13 -0.26
  ```
  *Output:*
  ```json
  [{"fault_code":"1001","fault_name":"A-G: fase A ke tanah"}]
  ```
- **Menggunakan berkas CSV:**
  ```bash
  python main_deploy.py --input data/contoh_input.csv --output hasil_prediksi.csv
  ```

---

### 2. Menjalankan di NVIDIA Jetson Nano
1. **Latih ulang model secara lokal (jika ada perbedaan versi Scikit-Learn):**
   ```bash
   python3 retrain_on_jetson.py
   ```
2. **Jalankan Web Server & Ngrok Tunnel:**
   ```bash
   cd phase-ai
   python3 app.py &
   ngrok http --url=overreadily-discontinuous-betty.ngrok-free.dev 5000
   ```
   *Domain Permanen API Jetson:* `https://overreadily-discontinuous-betty.ngrok-free.dev/predict`

3. **Uji Latensi Inferensi Edge:**
   ```bash
   python3 benchmark_jetson.py
   ```

---

## 📊 Hasil Evaluasi & Komparasi Kinerja

| Parameter Evaluasi | Cloud Server (Render) | Edge Device (NVIDIA Jetson Nano) |
|---|---|---|
| **Perangkat Komputasi** | Virtual Container Cloud | NVIDIA Tegra X1 (4-core Cortex-A57) |
| **Akurasi Model KNN** | 81.68% | **81.68%** |
| **Waktu Training Model** | ~1.200 ms | **34.15 ms** |
| **Latensi Inferensi Lokal** | Tidak Berlaku (Hanya Jaringan) | **~2.85 ms / sampel** |
| **Throughput Maksimal** | Terbatas Kuota Cloud | **~350 prediksi / detik** |
| **Operasional Saat Offline** | Gagal (Wajib Ada Internet) | **Bisa Beroperasi 100% Offline (Gardu)** |

---

## 👥 Pembagian Kerja Tim
- **Bryan:** Edge Computing Hardware, Deployment NVIDIA Jetson Nano, REST API & Tunneling, Pengukuran Latensi.
- **Reno:** Eksplorasi Dataset, Feature Engineering, Training Pipeline Model (KNN).
- **Arif:** Web Interface UI/UX, Cloud Hosting Render.
- **Anggota 4:** Preprocessing data & pembagian partisi (Purge, Train, Block, Random).
- **Anggota 5:** Evaluasi metrik, cross-validation, dan analisis confusion matrix.
- **Anggota 6:** Integrasi sistem, penyusunan laporan 5–8 halaman, dan dokumentasi video YouTube.
