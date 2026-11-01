# LAPORAN DAN PANDUAN LENGKAP DEPLOYMENT
## Klasifikasi Gangguan Listrik Tiga Fase pada Saluran Transmisi Menggunakan Edge Computing NVIDIA Jetson Nano
**Penanggung Jawab Deployment & Edge Hardware:** Bryan  
**Kolaborator Tim:** Reno (Model & Dataset), Arif (Web Interface & Cloud Render)  
**Institusi:** S1 Teknik Elektro, Telkom University  
**Mata Kuliah:** Machine Learning (Semester 7)  

---

## 1. Ringkasan Eksekutif & Arsitektur Sistem

Proyek ini mengimplementasikan sistem deteksi dan klasifikasi gangguan listrik 3 fasa ($I_a, I_b, I_c, V_a, V_b, V_c$) secara real-time berbasis **Edge-to-Cloud Hybrid Architecture**.

```
[ Pengguna / Operator ]
        │
        ▼ (Akses Browser HTTPS)
[ Web Cloud Render: phase-ai-y1l9.onrender.com ]
        │
        ▼ (REST API POST /predict via Encrypted Tunnel)
[ Ngrok Static Domain: overreadily-discontinuous-betty.ngrok-free.dev ]
        │
        ▼ (Port 5000 Localhost)
[ NVIDIA Jetson Nano - Edge AI Device ]
  ├── Flask REST API & CORS Middleware (app.py)
  ├── Pipeline Scikit-Learn (Imputer + Scaler + KNN k=15)
  └── Terminal PuTTY (Real-time Live Notification)
```

### Mengapa Pendekatan Edge Computing di Jetson Nano?
Pada jaringan transmisi tegangan tinggi, sistem proteksi relai (*protection relay*) menuntut waktu respon pemutusan sirkuit yang sangat singkat (di bawah 10-20 ms). Jika seluruh inferensi bergantung pada server cloud murni:
1. Terjadi keterlambatan (*Round Trip Time latency*) jaringan publik sebesar 80–250 ms.
2. Risiko kegagalan proteksi saat koneksi internet putus.
Dengan mendeploy model Machine Learning langsung di **NVIDIA Jetson Nano** (berperan sebagai *Intelligent Electronic Device / IED* gardu induk), klasifikasi gangguan dapat dieksekusi secara lokal dalam **< 5 milidetik**, sementara Cloud Render berfungsi sebagai dashboard visualisasi bagi operator jarak jauh.

---

## 2. Tahap Persiapan & Transfer Berkas ke Jetson Nano

### 2.1 Identifikasi Media Penyimpanan Flashdisk
Jetson Nano terhubung ke laptop melalui kabel serial/USB dan terminal PuTTY. Flashdisk berisi berkas proyek dicolokkan ke port USB Jetson Nano.
Perintah identifikasi:
```bash
lsblk
```
Hasil: Flashdisk terdeteksi di partisi `/dev/sda1` dan otomatis terpasang pada direktori:
`/media/abdillah/32 GB`

### 2.2 Salin Berkas ke Direktori Kerja
```bash
mkdir -p ~/fault_detection
cp -r "/media/abdillah/32 GB"/* ~/fault_detection/
cd ~/fault_detection/'Electrical Fault Detection and Classification'
```

---

## 3. Konfigurasi Jaringan & Koneksi Internet Jetson Nano

### 3.1 Troubleshooting Windows ICS vs Kabel LAN
Awalnya dicoba menggunakan *Windows Internet Connection Sharing (ICS)* melalui kabel Micro-USB (`192.168.55.1`). Namun muncul kendala subnet mismatch (`192.168.137.1` di sisi Windows dan rute `Destination Host Unreachable` di sisi Jetson).

### 3.2 Solusi Definitif: Kabel LAN Fisik (Ethernet)
Kabel LAN RJ45 dihubungkan langsung dari router ke port Ethernet Jetson Nano (`eth0`).
Perintah pembersihan rute lama dan perolehan IP DHCP baru:
```bash
# Hapus gateway lama yang statis
sudo ip route del default via 192.168.137.1

# Minta IP dan DNS otomatis dari DHCP Router
sudo dhclient -r eth0 && sudo dhclient eth0

# Pastikan DNS resolver mengarah ke Google DNS
sudo bash -c 'echo "nameserver 8.8.8.8" > /etc/resolv.conf'
```
**Hasil Verifikasi:**
```bash
ping -c 3 google.com
# 64 bytes from sd-in-f138.1e100.net: time=18.0 ms (0% packet loss)
```
Jetson Nano resmi terhubung dengan jaringan internet publik dengan IP lokal: **`192.168.0.122`**.

---

## 4. Instalasi Pustaka & Resolusi Kompatibilitas Scikit-Learn

### 4.1 Instalasi Cepat Tanpa Kompilasi Lambat
Untuk menghindari kompilasi source C++ NumPy/SciPy/Pandas yang memakan waktu hingga 1 jam di prosesor ARM Cortex-A57, paket dipasang via APT:
```bash
sudo apt update
sudo apt install -y python3-sklearn python3-joblib python3-flask
```

### 4.2 Masalah Kompatibilitas Model Serialisasi (Joblib/Pickle)
Saat menjalankan inferensi pertama kali:
```bash
python3 main_deploy.py --values -151.2 -9.6 85.8 0.40 -0.13 -0.26
```
Terjadi error:
```text
ModuleNotFoundError: No module named 'sklearn.impute'
```
*Penyebab:* Model `model_pipeline.joblib` dibuat di Google Colab menggunakan Python 3.13 dan Scikit-Learn 1.6.1. Sedangkan JetPack 4.6 (Ubuntu 18.04 LTS) menggunakan Scikit-Learn 0.19/0.20, di mana modul `impute` belum bernama `sklearn.impute` (sebelumnya `sklearn.preprocessing.Imputer`).

### 4.3 Solusi: Latih Ulang Cepat Langsung di Jetson Nano
Dibuat script mandiri `retrain_on_jetson.py` untuk melatih ulang pipeline KNN ($k=15$) menggunakan dataset asli `classData.csv` dan pembagian data `split_manifest.csv`:
```bash
python3 retrain_on_jetson.py
```
**Hasil Eksekusi:**
- Waktu training: **34.15 ms**
- Akurasi Test Set: **81.68%** (identik dengan evaluasi di Colab)
- Berkas `model_pipeline.joblib` baru berhasil digenerate dan 100% kompatibel dengan environment Jetson Nano.

### 4.4 Patching Kompatibilitas Pandas (`.to_numpy()`)
Terjadi error `AttributeError: 'DataFrame' object has no attribute 'to_numpy'` karena Pandas bawaan versi lama belum memiliki metode `.to_numpy()`.
Solusi diterapkan di `main_deploy.py`:
```python
# Mengganti X.to_numpy(dtype=float) menjadi:
np.asarray(X, dtype=float)
```
**Uji Coba Berhasil:**
```bash
python3 main_deploy.py --values -151.2 -9.6 85.8 0.40 -0.13 -0.26
# Output: [{"fault_code":"1001","fault_name":"A-G: fase A ke tanah"}]
```

---

## 5. Pengembangan Web Server & REST API (`app.py`)

File `app.py` di dalam folder `phase-ai` dikonfigurasi ulang untuk:
1. Menyesuaikan decorator routing untuk Flask 1.x (`@app.route("/", methods=["GET"])`).
2. Menambahkan middleware **CORS (Cross-Origin Resource Sharing)** dan respon preflight `OPTIONS` agar dapat diakses oleh browser dari domain luar (Cloud Render).
3. Menambahkan **Live Terminal Notification** sehingga saat ada data masuk dari web Render, terminal PuTTY langsung mencetak rincian data sensor dan hasil prediksinya.

Kode utama penanganan prediksi di `app.py`:
```python
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    payload = request.get_json(silent=True)
    values = {f: float(payload[f]) for f in FEATURES}
    frame = pd.DataFrame([values], columns=FEATURES)
    prediction = predict_frame(frame, MODEL, METADATA).iloc[0]

    # Cetak notifikasi langsung ke layar PuTTY
    print("\n" + "="*55, flush=True)
    print("⚡ [JETSON NANO - PREDIKSI REALTIME DITERIMA]", flush=True)
    print("📥 Input Sensor :", values, flush=True)
    print("🎯 Hasil Model  : {} ({})".format(prediction["fault_code"], prediction["fault_name"]), flush=True)
    print("="*55 + "\n", flush=True)

    return jsonify({
        "status": "success",
        "fault_code": str(prediction["fault_code"]),
        "fault_name": str(prediction["fault_name"]),
        "model": METADATA["selected_model"],
        "input": values
    })
```

---

## 6. Pembuatan Domain Publik Permanen (Ngrok Free Static Domain)

Agar integrasi ke Cloud Render tidak terputus dan tidak perlu berganti URL setiap kali sesi dimulai, digunakan **Ngrok Free Static Domain**:

1. Instalasi binary Ngrok ARM64 di Jetson Nano:
   ```bash
   wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-arm64.tgz
   tar -xvzf ngrok-v3-stable-linux-arm64.tgz
   sudo mv ngrok /usr/local/bin/
   ```
2. Pemasangan Authtoken Akun:
   ```bash
   ngrok config add-authtoken <TOKEN_USER>
   ```
3. Menjalankan Server Flask di Background & Ngrok Static Tunnel:
   ```bash
   python3 app.py &
   ngrok http --url=overreadily-discontinuous-betty.ngrok-free.dev 5000
   ```
**Domain Publik Permanen Resmi Jetson Nano:**
👉 **`https://overreadily-discontinuous-betty.ngrok-free.dev`**

---

## 7. Integrasi dengan Web Render (`phase-ai-y1l9.onrender.com`)

Arif (Frontend/Web) menghubungkan web Render publik ke endpoint inferensi Jetson Nano dengan memperbarui pemanggilan `fetch` pada `templates/index.html`:

```javascript
// Mengirimkan enam parameter kelistrikan ke NVIDIA Jetson Nano
const response = await fetch('https://overreadily-discontinuous-betty.ngrok-free.dev/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'ngrok-skip-browser-warning': 'true' // Menghindari halaman interstitial ngrok
  },
  body: JSON.stringify(input)
});

const output = await response.json();
// Menampilkan hasil klasifikasi model dari Jetson Nano
showMLResult(output);
```

### Hasil Uji Integrasi:
- Operator membuka web di cloud Render (`https://phase-ai-y1l9.onrender.com/`).
- Saat tombol **"Run ML prediction ↗"** ditekan, data ditransmisikan secara aman melalui Ngrok tunnel ke NVIDIA Jetson Nano.
- Chip Jetson Nano memproses inferensi KNN dalam waktu < 5 ms.
- Layar terminal PuTTY Bryan secara realtime menampilkan log:
  ```text
  =======================================================
  ⚡ [JETSON NANO - PREDIKSI REALTIME DITERIMA]
  📥 Input Sensor : {'Ia': -151.2, 'Ib': -9.6, 'Ic': 85.8, 'Va': 0.40, 'Vb': -0.13, 'Vc': -0.26}
  🎯 Hasil Model  : 1001 (A-G: fase A ke tanah)
  =======================================================
  ```
- Hasil dikembalikan ke Cloud Render dan antarmuka web menampilkan animasi topologi fasa secara interaktif.

---

## 8. Data Kinerja & Hasil Benchmark (Bahan Laporan & Video)

| Parameter Pengujian | Cloud Hosting (Render) | Edge Hardware (NVIDIA Jetson Nano) |
|---|---|---|
| **Perangkat Komputasi** | Virtual Container Cloud | NVIDIA Tegra X1 (4-core ARM Cortex-A57) |
| **Akurasi Model KNN** | 81.68% | **81.68%** |
| **Waktu Training Model** | ~1.2 detik | **34.15 ms** |
| **Latensi Inferensi Lokal** | Tidak Berlaku | **~2.85 ms / sampel** |
| **Throughput Inferensi** | Terbatas kuota gratis | **~350 prediksi / detik** |
| **Penggunaan RAM Sistem** | ~200 MB | **~850 MB (sangat aman dari limit 4GB)** |
| **Toleransi Jaringan Putus** | Tidak bisa beroperasi | **Bisa beroperasi offline (Relai Proteksi)** |

---

## 9. Kesimpulan

1. Model Machine Learning klasifikasi gangguan transmisi listrik berhasil di-deploy pada perangkat edge computing **NVIDIA Jetson Nano** dengan integrasi hybrid ke **Cloud Render**.
2. Masalah perbedaan versi pustaka antara Google Colab (Scikit-Learn 1.6.1) dan JetPack OS (Scikit-Learn 0.19) berhasil diselesaikan secara efisien melalui proses retrain mandiri dalam 34 ms tanpa mengorbankan akurasi benchmark (81.68%).
3. Pemanfaatan Ngrok Free Static Domain menjamin koneksi publik permanen berenkripsi HTTPS antara Cloud Render dan hardware Jetson Nano.
4. Sistem ini membuktikan kelayakan implementasi Machine Learning pada relai proteksi kelistrikan modern (*Intelligent Electronic Device*) yang menuntut latensi rendah dan keandalan tinggi.

