# Paket Klasifikasi Gangguan Listrik Tiga Fase

## Cara menjalankan
Lingkungan pembuat: Python 3.13.18, scikit-learn 1.6.1.
1. Siapkan lingkungan Python yang kompatibel dengan versi paket pada requirements.txt.
2. Pasang dependensi: python -m pip install -r requirements.txt
3. Jalankan: python main_deploy.py --input contoh_input.csv --output hasil_prediksi.csv
4. Untuk input angka: python main_deploy.py --values <Ia> <Ib> <Ic> <Va> <Vb> <Vc>

Jangan mengetik tanda < >; ganti dengan angka. main_deploy.py tidak melatih ulang model.
Untuk integrasi API: model, metadata = load_artifacts(); kemudian panggil
predict_frame(frame, model, metadata) dan ubah hasilnya menjadi JSON.

## Berkas
- model_pipeline.joblib: pipeline preprocessing dan model yang dievaluasi.
- metadata.json: urutan fitur, kelas, versi paket, hash, parameter, dan metrik.
- main_deploy.py: inferensi CSV/angka mandiri.
- contoh_input.csv: satu sampel uji per kelas; bukan label untuk dimasukkan ke model.
- contoh_label_referensi.csv: jawaban acuan untuk contoh, bukan input model.
- metrik_holdout.csv, classification_report_*.csv, prediksi_*.csv: hasil evaluasi.
- split_manifest.csv: pembagian baris setelah pembersihan.
- perbandingan_cv.csv, hasil_tuning.csv, *.png: bahan analisis laporan.
- ringkasan_hasil.txt: ringkasan hasil yang dihitung saat eksekusi.

## Pembagian kerja
Reno: dataset dan model. Arif: web/API. Bryan: Jetson dan deployment.
Anggota 4: preprocessing/fitur. Anggota 5: evaluasi/validasi.
Anggota 6: integrasi, dokumentasi, laporan 5–8 halaman, dan video YouTube.
Semua anggota memahami kode dan menulis penjelasan bagiannya.

## Batasan dan reproduksibilitas
Sumber: https://www.kaggle.com/datasets/esathyaprakash/electrical-fault-detection-and-classification
File: dataset.zip::classData.csv; SHA-256: c8e7d1f2a6d4aefd74b5ed48d9ca4afdfdf2a5b676a5ed0935625e48972a36b2
Data simulasi; unit/skala sensor harus dikonfirmasi, tidak boleh langsung disamakan dengan sensor fisik.
Holdout acak dan holdout blok tidak digunakan untuk training atau tuning; model ekspor sama
dengan model evaluasi. Nomor baris bukan timestamp atau ID kejadian.
Perlu validasi pada kejadian dan sensor baru sebelum klaim kemampuan di lapangan.
Bryan perlu menguji kompatibilitas paket dengan OS/JetPack perangkat. Jika versi ini tidak
tersedia pada Jetson, buat ulang lingkungan training yang kompatibel dan latih ulang;
jangan menganggap file joblib mendukung perbedaan versi scikit-learn.
Gunakan model joblib hanya dari sumber yang dipercaya.
Sertakan notebook .ipynb dari Colab, laporan, dan tautan video saat mengumpulkan tugas.
