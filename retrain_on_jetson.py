#!/usr/bin/env python3
"""
retrain_on_jetson.py
====================
Script mandiri untuk melatih ulang pipeline klasifikasi gangguan listrik
langsung di Jetson Nano menggunakan versi scikit-learn lokal.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
try:
    from sklearn.impute import SimpleImputer
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

BASE_DIR = Path(__file__).resolve().parent

def main():
    print("=" * 60)
    print("JETSON NANO - RETRAIN PIPELINE KLASIFIKASI GANGGUAN LISTRIK")
    print("=" * 60)
    import sklearn
    print(f"Python      : {sys.version.split()[0]}")
    print(f"Scikit-Learn: {sklearn.__version__}")
    print("-" * 60)

    # Cek lokasi file (di root atau di subfolder data/)
    data_path = BASE_DIR / "classData.csv"
    if not data_path.exists():
        data_path = BASE_DIR / "data" / "classData.csv"

    manifest_path = BASE_DIR / "split_manifest.csv"
    if not manifest_path.exists():
        manifest_path = BASE_DIR / "data" / "split_manifest.csv"

    metadata_path = BASE_DIR / "metadata.json"

    if not data_path.exists() or not manifest_path.exists():
        print("ERROR: classData.csv atau split_manifest.csv tidak ditemukan!")
        sys.exit(1)

    print(f"1. Memuat dataset dari: {data_path.name}...")
    df = pd.read_csv(data_path)
    manifest = pd.read_csv(manifest_path)

    features = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']
    label_cols = ['G', 'C', 'B', 'A']
    df['fault_code'] = df[label_cols].astype(str).agg(''.join, axis=1)

    train_mask = manifest['split'] == 'train'
    test_rand_mask = manifest['split'] == 'test_random'
    test_block_mask = manifest['split'] == 'test_block'

    X_train = df.loc[train_mask, features]
    y_train = df.loc[train_mask, 'fault_code']

    X_test_rand = df.loc[test_rand_mask, features]
    y_test_rand = df.loc[test_rand_mask, 'fault_code']

    # 2. Pipeline training
    print("2. Melatih model KNN (k=15)...")
    start_t = time.perf_counter()
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier(n_neighbors=15, weights='uniform'))
    ])

    pipeline.fit(X_train, y_train)
    train_duration = (time.perf_counter() - start_t) * 1000
    print(f"   Selesai dalam {train_duration:.2f} ms!")

    # 3. Evaluasi
    pred_rand = pipeline.predict(X_test_rand)
    acc_rand = accuracy_score(y_test_rand, pred_rand)
    f1_rand = f1_score(y_test_rand, pred_rand, average='macro')
    print(f"3. [Test Set] Accuracy: {acc_rand*100:.2f}% | Macro F1: {f1_rand:.4f}")

    # 4. Simpan model_pipeline.joblib
    out_model = BASE_DIR / "model_pipeline.joblib"
    joblib.dump(pipeline, out_model, compress=3)
    print(f"4. Model berhasil disimpan: {out_model.name}")

    # Sinkronisasi ke phase-ai jika ada
    phase_ai_dir = BASE_DIR / "phase-ai"
    if phase_ai_dir.exists():
        joblib.dump(pipeline, phase_ai_dir / "model_pipeline.joblib", compress=3)

    print("-" * 60)
    print("SELESAI! Model siap digunakan untuk inferensi.")
    print("=" * 60)

if __name__ == '__main__':
    main()
