#!/usr/bin/env python3
"""
benchmark_jetson.py
===================
Script untuk menguji performa & latensi inferensi pada NVIDIA Jetson Nano.
Hasil pengujian dapat langsung disertakan dalam Laporan Tugas (5-8 halaman)
dan video presentasi pada bagian: "Pengukuran Latensi Inferensi Edge".
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from main_deploy import load_artifacts, predict_frame

BASE_DIR = Path(__file__).resolve().parent

def benchmark(n_runs=200):
    print("=" * 65)
    print("BENCHMARK LATENSI INFERENSI - NVIDIA JETSON NANO")
    print("=" * 65)
    
    t0 = time.perf_counter()
    model, metadata = load_artifacts()
    load_time = (time.perf_counter() - t0) * 1000
    print(f"Waktu load model & metadata : {load_time:.2f} ms")
    print(f"Model terpilih              : {metadata.get('selected_model', 'KNN')}")
    print(f"Fitur input (6 parameter)  : {', '.join(metadata['feature_order'])}")
    print("-" * 65)

    features = metadata['feature_order']
    sample_df = pd.DataFrame([[-151.2, -9.6, 85.8, 0.40, -0.13, -0.26]], columns=features)

    # Warmup
    _ = predict_frame(sample_df, model, metadata)

    latencies = []
    for _ in range(n_runs):
        t_start = time.perf_counter()
        _ = predict_frame(sample_df, model, metadata)
        t_end = time.perf_counter()
        latencies.append((t_end - t_start) * 1000)

    latencies = np.array(latencies)
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    p50_lat = np.percentile(latencies, 50)
    p95_lat = np.percentile(latencies, 95)
    throughput = 1000.0 / mean_lat

    print(f"HASIL PENGUJIAN SINGLE-SAMPLE INFERENCE ({n_runs} iterasi):")
    print(f"  • Rata-rata Latensi (Mean) : {mean_lat:.3f} ms")
    print(f"  • Standar Deviasi          : {std_lat:.3f} ms")
    print(f"  • Median (P50)             : {p50_lat:.3f} ms")
    print(f"  • 95th Percentile (P95)    : {p95_lat:.3f} ms")
    print(f"  • Throughput Inferensi     : {throughput:.1f} sampel / detik")
    print("-" * 65)

    # Batch test
    contoh_path = BASE_DIR / "contoh_input.csv"
    if not contoh_path.exists():
        contoh_path = BASE_DIR / "data" / "contoh_input.csv"

    if contoh_path.exists():
        batch_df = pd.read_csv(contoh_path)
        batch_latencies = []
        for _ in range(50):
            t_start = time.perf_counter()
            _ = predict_frame(batch_df, model, metadata)
            t_end = time.perf_counter()
            batch_latencies.append((t_end - t_start) * 1000)
        
        mean_batch = np.mean(batch_latencies)
        per_sample_batch = mean_batch / len(batch_df)
        print(f"HASIL BATCH INFERENCE ({len(batch_df)} sampel per batch, 50 run):")
        print(f"  • Rata-rata Latensi Batch  : {mean_batch:.3f} ms")
        print(f"  • Rata-rata per Sampel     : {per_sample_batch:.3f} ms")
        print("-" * 65)

    print("\nRINGKASAN UNTUK LAPORAN / VIDEO:")
    print(f"| Metrik Evaluasi Edge | Nilai di Jetson Nano |")
    print(f"|---|---|")
    print(f"| Waktu Load Model     | {load_time:.2f} ms |")
    print(f"| Rata-rata Latensi    | {mean_lat:.2f} ± {std_lat:.2f} ms |")
    print(f"| Latensi P95          | {p95_lat:.2f} ms |")
    print(f"| Throughput Maksimal  | ~{int(throughput)} pred / detik |")
    print("=" * 65)

if __name__ == '__main__':
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    benchmark(n_runs=runs)
