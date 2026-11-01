
"""Inferensi mandiri: muat pipeline tanpa melatih ulang.
Contoh: python main_deploy.py --input contoh_input.csv --output hasil_prediksi.csv
Contoh: python main_deploy.py --values -151.2 -9.6 85.8 0.40 -0.13 -0.26
Jalankan hanya model dari sumber yang dipercaya.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import joblib

BASE = Path(__file__).resolve().parent

def load_artifacts(model_path=None, metadata_path=None):
    model_path = Path(model_path) if model_path else BASE / 'model_pipeline.joblib'
    metadata_path = Path(metadata_path) if metadata_path else BASE / 'metadata.json'
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    return model, metadata

def predict_frame(frame, model, metadata):
    features = metadata['feature_order']
    missing = [f for f in features if f not in frame.columns]
    if missing:
        raise ValueError('Kolom input belum lengkap: ' + ', '.join(missing))
    X = frame.loc[:, features].apply(pd.to_numeric, errors='raise')
    if X.empty or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('Input harus berisi angka hingga, tidak kosong, dan tanpa NaN/inf.')
    codes = model.predict(X)
    result = pd.DataFrame({'fault_code': codes})
    result['fault_name'] = result['fault_code'].map(metadata['class_names'])
    return result

def main():
    parser = argparse.ArgumentParser(description='Prediksi gangguan listrik tanpa training ulang.')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path, help='CSV dengan enam kolom fitur.')
    source.add_argument('--values', nargs=6, type=float, metavar='N', help='Urutan Ia Ib Ic Va Vb Vc.')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    model, metadata = load_artifacts(args.model, args.metadata)
    frame = (pd.read_csv(args.input) if args.input else
             pd.DataFrame([args.values], columns=metadata['feature_order']))
    result = predict_frame(frame, model, metadata)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
    print(result.to_json(orient='records', force_ascii=False))

if __name__ == '__main__':
    main()
