
from flask import Flask, render_template, request, jsonify
from main_deploy import load_artifacts, predict_frame
import pandas as pd
import math

app = Flask(__name__)

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
    return response

# Memuat model satu kali saat Flask dimulai,
# bukan setiap tombol prediksi diklik.
MODEL, METADATA = load_artifacts()

# Mengambil urutan fitur dari metadata Reno.
FEATURES = METADATA["feature_order"]


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({
        "status": "ready",
        "model": METADATA["selected_model"],
        "classes": METADATA["class_names"]
    })


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({
            "error": "Kirim data dalam format JSON."
        }), 400

    # Validasi enam parameter kelistrikan.
    try:
        values = {}

        for feature in FEATURES:
            raw = payload[feature]

            if isinstance(raw, bool):
                raise ValueError("Input boolean tidak didukung")

            number = float(raw)

            if not math.isfinite(number):
                raise ValueError("Nilai tidak finite")

            values[feature] = number

    except (KeyError, ValueError, TypeError):
        return jsonify({
            "error": "Isi Ia, Ib, Ic, Va, Vb, Vc dengan angka valid."
        }), 400

    # Mengubah data JSON menjadi DataFrame
    # dengan urutan fitur sesuai training.
    frame = pd.DataFrame([values], columns=FEATURES)

    try:
        prediction = predict_frame(
            frame, MODEL, METADATA
        ).iloc[0]

    except Exception:
        app.logger.exception(
            "Model gagal melakukan prediksi"
        )

        return jsonify({
            "error": "Inferensi model gagal. Periksa terminal Flask."
        }), 500

    # Mengembalikan hasil prediksi ke website.
    return jsonify({
        "status": "success",
        "fault_code": str(prediction["fault_code"]),
        "fault_name": str(prediction["fault_name"]),
        "model": METADATA["selected_model"],
        "input": values
    })


if __name__ == "__main__":
    app.run(debug=True)
