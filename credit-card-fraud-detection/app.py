import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from preprocessing import RAW_FEATURES  # needed so joblib can load the pipeline

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model", "fraud_model.pkl")
OPTIONS_PATH = os.path.join(BASE, "model", "options.json")

app = Flask(__name__)

if not os.path.exists(MODEL_PATH):
    raise SystemExit("Model not found. Run: python training/train_model.py")

model = joblib.load(MODEL_PATH)
with open(OPTIONS_PATH) as f:
    options = json.load(f)


@app.route("/")
def index():
    return render_template("index.html", options=options)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    missing = [c for c in RAW_FEATURES if str(data.get(c, "")).strip() == ""]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        row = pd.DataFrame([{c: data[c] for c in RAW_FEATURES}])
        fraud_prob = float(model.predict_proba(row)[0][1])
    except Exception as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    is_fraud = fraud_prob >= 0.5
    return jsonify({
        "prediction": "FRAUDULENT" if is_fraud else "LEGITIMATE",
        "fraud_probability": round(fraud_prob, 4),
    })


if __name__ == "__main__":
    app.run(debug=True)
