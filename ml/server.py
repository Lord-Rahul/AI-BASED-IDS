import os

import joblib
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from inference import clean_features, predict_attack_scores, predict_with_threshold


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
MODEL_PATH = os.environ.get("IDS_MODEL_PATH", os.path.join(BASE_DIR, "model.pkl"))


def load_artifact():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


artifact = load_artifact()
model = artifact["model"]
feature_cols = artifact["feature_columns"]
attack_threshold = artifact.get("attack_threshold", 0.2)
feature_importances = getattr(model, "feature_importances_", None)
top_feature_names = []
if feature_importances is not None:
    top_feature_indices = np.argsort(feature_importances)[::-1][:8]
    top_feature_names = [feature_cols[index] for index in top_feature_indices]

app = Flask(__name__)


def build_attack_details(cleaned_frame, predictions, attack_scores):
    attack_mask = predictions == "ATTACK"
    attack_indices = np.flatnonzero(attack_mask)
    details = []

    for index in attack_indices:
        attack_score = float(attack_scores[index])
        if attack_score >= 0.8:
            risk = "critical"
        elif attack_score >= 0.6:
            risk = "high"
        elif attack_score >= 0.4:
            risk = "medium"
        else:
            risk = "low"

        feature_snapshot = {}
        for feature_name in top_feature_names:
            feature_snapshot[feature_name] = round(float(cleaned_frame.iloc[index][feature_name]), 4)

        details.append(
            {
                "row_number": int(index + 1),
                "predicted_label": "ATTACK",
                "attack_score": round(attack_score, 6),
                "risk": risk,
                "feature_snapshot": feature_snapshot,
            }
        )

    return details


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/app.js")
def app_js():
    return send_from_directory(WEB_DIR, "app.js")


@app.get("/styles.css")
def styles_css():
    return send_from_directory(WEB_DIR, "styles.css")


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_path": MODEL_PATH,
            "attack_threshold": attack_threshold,
            "classes": model.classes_.tolist(),
        }
    )


@app.post("/predict")
def predict():
    uploaded_file = request.files.get("file")
    if uploaded_file is None:
        return jsonify({"error": "Upload a CSV file in form field 'file'."}), 400

    frame = pd.read_csv(uploaded_file, low_memory=True, skipinitialspace=True)
    if frame.empty:
        return jsonify({"error": "Uploaded CSV is empty."}), 400

    cleaned = clean_features(frame, feature_cols)
    feature_matrix = cleaned.to_numpy()
    attack_scores = predict_attack_scores(model, feature_matrix)
    predictions = predict_with_threshold(model, feature_matrix, attack_threshold)
    attack_details = build_attack_details(cleaned, predictions, attack_scores)
    attack_average_score = float(np.mean(attack_scores[predictions == "ATTACK"])) if attack_details else 0.0

    response_frame = frame.copy()
    response_frame.columns = response_frame.columns.str.strip()
    response_frame["PredictedLabel"] = predictions
    response_frame["AttackScore"] = np.round(attack_scores, 6)

    return jsonify(
        {
            "rows": int(len(response_frame)),
            "attack_rows": int((response_frame["PredictedLabel"] == "ATTACK").sum()),
            "benign_rows": int((response_frame["PredictedLabel"] == "BENIGN").sum()),
            "attack_average_score": round(attack_average_score, 6),
            "attack_threshold": attack_threshold,
            "top_features": top_feature_names,
            "predictions": response_frame["PredictedLabel"].tolist(),
            "attack_scores": np.round(attack_scores, 6).tolist(),
            "attack_details": attack_details,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)