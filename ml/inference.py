import argparse
import os

import joblib
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Run IDS inference using saved model.pkl")
    parser.add_argument("--model", default="model.pkl", help="Path to trained model artifact")
    parser.add_argument("--input", required=True, help="Path to input CSV for prediction")
    parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Path to output CSV with predictions",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Rows per chunk while reading prediction input",
    )
    return parser.parse_args()


def clean_features(df, feature_cols):
    df.columns = df.columns.str.strip()
    X = df.reindex(columns=feature_cols)
    X = X.apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill missing values with 0 for robust batch inference.
    return X.fillna(0.0).astype(np.float32)


def predict_attack_scores(model, X):
    attack_index = int(np.where(model.classes_ == "ATTACK")[0][0])
    return model.predict_proba(X)[:, attack_index]


def predict_with_threshold(model, X, attack_threshold):
    attack_scores = predict_attack_scores(model, X)
    return np.where(attack_scores >= attack_threshold, "ATTACK", "BENIGN")


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    artifact = joblib.load(args.model)
    model = artifact["model"]
    feature_cols = artifact["feature_columns"]
    attack_threshold = artifact.get("attack_threshold", 0.2)

    if os.path.exists(args.output):
        os.remove(args.output)

    wrote_header = False
    total_rows = 0

    for chunk in pd.read_csv(args.input, chunksize=args.chunk_size, low_memory=True, skipinitialspace=True):
        X = clean_features(chunk, feature_cols)
        attack_scores = predict_attack_scores(model, X.to_numpy())
        y_pred = np.where(attack_scores >= attack_threshold, "ATTACK", "BENIGN")

        out_chunk = chunk.copy()
        out_chunk.columns = out_chunk.columns.str.strip()
        out_chunk["PredictedLabel"] = y_pred
        out_chunk["AttackScore"] = np.round(attack_scores, 6)

        out_chunk.to_csv(args.output, mode="a", index=False, header=not wrote_header)
        wrote_header = True
        total_rows += len(out_chunk)

    print(f"[DONE] Wrote {total_rows} predictions to {args.output}")


if __name__ == "__main__":
    main()
