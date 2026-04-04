import glob
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def preprocess_chunk(df, feature_cols, label_col) -> Tuple[pd.DataFrame, pd.Series]:
    # Normalize column names and replace non-finite values before filtering.
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Keep only expected feature columns and coerce everything to numeric.
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df[label_col].astype(str).str.strip()

    # Drop rows with any invalid feature value or missing label.
    valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    return X.astype(np.float32), y


def main():
    # Core runtime settings.
    dataset_dir = "../Datasets"
    model_output_path = "model.pkl"
    chunk_size = 100_000
    eval_ratio = 0.1
    max_train_rows = 400_000
    max_eval_rows = 100_000
    rng = np.random.default_rng(seed=42)

    # Discover all input CSV files.
    print(f"[INFO] Looking for CSV files in {dataset_dir}")
    csv_files = sorted(glob.glob(os.path.join(dataset_dir, "*.csv")))

    if not csv_files:
        print("[ERROR] No CSV files found. Check dataset path.")
        return

    print(f"[INFO] Found {len(csv_files)} CSV files")

    # Infer schema once from the first file. Last column is treated as the label.
    header = pd.read_csv(csv_files[0], nrows=0, skipinitialspace=True)
    header.columns = header.columns.str.strip()
    if len(header.columns) < 2:
        print("[ERROR] Dataset must have at least one feature column and one label column.")
        return

    label_col = header.columns[-1]
    feature_cols = list(header.columns[:-1])
    print(f"[INFO] Label column: {label_col}")
    print(f"[INFO] Number of feature columns: {len(feature_cols)}")

    # Pass 1: read all files in chunks and build bounded-size train/eval samples.
    print("[INFO] Pass 1/1: collecting bounded training/evaluation samples...")
    total_rows = 0
    train_rows = 0
    eval_rows = 0
    X_train_parts = []
    y_train_parts = []
    X_eval_parts = []
    y_eval_parts = []

    # Use per-class quotas so minority attacks are not dropped by random sampling.
    class_counts = {}
    class_train_counts = {}
    class_eval_counts = {}

    for i, file_path in enumerate(csv_files, start=1):
        print(f"[INFO] Processing file {i}/{len(csv_files)}: {os.path.basename(file_path)}")
        for chunk in pd.read_csv(
            file_path,
            chunksize=chunk_size,
            low_memory=True,
            skipinitialspace=True,
        ):
            # Apply per-chunk cleaning and type conversion.
            X, y = preprocess_chunk(chunk, feature_cols, label_col)
            if X.empty:
                continue

            total_rows += len(X)

            # Split each chunk into train/eval and cap collected rows to avoid RAM spikes.
            eval_mask = rng.random(len(X)) < eval_ratio
            train_mask = ~eval_mask

            if train_mask.any():
                X_train = X.loc[train_mask]
                y_train = y.loc[train_mask]

                # Update running class distribution using the current training slice.
                for cls_name, cnt in y_train.value_counts().items():
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + int(cnt)

                # Sample per-class rows with dynamic quotas to keep class balance under a global cap.
                for cls_name, grp_idx in y_train.groupby(y_train).groups.items():
                    if train_rows >= max_train_rows:
                        break
                    remaining_global = max_train_rows - train_rows

                    # Allocate quota proportionally to observed frequency, with a small floor.
                    cls_total = class_counts.get(cls_name, 1)
                    cls_target = max(50, int(max_train_rows * (cls_total / max(total_rows, 1))))
                    cls_seen = class_train_counts.get(cls_name, 0)
                    cls_remaining = max(0, cls_target - cls_seen)
                    if cls_remaining <= 0:
                        continue

                    take_n = min(len(grp_idx), cls_remaining, remaining_global)
                    if take_n <= 0:
                        continue

                    pick = rng.choice(np.asarray(grp_idx), size=take_n, replace=False)
                    X_train_parts.append(X_train.loc[pick].to_numpy())
                    y_train_parts.append(y_train.loc[pick].to_numpy())
                    class_train_counts[cls_name] = cls_seen + take_n
                    train_rows += take_n

            if eval_mask.any() and eval_rows < max_eval_rows:
                X_eval = X.loc[eval_mask]
                y_eval = y.loc[eval_mask]

                # Build a bounded validation set with minimum per-class representation.
                for cls_name, grp_idx in y_eval.groupby(y_eval).groups.items():
                    if eval_rows >= max_eval_rows:
                        break
                    remaining_global = max_eval_rows - eval_rows

                    cls_target = max(20, int(max_eval_rows * eval_ratio))
                    cls_seen = class_eval_counts.get(cls_name, 0)
                    cls_remaining = max(0, cls_target - cls_seen)
                    if cls_remaining <= 0:
                        continue

                    take_n = min(len(grp_idx), cls_remaining, remaining_global)
                    if take_n <= 0:
                        continue

                    pick = rng.choice(np.asarray(grp_idx), size=take_n, replace=False)
                    X_eval_parts.append(X_eval.loc[pick].to_numpy())
                    y_eval_parts.append(y_eval.loc[pick].to_numpy())
                    class_eval_counts[cls_name] = cls_seen + take_n
                    eval_rows += take_n

    if not X_train_parts:
        print("[ERROR] Training sample is empty. All rows may have been dropped during cleaning.")
        return

    print(f"[INFO] Total cleaned rows seen: {total_rows}")
    print(f"[INFO] Sampled rows used for training: {train_rows}")
    print(f"[INFO] Sampled rows used for evaluation: {eval_rows}")

    # Merge sampled chunks into contiguous arrays before model fitting.
    X_train_all = np.vstack(X_train_parts)
    y_train_all = np.concatenate(y_train_parts)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    print("[INFO] Training RandomForestClassifier...")
    model.fit(X_train_all, y_train_all)

    if X_eval_parts:
        # Evaluate on the sampled validation pool collected during chunk scanning.
        X_eval_all = np.vstack(X_eval_parts)
        y_eval_all = np.concatenate(y_eval_parts)
        y_hat = model.predict(X_eval_all)
        accuracy = accuracy_score(y_eval_all, y_hat)
        print(f"[RESULT] Streaming validation accuracy: {accuracy:.4f}")
    else:
        print("[WARN] No validation rows were sampled; accuracy not available.")

    # Save everything needed for consistent inference later.
    artifact = {
        "model": model,
        "feature_columns": feature_cols,
        "label_column": label_col,
        "algorithm": "RandomForestClassifier",
        "max_train_rows": max_train_rows,
        "max_eval_rows": max_eval_rows,
    }

    print(f"[INFO] Saving model to: {model_output_path}")
    joblib.dump(artifact, model_output_path)
    print("[DONE] Model artifact saved successfully as model.pkl")


if __name__ == "__main__":
    main()