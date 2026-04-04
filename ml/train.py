import glob
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


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
    holdout_file_name = "Wednesday-workingHours.pcap_ISCX.csv"
    model_output_path = "model.pkl"
    report_output_path = "metrics_report.txt"
    confusion_matrix_output_path = "confusion_matrix.csv"
    per_class_metrics_output_path = "per_class_metrics.csv"
    chunk_size = 100_000
    max_holdout_eval_rows = 0
    max_train_rows = 400_000
    rng = np.random.default_rng(seed=42)

    # Discover all input CSV files.
    print(f"[INFO] Looking for CSV files in {dataset_dir}")
    csv_files = sorted(glob.glob(os.path.join(dataset_dir, "*.csv")))

    if not csv_files:
        print("[ERROR] No CSV files found. Check dataset path.")
        return

    print(f"[INFO] Found {len(csv_files)} CSV files")

    # Pick one unseen file/day for strict hold-out testing.
    holdout_file_path = next(
        (p for p in csv_files if os.path.basename(p) == holdout_file_name),
        csv_files[-1],
    )
    train_csv_files = [p for p in csv_files if p != holdout_file_path]

    if not train_csv_files:
        print("[ERROR] Need at least one training file besides the hold-out file.")
        return

    print(f"[INFO] Hold-out file: {os.path.basename(holdout_file_path)}")
    print(f"[INFO] Training files: {len(train_csv_files)}")

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

    # Pass 0: scan train labels only to estimate class distribution for balanced sampling.
    print("[INFO] Pass 0/1: discovering class distribution on training files...")
    class_total_counts = {}
    for file_path in train_csv_files:
        for y_chunk in pd.read_csv(
            file_path,
            usecols=lambda c: isinstance(c, str) and c.strip() == label_col,
            chunksize=chunk_size,
            low_memory=True,
            skipinitialspace=True,
        ):
            y_chunk.columns = y_chunk.columns.str.strip()
            labels = y_chunk[label_col].astype(str).str.strip()
            labels = labels[(labels != "") & (labels.str.lower() != "nan")]
            for cls_name, cnt in labels.value_counts().items():
                class_total_counts[cls_name] = class_total_counts.get(cls_name, 0) + int(cnt)

    class_names = sorted(class_total_counts.keys())
    if not class_names:
        print("[ERROR] No class labels found in dataset.")
        return

    per_class_train_target = max(1, max_train_rows // len(class_names))
    print(f"[INFO] Detected {len(class_names)} classes")
    print(f"[INFO] Per-class train target: {per_class_train_target}")

    # Pass 1: read training files in chunks and build bounded-size training samples.
    print("[INFO] Pass 1/1: collecting bounded training samples...")
    total_rows = 0
    train_rows = 0
    X_train_parts = []
    y_train_parts = []

    # Use balanced per-class quotas so minority attacks are represented.
    class_train_counts = {cls_name: 0 for cls_name in class_names}

    for i, file_path in enumerate(train_csv_files, start=1):
        print(f"[INFO] Processing train file {i}/{len(train_csv_files)}: {os.path.basename(file_path)}")
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

            # Sample per-class rows with fixed balanced quotas under a global cap.
            for cls_name, grp_idx in y.groupby(y).groups.items():
                if train_rows >= max_train_rows:
                    break
                remaining_global = max_train_rows - train_rows

                cls_seen = class_train_counts.get(cls_name, 0)
                cls_remaining = max(0, per_class_train_target - cls_seen)
                if cls_remaining <= 0:
                    continue

                take_n = min(len(grp_idx), cls_remaining, remaining_global)
                if take_n <= 0:
                    continue

                pick = rng.choice(np.asarray(grp_idx), size=take_n, replace=False)
                X_train_parts.append(X.loc[pick].to_numpy())
                y_train_parts.append(y.loc[pick].to_numpy())
                class_train_counts[cls_name] = cls_seen + take_n
                train_rows += take_n

    if not X_train_parts:
        print("[ERROR] Training sample is empty. All rows may have been dropped during cleaning.")
        return

    print(f"[INFO] Total cleaned rows seen: {total_rows}")
    print(f"[INFO] Sampled rows used for training: {train_rows}")

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

    # Strict hold-out evaluation on an unseen file/day.
    print(f"[INFO] Evaluating on hold-out file: {os.path.basename(holdout_file_path)}")
    y_true = []
    y_pred = []
    holdout_rows = 0

    for chunk in pd.read_csv(
        holdout_file_path,
        chunksize=chunk_size,
        low_memory=True,
        skipinitialspace=True,
    ):
        X_holdout, y_holdout = preprocess_chunk(chunk, feature_cols, label_col)
        if X_holdout.empty:
            continue

        if max_holdout_eval_rows > 0:
            remaining = max_holdout_eval_rows - holdout_rows
            if remaining <= 0:
                break
            if len(X_holdout) > remaining:
                X_holdout = X_holdout.iloc[:remaining]
                y_holdout = y_holdout.iloc[:remaining]

        y_hat = model.predict(X_holdout.to_numpy())
        y_true.extend(y_holdout.tolist())
        y_pred.extend(y_hat.tolist())
        holdout_rows += len(X_holdout)

    if not y_true:
        print("[WARN] Hold-out evaluation set is empty after preprocessing.")
    else:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        accuracy = accuracy_score(y_true_arr, y_pred_arr)
        f1_macro = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        report_text = str(classification_report(y_true_arr, y_pred_arr, zero_division=0))

        print(f"[RESULT] Hold-out accuracy: {accuracy:.4f}")
        print(f"[RESULT] Hold-out F1 (macro): {f1_macro:.4f}")
        print(f"[RESULT] Hold-out F1 (weighted): {f1_weighted:.4f}")
        print("[REPORT] Hold-out classification report:\n" + report_text)

        # Save class-focused metrics for detailed analysis of minority classes.
        report_dict = classification_report(y_true_arr, y_pred_arr, zero_division=0, output_dict=True)
        per_class_metrics_df = pd.DataFrame(report_dict).T
        per_class_metrics_df.to_csv(per_class_metrics_output_path)

        class_rows = per_class_metrics_df.loc[
            ~per_class_metrics_df.index.isin(["accuracy", "macro avg", "weighted avg"])
        ]
        class_rows = class_rows[class_rows["support"] > 0]
        worst_recall_rows = class_rows.sort_values("recall", ascending=True).head(5)
        print("[REPORT] Lowest-recall classes on hold-out:")
        for cls_name, row in worst_recall_rows.iterrows():
            print(
                f"  - {cls_name}: recall={row['recall']:.4f}, "
                f"precision={row['precision']:.4f}, f1={row['f1-score']:.4f}, "
                f"support={int(row['support'])}"
            )

        labels = model.classes_
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_df.to_csv(confusion_matrix_output_path)

        with open(report_output_path, "w", encoding="utf-8") as report_file:
            report_file.write("Hold-out validation metrics\n")
            report_file.write(f"holdout_file: {os.path.basename(holdout_file_path)}\n")
            report_file.write(f"rows_evaluated: {holdout_rows}\n")
            report_file.write(f"accuracy: {accuracy:.6f}\n")
            report_file.write(f"f1_macro: {f1_macro:.6f}\n")
            report_file.write(f"f1_weighted: {f1_weighted:.6f}\n\n")
            report_file.write("Classification report\n")
            report_file.write(report_text)

        print(f"[INFO] Saved classification report to: {report_output_path}")
        print(f"[INFO] Saved confusion matrix to: {confusion_matrix_output_path}")
        print(f"[INFO] Saved per-class metrics to: {per_class_metrics_output_path}")

    # Save everything needed for consistent inference later.
    artifact = {
        "model": model,
        "feature_columns": feature_cols,
        "label_column": label_col,
        "algorithm": "RandomForestClassifier",
        "max_train_rows": max_train_rows,
        "holdout_file": os.path.basename(holdout_file_path),
    }

    print(f"[INFO] Saving model to: {model_output_path}")
    joblib.dump(artifact, model_output_path)
    print("[DONE] Model artifact saved successfully as model.pkl")


if __name__ == "__main__":
    main()