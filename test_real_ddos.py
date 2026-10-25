import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_prep import normalize_labels
from predict_ddos import load_inference_components

# ============================================================
# 1. ARGUMENTS
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DNN on raw CICDDoS2019 CSV files"
    )

    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--chunksize", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("evaluation_real"))
    parser.add_argument(
        "--label-mapping",
        type=Path,
        default=Path("cleaned_dataset/label_mapping.json"),
    )

    return parser.parse_args()


# ============================================================
# 2. LABEL MAPPING
# ============================================================


def load_valid_labels(mapping_path):
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    if mapping.get("BENIGN") != 0:
        raise ValueError("Invalid label mapping: BENIGN must be 0")

    return set(mapping.keys())


def map_labels(series, valid_labels):
    """
    BENIGN = 0
    Known attack labels = 1
    Unknown labels = -1
    """
    normalized = normalize_labels(series.astype(str))
    valid = normalized.isin(valid_labels)

    targets = pd.Series(-1, index=series.index, dtype=int)
    targets.loc[valid] = (normalized.loc[valid] != "BENIGN").astype(int)

    return normalized, targets


# ============================================================
# 3. SAMPLE DATASET
# ============================================================


def sample_dataset(
    folder,
    feature_columns,
    valid_labels,
    total_samples,
    chunksize,
    seed,
):
    folder = Path(folder)

    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

    csv_files = sorted(
        p
        for p in folder.rglob("*.csv")
        if p.is_file() and not p.name.startswith((".~", "~$"))
    )

    if not csv_files:
        raise FileNotFoundError("No CSV files found")

    quotas = {
        0: total_samples // 2,
        1: total_samples - total_samples // 2,
    }

    reservoirs = {0: pd.DataFrame(), 1: pd.DataFrame()}
    rng = np.random.default_rng(seed)

    scanned = {0: 0, 1: 0}
    excluded_labels = set()
    file_statistics = []

    required = set(feature_columns) | {"Label"}

    for file_path in csv_files:
        print(f"Scanning: {file_path.name}", flush=True)

        try:
            header = pd.read_csv(file_path, nrows=0)
        except (pd.errors.EmptyDataError, UnicodeError) as exc:
            print(f"  Skipped: {exc}")
            continue

        header.columns = header.columns.str.strip()

        if header.columns.duplicated().any():
            print("  Skipped: duplicate columns")
            continue

        missing = sorted(required - set(header.columns))

        if missing:
            print(f"  Skipped: missing {len(missing)} features")
            file_statistics.append(
                {
                    "file": str(file_path),
                    "status": "missing_columns",
                    "rows": 0,
                    "benign": 0,
                    "attack": 0,
                    "unmapped": 0,
                    "missing_columns": "; ".join(missing),
                }
            )
            continue

        counts = {
            "rows": 0,
            "benign": 0,
            "attack": 0,
            "unmapped": 0,
        }

        reader = pd.read_csv(
            file_path,
            chunksize=chunksize,
            usecols=lambda c: c.strip() in required,
            low_memory=False,
        )

        for chunk in reader:
            chunk.columns = chunk.columns.str.strip()
            counts["rows"] += len(chunk)

            normalized, targets = map_labels(chunk["Label"], valid_labels)

            unknown_mask = targets.eq(-1)

            if unknown_mask.any():
                excluded_labels.update(normalized.loc[unknown_mask].unique().tolist())

            counts["unmapped"] += int(unknown_mask.sum())

            for target, name in ((0, "benign"), (1, "attack")):
                positions = np.flatnonzero(targets.to_numpy() == target)

                n = len(positions)
                counts[name] += n
                scanned[target] += n

                if n == 0:
                    continue

                selected = chunk.iloc[positions][list(feature_columns)].copy()

                selected["_label"] = target
                selected["_attack_type"] = normalized.iloc[positions].to_numpy()
                selected["_source_file"] = str(file_path)
                selected["_source_row"] = counts["rows"] - len(chunk) + positions + 2
                selected["_random_key"] = rng.random(n)

                combined = pd.concat(
                    [reservoirs[target], selected],
                    ignore_index=True,
                )

                reservoirs[target] = combined.nlargest(
                    quotas[target], "_random_key"
                ).reset_index(drop=True)

        file_statistics.append(
            {
                "file": str(file_path),
                "status": "scanned",
                **counts,
                "missing_columns": "",
            }
        )

    print("\n=== DATASET SCAN SUMMARY ===")
    print(f"BENIGN: {scanned[0]:,}")
    print(f"DDoS:   {scanned[1]:,}")

    if excluded_labels:
        print("\nUnmapped labels (excluded):")
        for label in sorted(excluded_labels):
            print(f" - {label}")
    else:
        print("\nUnmapped labels: none")

    if reservoirs[0].empty or reservoirs[1].empty:
        raise ValueError("Both BENIGN and Attack samples are required")

    sample = pd.concat(
        [reservoirs[0], reservoirs[1]],
        ignore_index=True,
    )

    sample = sample.sample(frac=1, random_state=seed).reset_index(drop=True)

    sample = sample.drop(columns="_random_key")

    if len(sample) < total_samples:
        print(
            f"[WARN] Requested {total_samples:,} samples, "
            f"but only {len(sample):,} available"
        )

    return sample, pd.DataFrame(file_statistics)


# ============================================================
# 4. FEATURE PREPROCESSING
# ============================================================


def prepare_features(sample, feature_columns, scaler):
    """
    Convert raw features to numeric and apply the
    original training scaler exactly once.
    """
    x = sample[list(feature_columns)].apply(pd.to_numeric, errors="coerce")

    x = x.replace([np.inf, -np.inf], np.nan)

    if x.isna().any().any():
        raise ValueError("Invalid feature values remain after filtering")

    scaled = scaler.transform(x)

    if not np.isfinite(scaled).all():
        raise ValueError("Non-finite values found after scaling")

    return np.asarray(scaled, dtype=np.float32)


# ============================================================
# 5. DNN PREDICTION
# ============================================================


def predict_in_batches(model, x, batch_size):
    model.eval()
    probabilities = []

    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size])

            logits = model(batch)

            probs = torch.sigmoid(logits).reshape(-1).cpu().numpy()

            probabilities.append(probs)

    return np.concatenate(probabilities)


# ============================================================
# 6. MAIN EVALUATION
# ============================================================


def main():
    args = parse_args()
    started = time.perf_counter()

    if args.samples < 2 or args.chunksize < 1 or args.batch_size < 1:
        raise ValueError("Invalid sample, chunk or batch size")

    valid_labels = load_valid_labels(args.label_mapping)

    model, scaler, feature_columns = load_inference_components()

    print("\n=== MODEL INFORMATION ===")
    print(f"Input features: {len(feature_columns)}")

    if len(feature_columns) != 67:
        print(f"[WARN] Expected 67 features, " f"found {len(feature_columns)}")

    if scaler is None:
        raise ValueError("Saved training scaler is required")

    # Sample raw CSVs
    sample, file_stats = sample_dataset(
        folder=args.dataset,
        feature_columns=feature_columns,
        valid_labels=valid_labels,
        total_samples=args.samples,
        chunksize=args.chunksize,
        seed=args.seed,
    )

    # --------------------------------------------------------
    # Remove rows containing NaN, Inf or non-numeric features
    # --------------------------------------------------------
    numeric_features = sample[list(feature_columns)].apply(
        pd.to_numeric,
        errors="coerce",
    )

    numeric_features = numeric_features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    valid_mask = numeric_features.notna().all(axis=1)
    removed = int((~valid_mask).sum())

    print(f"\nInvalid rows removed: {removed:,}")

    sample = sample.loc[valid_mask].reset_index(drop=True)

    if sample.empty:
        raise ValueError("No valid samples remain after cleaning")

    # Preprocess features using the original scaler
    x = prepare_features(
        sample,
        feature_columns,
        scaler,
    )

    y_true = sample["_label"].to_numpy(dtype=int)

    # Predict
    probabilities = predict_in_batches(
        model,
        x,
        args.batch_size,
    )

    y_pred = (probabilities >= 0.5).astype(int)

    # Metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    fpr = float(fp / (fp + tn)) if (fp + tn) else None

    auc = (
        float(roc_auc_score(y_true, probabilities))
        if len(np.unique(y_true)) == 2
        else None
    )

    elapsed = time.perf_counter() - started

    print("\n=== REAL DATA DNN EVALUATION ===")
    print(f"Samples   : {len(y_true):,}")
    print(f"Accuracy  : {accuracy:.6f}")
    print(f"Precision : {precision:.6f}")
    print(f"Recall    : {recall:.6f}")
    print(f"F1-score  : {f1:.6f}")
    print(f"FPR       : {fpr:.6f}" if fpr is not None else "FPR       : N/A")
    print(f"ROC AUC   : {auc:.6f}" if auc is not None else "ROC AUC   : N/A")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    print(
        "\n"
        + classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["BENIGN", "DDoS"],
            zero_division=0,
        )
    )

    print(f"Elapsed: {elapsed:.2f} seconds")

    # ========================================================
    # 7. SAVE RESULTS
    # ========================================================

    args.output.mkdir(parents=True, exist_ok=True)

    predictions = sample[
        [
            "_source_file",
            "_source_row",
            "_attack_type",
            "_label",
        ]
    ].copy()

    predictions.columns = [
        "source_file",
        "source_row",
        "attack_type",
        "true_label",
    ]

    predictions["predicted_label"] = y_pred
    predictions["ddos_probability"] = probabilities
    predictions["correct"] = y_true == y_pred

    predictions.to_csv(
        args.output / "predictions.csv",
        index=False,
    )

    file_stats.to_csv(
        args.output / "file_statistics.csv",
        index=False,
    )

    attack_summary = (
        predictions.groupby("attack_type", dropna=False)
        .agg(
            samples=("correct", "size"),
            correct=("correct", "sum"),
            avg_ddos_probability=("ddos_probability", "mean"),
        )
        .reset_index()
    )

    attack_summary["accuracy"] = attack_summary["correct"] / attack_summary["samples"]

    attack_summary.to_csv(
        args.output / "attack_type_summary.csv",
        index=False,
    )

    metrics = {
        "samples": len(y_true),
        "benign_samples": int((y_true == 0).sum()),
        "attack_samples": int((y_true == 1).sum()),
        "invalid_rows_removed": removed,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "fpr": fpr,
        "roc_auc": auc,
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        },
        "elapsed_seconds": elapsed,
        "seed": args.seed,
        "note": (
            "Approximately balanced sample from raw CSVs. "
            "Invalid feature rows removed before evaluation."
        ),
    }

    with open(
        args.output / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved evaluation files to: " f"{args.output.resolve()}")


if __name__ == "__main__":
    main()
