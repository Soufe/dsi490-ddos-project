"""Evaluate DNN on raw CICDDoS2019 CSV files.

Sampling modes:
    balanced   : 50% BENIGN / 50% DDoS
    stratified : up to --per-type samples per attack type

Uses the existing trained DNN and scaler.
"""

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
    parser = argparse.ArgumentParser(description="Evaluate DNN on raw CICDDoS2019 CSVs")

    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--sampling",
        choices=("balanced", "stratified"),
        default="balanced",
    )
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--per-type", type=int, default=10000)
    parser.add_argument("--chunksize", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_real"),
    )

    parser.add_argument(
        "--label-mapping",
        type=Path,
        default=Path("cleaned_dataset/label_mapping.json"),
    )

    return parser.parse_args()


# ============================================================
# 2. LABEL MAPPING
# ============================================================


def load_valid_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    if mapping.get("BENIGN") != 0:
        raise ValueError("BENIGN must be mapped to 0")

    return set(mapping)


def map_labels(series, valid_labels):
    normalized = normalize_labels(series.astype(str))

    if not isinstance(normalized, pd.Series):
        normalized = pd.Series(normalized, index=series.index)

    normalized = normalized.astype(str)
    valid = normalized.isin(valid_labels)

    binary = pd.Series(-1, index=series.index, dtype="int8")

    binary.loc[valid] = (normalized.loc[valid] != "BENIGN").astype("int8")

    return normalized, binary


# ============================================================
# 3. DATASET SAMPLING
# ============================================================


def sample_dataset(
    folder,
    feature_columns,
    valid_labels,
    sampling,
    samples,
    per_type,
    chunksize,
    seed,
):
    folder = Path(folder)

    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset not found: {folder.resolve()}")

    csv_files = sorted(
        p
        for p in folder.rglob("*.csv")
        if p.is_file() and not p.name.startswith((".~", "~$"))
    )

    if not csv_files:
        raise FileNotFoundError("No CSV files found")

    if sampling == "balanced":
        quotas = {
            "BENIGN": samples // 2,
            "ATTACK": samples - samples // 2,
        }
    else:
        quotas = {label: per_type for label in sorted(valid_labels)}

    reservoirs = {key: pd.DataFrame() for key in quotas}

    rng = np.random.default_rng(seed)

    source_counts = {label: 0 for label in sorted(valid_labels)}

    unknown_counts = {}
    file_statistics = []

    required = set(feature_columns) | {"Label"}

    for file_path in csv_files:
        print(f"Scanning: {file_path.name}", flush=True)

        try:
            header = pd.read_csv(file_path, nrows=0)
        except (pd.errors.EmptyDataError, UnicodeError) as exc:
            print(f"Skipped: {exc}")

            file_statistics.append(
                {
                    "file": str(file_path.resolve()),
                    "status": "unreadable",
                    "rows": 0,
                    "benign": 0,
                    "attack": 0,
                    "unmapped": 0,
                }
            )
            continue

        header.columns = header.columns.str.strip()

        if header.columns.duplicated().any() or not required.issubset(header.columns):
            missing = sorted(required - set(header.columns))
            print(f"Skipped invalid columns: {missing[:8]}")

            file_statistics.append(
                {
                    "file": str(file_path.resolve()),
                    "status": "invalid_columns",
                    "rows": 0,
                    "benign": 0,
                    "attack": 0,
                    "unmapped": 0,
                }
            )
            continue

        file_rows = 0
        file_benign = 0
        file_attack = 0
        file_unmapped = 0

        reader = pd.read_csv(
            file_path,
            chunksize=chunksize,
            usecols=lambda c: c.strip() in required,
            low_memory=False,
        )

        for chunk in reader:
            chunk.columns = chunk.columns.str.strip()

            normalized, binary = map_labels(
                chunk["Label"],
                valid_labels,
            )

            file_benign += int(binary.eq(0).sum())
            file_attack += int(binary.eq(1).sum())
            file_unmapped += int(binary.eq(-1).sum())

            for label, count in normalized.value_counts().items():
                if label in valid_labels:
                    source_counts[label] += int(count)
                else:
                    unknown_counts[label] = unknown_counts.get(label, 0) + int(count)

            if sampling == "balanced":
                labels = normalized.where(
                    binary.eq(0),
                    "ATTACK",
                )
                labels = labels.where(binary.ne(-1), "UNKNOWN")
            else:
                labels = normalized

            label_array = labels.to_numpy()

            for key, quota in quotas.items():
                if quota == 0:
                    continue

                positions = np.flatnonzero(label_array == key)

                if len(positions) == 0:
                    continue

                selected = chunk.iloc[positions][list(feature_columns)].copy()

                selected["_label"] = binary.iloc[positions].to_numpy(dtype=int)

                selected["_attack_type"] = normalized.iloc[positions].to_numpy()

                selected["_source_file"] = str(file_path.resolve())

                selected["_source_row"] = file_rows + positions + 2

                selected["_random_key"] = rng.random(len(positions))

                combined = pd.concat(
                    [reservoirs[key], selected],
                    ignore_index=True,
                )

                reservoirs[key] = combined.nlargest(
                    quota,
                    "_random_key",
                ).reset_index(drop=True)

            file_rows += len(chunk)

        file_statistics.append(
            {
                "file": str(file_path.resolve()),
                "status": "scanned",
                "rows": file_rows,
                "benign": file_benign,
                "attack": file_attack,
                "unmapped": file_unmapped,
            }
        )

    print("\n=== DATASET SCAN SUMMARY ===")

    print(f"BENIGN: {source_counts.get('BENIGN', 0):,}")

    total_attack = sum(
        count for label, count in source_counts.items() if label != "BENIGN"
    )

    print(f"DDoS:   {total_attack:,}")

    if unknown_counts:
        print(f"Unmapped labels: {unknown_counts}")
    else:
        print("Unmapped labels: none")

    print("\n=== SOURCE ATTACK TYPE COUNTS ===")

    for label, count in source_counts.items():
        print(f"{label}: {count:,}")

    parts = [df for df in reservoirs.values() if not df.empty]

    if not parts:
        raise ValueError("No recognized samples found")

    sample = pd.concat(
        parts,
        ignore_index=True,
    )

    sample = sample.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    sample = sample.drop(columns="_random_key")

    return (
        sample,
        pd.DataFrame(file_statistics),
        source_counts,
        unknown_counts,
    )


# ============================================================
# 4. DATA CLEANING
# ============================================================


def clean_sample(sample, feature_columns):
    numeric = sample[list(feature_columns)].apply(
        pd.to_numeric,
        errors="coerce",
    )

    numeric = numeric.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    valid_mask = numeric.notna().all(axis=1)

    invalid_by_type = (
        sample.loc[~valid_mask, "_attack_type"]
        .value_counts()
        .rename_axis("attack_type")
        .reset_index(name="invalid_rows")
    )

    cleaned = sample.loc[valid_mask].copy().reset_index(drop=True)

    cleaned.loc[:, list(feature_columns)] = numeric.loc[valid_mask].to_numpy()

    return (
        cleaned,
        int((~valid_mask).sum()),
        invalid_by_type,
    )


# ============================================================
# 5. FEATURE PREPROCESSING
# ============================================================


def prepare_features(sample, feature_columns, scaler):
    features = sample[list(feature_columns)].astype("float64")

    scaled = scaler.transform(features)

    if not np.isfinite(scaled).all():
        raise ValueError("Non-finite values after applying training scaler")

    return np.asarray(
        scaled,
        dtype=np.float32,
    )


# ============================================================
# 6. DNN PREDICTION
# ============================================================


def predict_in_batches(model, features, batch_size):
    model.eval()

    device = next(model.parameters()).device
    probabilities = []

    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)

            logits = model(batch)

            probs = torch.sigmoid(logits).reshape(-1).cpu().numpy()

            probabilities.append(probs)

    return np.concatenate(probabilities)


# ============================================================
# 7. MAIN
# ============================================================


def main():
    args = parse_args()

    if (
        args.samples < 2
        or args.per_type < 1
        or args.chunksize < 1
        or args.batch_size < 1
    ):
        raise ValueError("Invalid sampling parameters")

    started = time.perf_counter()

    valid_labels = load_valid_labels(args.label_mapping)

    model, scaler, feature_columns = load_inference_components()

    if scaler is None:
        raise ValueError("Saved training scaler is required")

    print("\n=== MODEL INFORMATION ===")
    print(f"Input features: {len(feature_columns)}")
    print(f"Sampling: {args.sampling}")

    if len(feature_columns) != 67:
        print(f"[WARN] Expected 67 features, " f"found {len(feature_columns)}")

    # Sample dataset
    sample, file_stats, source_counts, unknown_counts = sample_dataset(
        folder=args.dataset,
        feature_columns=feature_columns,
        valid_labels=valid_labels,
        sampling=args.sampling,
        samples=args.samples,
        per_type=args.per_type,
        chunksize=args.chunksize,
        seed=args.seed,
    )

    selected_counts = sample["_attack_type"].value_counts()

    # Clean invalid rows
    sample, removed, invalid_by_type = clean_sample(
        sample,
        feature_columns,
    )

    print(f"\nInvalid rows removed: {removed:,}")

    if sample.empty:
        raise ValueError("No valid samples remain after cleaning")

    if sample["_label"].nunique() < 2:
        raise ValueError("Need both BENIGN and DDoS for evaluation")

    # Prepare features
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

    # Confusion Matrix
    tn, fp, fn, tp = (
        int(v)
        for v in confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()
    )

    # Metrics
    metrics = {
        "sampling": args.sampling,
        "requested_samples": (args.samples if args.sampling == "balanced" else None),
        "per_type_limit": (args.per_type if args.sampling == "stratified" else None),
        "selected_before_cleaning": int(len(sample) + removed),
        "invalid_rows_removed": removed,
        "samples": int(len(sample)),
        "benign_samples": int((y_true == 0).sum()),
        "attack_samples": int((y_true == 1).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "fpr": (float(fp / (fp + tn)) if fp + tn else None),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": {
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        },
        "seed": args.seed,
        "dataset": str(args.dataset.resolve()),
        "note": (
            "Rows with non-finite features are excluded. "
            "Training-only imputation medians were unavailable."
        ),
    }

    metrics["elapsed_seconds"] = float(time.perf_counter() - started)

    # Print evaluation
    print("\n=== REAL DATA DNN EVALUATION ===")
    print(f"Samples   : {metrics['samples']:,}")

    for name in (
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "fpr",
        "roc_auc",
    ):
        value = metrics[name]

        if value is not None:
            print(f"{name:<10}: {value:.6f}")
        else:
            print(f"{name}: N/A")

    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["BENIGN", "DDoS"],
            zero_division=0,
            digits=4,
        )
    )

    print(f"Elapsed: {metrics['elapsed_seconds']:.2f} seconds")

    # ========================================================
    # 8. SAVE RESULTS
    # ========================================================

    output = args.output / args.sampling
    output.mkdir(parents=True, exist_ok=True)

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
        output / "predictions.csv",
        index=False,
    )

    file_stats.to_csv(
        output / "file_statistics.csv",
        index=False,
    )

    attack_summary = (
        predictions.groupby(
            "attack_type",
            dropna=False,
        )
        .agg(
            samples=("correct", "size"),
            correct=("correct", "sum"),
            avg_ddos_probability=(
                "ddos_probability",
                "mean",
            ),
        )
        .reset_index()
    )

    attack_summary["accuracy"] = attack_summary["correct"] / attack_summary["samples"]

    attack_summary["missed_or_false_alarm"] = (
        attack_summary["samples"] - attack_summary["correct"]
    )

    attack_summary.to_csv(
        output / "attack_type_summary.csv",
        index=False,
    )

    # Sampling summary
    counts_df = pd.DataFrame({"attack_type": sorted(valid_labels)})

    counts_df["source_rows"] = (
        counts_df["attack_type"].map(source_counts).fillna(0).astype(int)
    )

    counts_df["selected_before_cleaning"] = (
        counts_df["attack_type"].map(selected_counts).fillna(0).astype(int)
    )

    if invalid_by_type.empty:
        invalid_counts = pd.Series(dtype=int)
    else:
        invalid_counts = invalid_by_type.set_index("attack_type")["invalid_rows"]

    counts_df["invalid_removed"] = (
        counts_df["attack_type"].map(invalid_counts).fillna(0).astype(int)
    )

    counts_df["evaluated"] = (
        counts_df["selected_before_cleaning"] - counts_df["invalid_removed"]
    )

    counts_df.to_csv(
        output / "sampling_summary.csv",
        index=False,
    )

    invalid_by_type.to_csv(
        output / "invalid_by_type.csv",
        index=False,
    )

    with open(
        output / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved evaluation to: {output.resolve()}")

    print("Note: Overall metrics depend on sampling distribution.")


if __name__ == "__main__":
    main()
