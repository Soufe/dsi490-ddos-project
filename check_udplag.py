from pathlib import Path

import numpy as np
import pandas as pd

from data_prep import normalize_labels
from predict_ddos import load_inference_components

# เปลี่ยน Path ให้ตรงกับ Dataset ของคุณ
DATASET_DIR = Path("dataset/03-11")
CHUNKSIZE = 20000

_, _, feature_columns = load_inference_components()

csv_files = sorted(DATASET_DIR.rglob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {DATASET_DIR.resolve()}")

total_rows = 0
udplag_rows = 0
udplag_valid = 0
udplag_invalid = 0

print("\n=== UDPLag DATASET INSPECTION ===")

for file_path in csv_files:
    file_total = 0
    file_udplag = 0
    file_valid = 0
    file_invalid = 0

    for chunk in pd.read_csv(
        file_path,
        chunksize=CHUNKSIZE,
        low_memory=False,
    ):
        chunk.columns = chunk.columns.str.strip()
        file_total += len(chunk)

        normalized = normalize_labels(chunk["Label"].astype(str))

        mask = normalized.eq("UDPLag")
        selected = chunk.loc[mask]

        if selected.empty:
            continue

        features = selected[list(feature_columns)].apply(
            pd.to_numeric,
            errors="coerce",
        )

        features = features.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        valid_mask = features.notna().all(axis=1)

        file_udplag += len(selected)
        file_valid += int(valid_mask.sum())
        file_invalid += int((~valid_mask).sum())

    total_rows += file_total
    udplag_rows += file_udplag
    udplag_valid += file_valid
    udplag_invalid += file_invalid

    print(
        f"{file_path.name}: "
        f"total={file_total:,}, "
        f"UDPLag={file_udplag:,}, "
        f"valid={file_valid:,}, "
        f"invalid={file_invalid:,}"
    )

print("\n=== UDPLag SUMMARY ===")
print(f"Total dataset rows : {total_rows:,}")
print(f"UDPLag total       : {udplag_rows:,}")
print(f"UDPLag valid       : {udplag_valid:,}")
print(f"UDPLag invalid     : {udplag_invalid:,}")

if udplag_rows:
    print(f"UDPLag invalid %   : " f"{udplag_invalid / udplag_rows * 100:.4f}%")
    print(f"UDPLag dataset %   : " f"{udplag_rows / total_rows * 100:.6f}%")
