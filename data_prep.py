import os
import glob
import json
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Metadata columns to drop from raw CICDDoS2019 CSVs
DROP_METADATA_COLS = [
    'Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port',
    'Destination IP', 'Destination Port', 'Timestamp', 'SimillarHTTP'
]

# Static constant / all-zero columns commonly found in CICDDoS2019
STATIC_CONSTANT_COLS = [
    'Bwd PSH Flags', 'Bwd URG Flags',
    'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate'
]

# Standardized Label Normalization Mapping (Harmonizing Day 1 & Day 2)
LABEL_NORMALIZATION = {
    'DrDoS_DNS': 'DNS',
    'DrDoS_LDAP': 'LDAP',
    'DrDoS_MSSQL': 'MSSQL',
    'DrDoS_NetBIOS': 'NetBIOS',
    'DrDoS_NTP': 'NTP',
    'DrDoS_SNMP': 'SNMP',
    'DrDoS_SSDP': 'SSDP',
    'DrDoS_UDP': 'UDP',
    'UDP-lag': 'UDPLag',
    'UDP_lag': 'UDPLag',
    'udplag': 'UDPLag',
    'UDPLag': 'UDPLag',
    'SYN': 'Syn',
    'Syn': 'Syn',
    'TFTP': 'TFTP',
    'Portmap': 'Portmap',
    'NetBIOS': 'NetBIOS',
    'MSSQL': 'MSSQL',
    'LDAP': 'LDAP',
    'UDP': 'UDP',
    'DNS': 'DNS',
    'WebDDoS': 'WebDDoS',
    'BENIGN': 'BENIGN',
    'Benign': 'BENIGN',
    'benign': 'BENIGN'
}

# Global unified multi-class numeric code mapping
GLOBAL_LABEL_TO_CODE = {
    'BENIGN': 0,
    'DNS': 1,
    'LDAP': 2,
    'MSSQL': 3,
    'NTP': 4,
    'NetBIOS': 5,
    'Portmap': 6,
    'SNMP': 7,
    'SSDP': 8,
    'Syn': 9,
    'TFTP': 10,
    'UDP': 11,
    'UDPLag': 12,
    'WebDDoS': 13
}

# Lightweight flow features compatible with Ryu OpenFlow OFPFlowStatsReply
SDN_LIGHTWEIGHT_FEATURES = [
    'Protocol',
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Fwd Packets/s',
    'Bwd Packets/s',
    'Packet Length Mean',
    'Packet Length Std',
    'Average Packet Size',
    'Fwd Header Length',
    'Bwd Header Length',
    'Inbound'
]


def normalize_labels(label_series: pd.Series) -> pd.Series:
    """
    Normalizes dataset labels across Day 1 (01-12) and Day 2 (03-11).
    Removes 'DrDoS_' prefix and standardizes aliases (e.g., 'UDP-lag' -> 'UDPLag').
    """
    cleaned = label_series.astype(str).str.strip()
    
    def _map_single_label(val: str) -> str:
        if val.upper() == 'BENIGN':
            return 'BENIGN'
        if val in LABEL_NORMALIZATION:
            return LABEL_NORMALIZATION[val]
        if val.startswith('DrDoS_'):
            clean_name = val.replace('DrDoS_', '', 1)
            return LABEL_NORMALIZATION.get(clean_name, clean_name)
        return val
    
    return cleaned.apply(_map_single_label)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean, standardize, sample, and scale CICDDoS2019 dataset for SDN + DNN DDoS detection."
    )
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_output = os.path.join(base_dir, "cleaned_dataset")

    parser.add_argument("--input_dir", type=str, default=None, 
                        help="Directory containing raw CSV files.")
    parser.add_argument("--output_dir", type=str, default=default_output, 
                        help="Directory to save cleaned dataset and artifacts (scaler.pkl, mappings).")
    parser.add_argument("--output_prefix", type=str, default=None, 
                        help="Prefix for output CSV (e.g., cleaned_ddos_01_12).")
    
    # Feature set selection
    parser.add_argument("--feature_set", type=str, default="full", choices=["full", "sdn_lightweight"],
                        help="Feature selection mode: 'full' (all valid flow features) or 'sdn_lightweight' (Ryu SDN-compatible real-time flow features).")
    
    # Sampling & BENIGN traffic control
    parser.add_argument("--samples_per_attack", type=int, default=20000, 
                        help="Number of attack samples captured per CSV file.")
    parser.add_argument("--max_chunks_per_file", type=int, default=5, 
                        help="Max chunks (100k rows each) to scan per CSV file.")
    parser.add_argument("--scan_all_benign", action="store_true", default=True,
                        help="Continue scanning chunks up to max_chunks_per_file for BENIGN traffic even if attack quota is met.")
    parser.add_argument("--balance", action="store_true", 
                        help="Balance all classes to have equal number of samples.")
    parser.add_argument("--samples_per_class", type=int, default=None, 
                        help="Exact samples per class for balancing.")
    
    # Scaler & Schema Alignment
    parser.add_argument("--scaler_type", type=str, default="minmax", choices=["minmax", "standard", "none"],
                        help="Scaler type for DNN normalisation ('minmax', 'standard', or 'none').")
    parser.add_argument("--fit_scaler", action="store_true", default=None,
                        help="Fit and save a new scaler.pkl. Default: Auto (True for Day 1/Train, False for Day 2/Test).")
    parser.add_argument("--scaler_path", type=str, default=None,
                        help="Explicit path to load/save scaler.pkl.")
    parser.add_argument("--feature_columns_path", type=str, default=None,
                        help="Explicit path to feature_columns.json for strict schema alignment.")
    
    return parser.parse_args()


def extract_file_samples(file_path: str, samples_per_attack: int, max_chunks_per_file: int, scan_all_benign: bool = True):
    """
    Reads CSV in chunks, drops metadata, normalizes labels, and captures BENIGN and Attack samples.
    Continues scanning for sparse BENIGN traffic even after attack sample quota is met.
    """
    file_name = os.path.basename(file_path)
    print(f"[PROCESSING] {file_name}...", flush=True)
    
    file_samples = []
    benign_count = 0
    attack_count = 0

    chunk_iter = pd.read_csv(
        file_path,
        chunksize=100000,
        low_memory=False
    )

    for chunk_idx, chunk in enumerate(chunk_iter, start=1):
        chunk.columns = chunk.columns.str.strip()

        # Drop non-feature metadata columns
        cols_to_drop = [c for c in DROP_METADATA_COLS if c in chunk.columns]
        chunk = chunk.drop(columns=cols_to_drop)

        # Drop static constant columns if present
        static_drop = [c for c in STATIC_CONSTANT_COLS if c in chunk.columns]
        if static_drop:
            chunk = chunk.drop(columns=static_drop)

        if 'Label' not in chunk.columns:
            continue

        chunk['Label'] = normalize_labels(chunk['Label'])

        # Handle numeric columns Inf and NaNs
        numeric_cols = chunk.select_dtypes(include=[np.number]).columns
        chunk[numeric_cols] = chunk[numeric_cols].replace([np.inf, -np.inf], np.nan)
        chunk = chunk.dropna(subset=['Label'])

        benign_df = chunk[chunk['Label'] == 'BENIGN']
        attack_df = chunk[chunk['Label'] != 'BENIGN']

        # Capture BENIGN samples (accumulate all found in chunk)
        if len(benign_df) > 0:
            file_samples.append(benign_df)
            benign_count += len(benign_df)

        # Capture Attack samples up to target quota
        if attack_count < samples_per_attack and len(attack_df) > 0:
            needed = samples_per_attack - attack_count
            take_attack = attack_df.head(needed)
            file_samples.append(take_attack)
            attack_count += len(take_attack)

        # Stop condition:
        # If not scanning all benign, break when attack is full and some benign is found.
        # Otherwise, keep scanning chunks up to max_chunks_per_file to extract maximum BENIGN background traffic.
        if not scan_all_benign and attack_count >= samples_per_attack and benign_count > 0:
            break

        if chunk_idx >= max_chunks_per_file:
            break

    if file_samples:
        file_combined = pd.concat(file_samples, ignore_index=True)
        print(f"  Captured: {len(file_combined):,} rows ({benign_count:,} BENIGN, {attack_count:,} Attack)", flush=True)
        return file_combined
    return None


def align_and_clean_features(df: pd.DataFrame, feature_set: str, ref_feature_columns: list = None):
    """
    Cleans features, fills missing numeric values, filters by feature set (full vs sdn_lightweight),
    and strictly aligns columns to ref_feature_columns if provided.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isna().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0.0)

    # Filter by lightweight feature set if requested
    if feature_set == "sdn_lightweight":
        selected_features = [col for col in SDN_LIGHTWEIGHT_FEATURES if col in df.columns]
        missing_sdn_cols = set(SDN_LIGHTWEIGHT_FEATURES) - set(selected_features)
        if missing_sdn_cols:
            print(f"[WARN] Some SDN features were not found in dataset and skipped: {missing_sdn_cols}", flush=True)
        feature_cols = selected_features
    else:
        # Full features: drop any remaining dynamic zero-variance columns only if ref_feature_columns is None
        if ref_feature_columns is None:
            constant_cols = [col for col in num_cols if df[col].nunique() <= 1]
            if constant_cols:
                print(f"[INFO] Dropping {len(constant_cols)} constant columns: {constant_cols}", flush=True)
                df = df.drop(columns=constant_cols)
            feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['Label_Binary', 'Label_Multi']]
        else:
            feature_cols = ref_feature_columns

    # Align with reference schema (100% schema match between Day 1 & Day 2)
    if ref_feature_columns is not None:
        print(f"[INFO] Aligning features strictly to reference schema ({len(ref_feature_columns)} features)...", flush=True)
        # Drop columns not in ref
        for col in list(df.columns):
            if col not in ref_feature_columns and col != 'Label':
                df = df.drop(columns=[col])
        # Add missing columns with 0.0
        for col in ref_feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        feature_cols = ref_feature_columns

    return df, feature_cols


def scale_features(df: pd.DataFrame, feature_cols: list, scaler_type: str, fit_scaler: bool, scaler_path: str):
    """
    Fits or transforms numerical features using MinMaxScaler or StandardScaler.
    Saves or loads scaler.pkl to prevent data leakage between Train (Day 1) and Test (Day 2),
    and produces the exact scaler artifact for the Ryu Controller.
    """
    if scaler_type == "none":
        print("[INFO] Scaler type set to 'none'. Skipping scaling.", flush=True)
        return df, None

    scaler = None
    if fit_scaler:
        print(f"[INFO] Fitting new {scaler_type.upper()} scaler on {len(feature_cols)} features...", flush=True)
        scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
        scaled_values = scaler.fit_transform(df[feature_cols])
        df[feature_cols] = scaled_values

        os.makedirs(os.path.dirname(os.path.abspath(scaler_path)), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"[SUCCESS] Saved fitted scaler to: {scaler_path}", flush=True)
    else:
        if os.path.exists(scaler_path):
            print(f"[INFO] Loading existing scaler from: {scaler_path}...", flush=True)
            scaler = joblib.load(scaler_path)
            scaled_values = scaler.transform(df[feature_cols])
            df[feature_cols] = scaled_values
            print("[SUCCESS] Applied existing scaler transformation to dataset.", flush=True)
        else:
            print(f"[WARN] Scaler file '{scaler_path}' not found! Fitting a local scaler as fallback.", flush=True)
            scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
            scaled_values = scaler.fit_transform(df[feature_cols])
            df[feature_cols] = scaled_values

    return df, scaler


def encode_labels(df: pd.DataFrame):
    """
    Encodes standard binary and multi-class labels consistently across datasets.
    """
    print("[INFO] Encoding standardized labels...", flush=True)
    df['Label'] = normalize_labels(df['Label'])
    
    # Binary Label: 0 for BENIGN, 1 for Attack
    df['Label_Binary'] = (df['Label'] != 'BENIGN').astype(int)

    # Multi-class Label: map to unified global codes
    present_labels = sorted(df['Label'].unique())
    # Ensure any label present in dataset has a code
    current_mapping = {}
    for lbl in present_labels:
        if lbl in GLOBAL_LABEL_TO_CODE:
            current_mapping[lbl] = GLOBAL_LABEL_TO_CODE[lbl]
        else:
            current_mapping[lbl] = len(current_mapping)

    df['Label_Multi'] = df['Label'].map(current_mapping)
    return df, current_mapping


def process_single_folder(
    source_folder: str, 
    output_dir: str, 
    prefix: str, 
    samples_per_attack: int = 20000, 
    max_chunks_per_file: int = 5, 
    scan_all_benign: bool = True,
    balance: bool = False, 
    samples_per_class: int = None,
    feature_set: str = "full",
    scaler_type: str = "minmax",
    fit_scaler: bool = True,
    scaler_path: str = None,
    ref_feature_columns: list = None
):
    source_folder = os.path.abspath(source_folder)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, f"{prefix}.csv")
    mapping_json = os.path.join(output_dir, f"{prefix}_mapping.json")
    feature_cols_json = os.path.join(output_dir, f"{prefix}_features.json")
    global_feature_cols_json = os.path.join(output_dir, "feature_columns.json")
    global_mapping_json = os.path.join(output_dir, "label_mapping.json")

    if scaler_path is None:
        scaler_path = os.path.join(output_dir, "scaler.pkl")

    csv_files = [f for f in glob.glob(os.path.join(source_folder, "*.csv")) if not os.path.basename(f).startswith(".~")]
    if not csv_files:
        print(f"[ERROR] No valid CSV files found in '{source_folder}'!", flush=True)
        return None, None

    print(f"\n==========================================", flush=True)
    print(f"[INFO] Processing Folder: '{source_folder}'", flush=True)
    print(f"[INFO] Found {len(csv_files)} CSV files.", flush=True)
    print(f"[INFO] Feature Set: '{feature_set}' | Scaler: '{scaler_type}' (fit={fit_scaler})", flush=True)
    print(f"[INFO] Target Output: '{output_csv}'", flush=True)
    print(f"==========================================", flush=True)

    sampled_chunks = []
    for file_path in csv_files:
        file_df = extract_file_samples(
            file_path, 
            samples_per_attack=samples_per_attack, 
            max_chunks_per_file=max_chunks_per_file,
            scan_all_benign=scan_all_benign
        )
        if file_df is not None:
            sampled_chunks.append(file_df)

    if not sampled_chunks:
        print(f"[ERROR] No data extracted from '{source_folder}'!", flush=True)
        return None, None

    print(f"\n[INFO] Combining extracted chunks for '{prefix}'...", flush=True)
    df = pd.concat(sampled_chunks, ignore_index=True)
    print(f"[INFO] Combined raw shape: {df.shape}", flush=True)

    # Feature Cleaning, Selection & Alignment
    df, feature_cols = align_and_clean_features(
        df, 
        feature_set=feature_set, 
        ref_feature_columns=ref_feature_columns
    )

    # Class Balancing if requested
    if balance or samples_per_class is not None:
        min_count = df['Label'].value_counts().min()
        target_count = samples_per_class if samples_per_class is not None else min_count
        print(f"[INFO] Balancing classes to {target_count:,} samples per class...", flush=True)
        balanced_chunks = []
        for label_name, group_df in df.groupby('Label'):
            n_take = min(len(group_df), target_count)
            balanced_chunks.append(group_df.sample(n=n_take, random_state=42))
        df = pd.concat(balanced_chunks, ignore_index=True)

    # Label Normalization and Encoding
    df, label_to_code = encode_labels(df)

    # Feature Scaling & Scaler Artifact generation
    df, scaler = scale_features(
        df, 
        feature_cols=feature_cols, 
        scaler_type=scaler_type, 
        fit_scaler=fit_scaler, 
        scaler_path=scaler_path
    )

    # Reorder columns: [feature_cols, 'Label', 'Label_Binary', 'Label_Multi']
    final_cols = feature_cols + ['Label', 'Label_Binary', 'Label_Multi']
    df = df[final_cols]

    # Save mapping and feature schema JSONs
    with open(mapping_json, 'w') as f:
        json.dump(label_to_code, f, indent=2)
    with open(global_mapping_json, 'w') as f:
        json.dump(GLOBAL_LABEL_TO_CODE, f, indent=2)

    with open(feature_cols_json, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    if fit_scaler or not os.path.exists(global_feature_cols_json):
        with open(global_feature_cols_json, 'w') as f:
            json.dump(feature_cols, f, indent=2)

    print(f"[INFO] Saving cleaned dataset to: {output_csv}...", flush=True)
    df.to_csv(output_csv, index=False)
    print(f"[SUCCESS] Saved '{prefix}' with shape {df.shape} ({len(feature_cols)} features)", flush=True)
    print("=== CLASS DISTRIBUTION ===", flush=True)
    print(df['Label'].value_counts(), flush=True)

    return df, feature_cols


def main():
    args = parse_args()
    
    # Custom input directory specified
    if args.input_dir:
        prefix = args.output_prefix if args.output_prefix else "cleaned_ddos_sample"
        fit_scaler = True if args.fit_scaler is None else args.fit_scaler
        scaler_path = args.scaler_path if args.scaler_path else os.path.join(args.output_dir, "scaler.pkl")
        
        ref_features = None
        if args.feature_columns_path and os.path.exists(args.feature_columns_path):
            with open(args.feature_columns_path, 'r') as f:
                ref_features = json.load(f)

        process_single_folder(
            args.input_dir,
            args.output_dir,
            prefix=prefix,
            samples_per_attack=args.samples_per_attack,
            max_chunks_per_file=args.max_chunks_per_file,
            scan_all_benign=args.scan_all_benign,
            balance=args.balance,
            samples_per_class=args.samples_per_class,
            feature_set=args.feature_set,
            scaler_type=args.scaler_type,
            fit_scaler=fit_scaler,
            scaler_path=scaler_path,
            ref_feature_columns=ref_features
        )
    else:
        # Default Pipeline: Process Day 1 (Train) -> Fit Scaler -> Process Day 2 (Test) -> Apply Scaler
        base_dir = os.path.dirname(os.path.abspath(__file__))

        candidates_01_12 = [
            os.path.join(base_dir, "raw_dataset", "CSV-01-12", "01-12"),
            os.path.join(base_dir, "raw_dataset", "CSV-01-12"),
            os.path.join(base_dir, "raw_data", "CSV-01-12", "01-12"),
            os.path.join(base_dir, "raw_data", "CSV-01-12")
        ]
        folder_01_12 = next((p for p in candidates_01_12 if os.path.exists(p)), candidates_01_12[0])

        candidates_03_11 = [
            os.path.join(base_dir, "raw_dataset", "CSV-03-11", "03-11"),
            os.path.join(base_dir, "raw_dataset", "CSV-03-11"),
            os.path.join(base_dir, "raw_data", "CSV-03-11", "03-11"),
            os.path.join(base_dir, "raw_data", "CSV-03-11")
        ]
        folder_03_11 = next((p for p in candidates_03_11 if os.path.exists(p)), candidates_03_11[0])

        day1_feature_cols = None
        scaler_path = args.scaler_path if args.scaler_path else os.path.join(args.output_dir, "scaler.pkl")

        # 1. Process Day 1 (01-12) - Training Set (Fit Scaler & Define Schema)
        if os.path.exists(folder_01_12):
            print("\n>>> [PIPELINE] STEP 1: Processing Day 1 (01-12) Training Set...", flush=True)
            df_day1, day1_feature_cols = process_single_folder(
                folder_01_12,
                args.output_dir,
                prefix="cleaned_ddos_01_12",
                samples_per_attack=args.samples_per_attack,
                max_chunks_per_file=args.max_chunks_per_file,
                scan_all_benign=args.scan_all_benign,
                balance=args.balance,
                samples_per_class=args.samples_per_class,
                feature_set=args.feature_set,
                scaler_type=args.scaler_type,
                fit_scaler=True if args.fit_scaler is None else args.fit_scaler,
                scaler_path=scaler_path,
                ref_feature_columns=None
            )

        # 2. Process Day 2 (03-11) - Test Set (Apply Day 1 Scaler & Align Day 1 Schema)
        if os.path.exists(folder_03_11):
            print("\n>>> [PIPELINE] STEP 2: Processing Day 2 (03-11) Test Set...", flush=True)
            process_single_folder(
                folder_03_11,
                args.output_dir,
                prefix="cleaned_ddos_03_11",
                samples_per_attack=args.samples_per_attack,
                max_chunks_per_file=args.max_chunks_per_file,
                scan_all_benign=args.scan_all_benign,
                balance=args.balance,
                samples_per_class=args.samples_per_class,
                feature_set=args.feature_set,
                scaler_type=args.scaler_type,
                fit_scaler=False,  # Prevent Data Leakage: Use Day 1 fitted scaler!
                scaler_path=scaler_path,
                ref_feature_columns=day1_feature_cols  # Enforce 100% column schema match
            )


if __name__ == '__main__':
    main()
