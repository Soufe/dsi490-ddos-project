# DDoS Attack Detection Project (CICDDoS2019)

Machine Learning and Data Science project for DDoS Attack Detection using the **CICDDoS2019** dataset (`CSV-01-12` and `CSV-03-11`).

---

## 📁 Repository Structure

```text
ddos project/
├── cleaned_dataset/
│   ├── cleaned_ddos_01_12.csv         # Cleaned dataset Day 1
│   ├── cleaned_ddos_01_12_mapping.json
│   ├── cleaned_ddos_03_11.csv         # Cleaned dataset Day 2
│   ├── cleaned_ddos_03_11_mapping.json
│   ├── cleaned_ddos_sample.csv        # Default preprocessed sample dataset (~240k rows, 70 features)
│   └── label_mapping.json             # Default multi-class label mapping dictionary
├── raw_dataset/                       # (Ignored by git) Raw CSV dataset folders
│   ├── CSV-01-12/
│   └── CSV-03-11/
├── data_prep.py                       # CLI pipeline script for data cleaning & sampling
├── eda.ipynb                          # Jupyter Notebook for Exploratory Data Analysis & EDA
├── requirements.txt                   # Required Python packages
└── .gitignore                         # Git ignore file for large raw data & checkpoints
```

---

## 🚀 Quick Start for Collaborators

### 1. Clone & Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run Data Preparation (Optional)

By default, the repository includes the preprocessed sample dataset `cleaned_dataset/cleaned_ddos_sample.csv`. 

If you have the raw `CSV-01-12` or `CSV-03-11` folders inside `raw_dataset/` or locally, you can re-run the preparation script:

```bash
# Run with default input directory (checks raw_dataset/ and raw_data/)
python data_prep.py

# Or specify custom input & output directories
python data_prep.py --input_dir /path/to/CSV-01-12/01-12 --output_dir cleaned_dataset/
```

#### Command-line Options:
- `--input_dir`: Path to folder containing raw `.csv` files.
- `--output_dir`: Path to save output cleaned dataset & artifacts (default: `./cleaned_dataset`).
- `--feature_set`: Choose between `full` (70+ flow features) or `sdn_lightweight` (15 real-time Ryu SDN OpenFlow stats compatible features).
- `--samples_per_attack`: Max attack samples captured per file (default: `20000`).
- `--max_chunks_per_file`: Max chunks (100k rows each) to scan per CSV file (default: `5`).
- `--scan_all_benign`: Keep scanning chunks up to limit for BENIGN traffic to enrich background normal flows (default: `True`).
- `--scaler_type`: Normalizer for DNN (`minmax`, `standard`, `none`). Fits on Day 1, saves `scaler.pkl`, and transforms Day 2 without data leakage.
- `--balance`: Balance all classes to equal sample count.

### 3. Exploratory Data Analysis (EDA)

Launch Jupyter Notebook and open `eda.ipynb`:

```bash
jupyter notebook eda.ipynb
```

---

## 🏷️ Label Encoding & SDN Artifacts

- **`Label_Binary`**:
  - `0`: BENIGN (Normal Traffic)
  - `1`: DDoS Attack
- **`Label_Multi`**: Multi-class standardized indices (`0` for `BENIGN`, `1`..`13` for normalized attack names like `DNS`, `LDAP`, `MSSQL`, `NTP`, `NetBIOS`, `Portmap`, `SNMP`, `SSDP`, `Syn`, `TFTP`, `UDP`, `UDPLag`, `WebDDoS`).
- **Artifacts for Ryu SDN Controller**:
  - `scaler.pkl`: Serialized scaler model fitted on training set.
  - `feature_columns.json`: Exact ordered list of features expected by the trained DNN model.
  - `label_mapping.json`: Mapping dictionary between string attack classes and integer codes.
