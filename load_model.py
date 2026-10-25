"""
load_model.py
=============
Script สำหรับโหลดและใช้งานโมเดล DDoS Detection DNN ที่ train ไว้แล้ว

ไฟล์โมเดลในโฟลเดอร์ models/:
  - best_dnn_weights.pt  : เก็บเฉพาะ state_dict (weights) ของโมเดลที่ดีที่สุด
  - dnn_ddos_model.pt    : Deployment package (มี architecture + weights + feature columns)

วิธีใช้งาน:
  python load_model.py                        # รันตัวอย่างทดสอบ
  python load_model.py --mode weights         # โหลดจาก best_dnn_weights.pt
  python load_model.py --mode full            # โหลดจาก dnn_ddos_model.pt (แนะนำ)
  python load_model.py --csv path/to/data.csv # ทำนายจากไฟล์ CSV
"""

import os
import sys
import json
import argparse
import numpy as np

# Configure UTF-8 stdout for Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import torch.nn as nn

# ──────────────────────────────────────────────
# 1. นิยาม Model Architecture (เหมือนกับใน train_dnn.py)
# ──────────────────────────────────────────────

class DDoSDeepNeuralNetwork(nn.Module):
    """
    DNN สำหรับตรวจจับ DDoS Attack บน SDN
    Architecture: Linear -> BatchNorm -> ReLU -> Dropout (x N layers) -> Linear
    """
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64],
                 output_dim: int = 1, dropout_rate: float = 0.3):
        super(DDoSDeepNeuralNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x):
        return self.network(x)


# ──────────────────────────────────────────────
# 2. ฟังก์ชันโหลดโมเดล
# ──────────────────────────────────────────────

def load_full_model(model_path: str, device: torch.device = None):
    """
    โหลด Deployment Package จาก dnn_ddos_model.pt
    (ไม่ต้องกำหนด architecture เอง เพราะ metadata อยู่ในไฟล์แล้ว)

    Returns:
        model          : โมเดลที่พร้อมใช้งาน (eval mode)
        feature_cols   : รายชื่อ feature ที่โมเดลต้องการ (list)
        metadata       : dict ที่มีข้อมูลอื่นๆ เช่น hidden_dims, dropout
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Loading deployment package from: '{model_path}'")
    checkpoint = torch.load(model_path, map_location=device)

    input_dim    = checkpoint['input_dim']
    hidden_dims  = checkpoint['hidden_dims']
    dropout      = checkpoint['dropout']
    feature_cols = checkpoint['feature_columns']

    model = DDoSDeepNeuralNetwork(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=1,
        dropout_rate=dropout
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    metadata = {
        'input_dim'   : input_dim,
        'hidden_dims' : hidden_dims,
        'dropout'     : dropout,
        'architecture': checkpoint.get('architecture', 'DDoSDeepNeuralNetwork'),
    }

    print(f"[OK]   Model loaded successfully!")
    print(f"       Architecture : {hidden_dims}  (input_dim={input_dim})")
    print(f"       Feature count: {len(feature_cols)}")
    print(f"       Device       : {device}")
    return model, feature_cols, metadata


def load_weights_only(weights_path: str, feature_cols_path: str,
                      hidden_dims: list = [256, 128, 64], dropout: float = 0.3,
                      device: torch.device = None):
    """
    โหลด state_dict จาก best_dnn_weights.pt
    ต้องระบุ feature_cols_path และ architecture เอง

    Returns:
        model        : โมเดลที่พร้อมใช้งาน (eval mode)
        feature_cols : รายชื่อ feature
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Loading feature columns from: '{feature_cols_path}'")
    with open(feature_cols_path, 'r') as f:
        feature_cols = json.load(f)

    input_dim = len(feature_cols)
    model = DDoSDeepNeuralNetwork(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=1,
        dropout_rate=dropout
    ).to(device)

    print(f"[INFO] Loading weights from: '{weights_path}'")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"[OK]   Weights loaded successfully!")
    print(f"       Architecture : {hidden_dims}  (input_dim={input_dim})")
    print(f"       Device       : {device}")
    return model, feature_cols


# ──────────────────────────────────────────────
# 3. ฟังก์ชันทำนาย
# ──────────────────────────────────────────────

def predict(model: nn.Module, X: np.ndarray, device: torch.device,
            threshold: float = 0.5, batch_size: int = 1024):
    """
    ทำนายผลจาก numpy array

    Args:
        model     : โมเดลที่โหลดแล้ว
        X         : numpy array รูปร่าง (n_samples, n_features)
        device    : torch device
        threshold : ค่า threshold สำหรับตัดสินใจ (default=0.5)
        batch_size: ขนาด batch สำหรับ inference

    Returns:
        preds     : numpy array of int (0=BENIGN, 1=DDoS Attack)
        probs     : numpy array of float (ความน่าจะเป็นของ class 1)
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            logits = model(batch)
            probs  = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)

    all_probs = np.array(all_probs)
    preds     = (all_probs >= threshold).astype(int)
    return preds, all_probs


def predict_from_csv(model: nn.Module, feature_cols: list, csv_path: str,
                     device: torch.device, threshold: float = 0.5):
    """
    ทำนายผลจากไฟล์ CSV และแสดงผลสรุป

    Args:
        model        : โมเดลที่โหลดแล้ว
        feature_cols : รายชื่อ feature columns
        csv_path     : path ของไฟล์ CSV
        device       : torch device
        threshold    : ค่า threshold
    """
    import pandas as pd

    print(f"\n[DATA] Loading CSV: '{csv_path}'")
    df = pd.read_csv(csv_path)
    print(f"       Shape: {df.shape}")

    # เติม 0 ให้กับ features ที่หายไป
    for col in feature_cols:
        if col not in df.columns:
            print(f"       [WARN] Column '{col}' not found, filling with 0")
            df[col] = 0.0

    X = df[feature_cols].values.astype(np.float32)
    preds, probs = predict(model, X, device, threshold)

    # สรุปผล
    n_attack = int(preds.sum())
    n_benign = int((preds == 0).sum())
    print(f"\n{'='*50}")
    print(f"  PREDICTION RESULTS")
    print(f"{'='*50}")
    print(f"  Total samples  : {len(preds):,}")
    print(f"  BENIGN (0)     : {n_benign:,}  ({n_benign/len(preds)*100:.2f}%)")
    print(f"  DDoS Attack (1): {n_attack:,}  ({n_attack/len(preds)*100:.2f}%)")
    print(f"  Threshold used : {threshold}")
    print(f"{'='*50}")

    # เพิ่มคอลัมน์ผลทำนายกลับเข้า DataFrame
    df['predicted_label'] = preds
    df['attack_probability'] = probs

    # แสดง ground truth ถ้ามี
    if 'Label_Binary' in df.columns:
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        y_true = df['Label_Binary'].values
        acc = accuracy_score(y_true, preds)
        f1  = f1_score(y_true, preds, zero_division=0)
        print(f"\n  [Evaluation vs Ground Truth]")
        print(f"  Accuracy : {acc*100:.4f}%")
        print(f"  F1-Score : {f1:.4f}")
        print(f"\n{classification_report(y_true, preds, target_names=['BENIGN','DDoS Attack'])}")

    return df


# ──────────────────────────────────────────────
# 4. Demo / Main
# ──────────────────────────────────────────────

def main():
    base_dir     = os.path.dirname(os.path.abspath(__file__))
    models_dir   = os.path.join(base_dir, "models")
    cleaned_dir  = os.path.join(base_dir, "cleaned_dataset")

    full_model_path    = os.path.join(models_dir, "dnn_ddos_model.pt")
    best_weights_path  = os.path.join(models_dir, "best_dnn_weights.pt")
    feature_cols_path  = os.path.join(cleaned_dir, "feature_columns.json")

    parser = argparse.ArgumentParser(description="Load and use trained DDoS DNN model")
    parser.add_argument("--mode", choices=["full", "weights"], default="full",
                        help="'full' = โหลดจาก dnn_ddos_model.pt (แนะนำ) | 'weights' = โหลดจาก best_dnn_weights.pt")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path ของไฟล์ CSV ที่ต้องการทำนาย (ถ้าไม่ระบุจะรัน demo ด้วยข้อมูลสุ่ม)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold สำหรับตัดสินใจ DDoS vs BENIGN (default=0.5)")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="บันทึกผลทำนายเป็นไฟล์ CSV (optional)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  DDoS Detection DNN - Model Loader")
    print(f"{'='*60}")
    print(f"  Device : {device}")
    print(f"  Mode   : {args.mode}")
    print()

    # ── โหลดโมเดล ──
    if args.mode == "full":
        if not os.path.exists(full_model_path):
            print(f"[ERROR] ไม่พบไฟล์: '{full_model_path}'")
            sys.exit(1)
        model, feature_cols, metadata = load_full_model(full_model_path, device)

    else:  # weights mode
        for path, label in [(best_weights_path, "best_dnn_weights.pt"),
                            (feature_cols_path, "feature_columns.json")]:
            if not os.path.exists(path):
                print(f"[ERROR] ไม่พบไฟล์: '{path}'")
                sys.exit(1)
        model, feature_cols = load_weights_only(
            weights_path=best_weights_path,
            feature_cols_path=feature_cols_path,
            device=device
        )

    # ── แสดง Model Architecture ──
    print(f"\n{'='*60}")
    print("  MODEL ARCHITECTURE")
    print(f"{'='*60}")
    print(model)

    # ── ทำนายผล ──
    if args.csv:
        # ทำนายจากไฟล์ CSV ที่ระบุ
        df_result = predict_from_csv(model, feature_cols, args.csv, device, args.threshold)
        if args.output_csv:
            df_result.to_csv(args.output_csv, index=False)
            print(f"\n[SAVED] Results saved to: '{args.output_csv}'")
    else:
        # Demo: สร้างข้อมูลสุ่มเพื่อทดสอบ
        print(f"\n{'='*60}")
        print("  DEMO: Predict on random synthetic data")
        print(f"{'='*60}")
        np.random.seed(42)
        n_demo    = 10
        X_demo    = np.random.randn(n_demo, len(feature_cols)).astype(np.float32)
        preds, probs = predict(model, X_demo, device, args.threshold)

        print(f"  Input shape : {X_demo.shape}")
        print(f"  Threshold   : {args.threshold}")
        print(f"\n  {'Sample':>8} | {'Prob (Attack)':>15} | {'Prediction':>12}")
        print(f"  {'-'*8} | {'-'*15} | {'-'*12}")
        for i, (prob, pred) in enumerate(zip(probs, preds)):
            label = "DDoS Attack" if pred == 1 else "BENIGN"
            print(f"  {i+1:>8} | {prob:>15.6f} | {label:>12}")

    print(f"\n{'='*60}")
    print("  Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
