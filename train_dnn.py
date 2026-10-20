import os
import sys
import time
import json
import argparse

# Configure UTF-8 stdout for Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class DDoSDeepNeuralNetwork(nn.Module):
    """
    3 to 5 layer Deep Neural Network (DNN) with Batch Normalization,
    Dropout Regularization, and LeakyReLU/ReLU activations for DDoS Attack Detection on SDN.
    """
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64], output_dim: int = 1, dropout_rate: float = 0.3):
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train Deep Neural Network (DNN) for SDN DDoS Detection.")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument("--train_data", type=str, 
                        default=os.path.join(base_dir, "cleaned_dataset", "cleaned_ddos_01_12.csv"),
                        help="Path to Day 1 (01-12) cleaned training CSV.")
    parser.add_argument("--test_data", type=str, 
                        default=os.path.join(base_dir, "cleaned_dataset", "cleaned_ddos_03_11.csv"),
                        help="Path to Day 2 (03-11) cleaned test CSV.")
    parser.add_argument("--features_path", type=str, 
                        default=os.path.join(base_dir, "cleaned_dataset", "feature_columns.json"),
                        help="Path to ordered feature columns JSON.")
    parser.add_argument("--output_dir", type=str, 
                        default=os.path.join(base_dir, "models"),
                        help="Directory to save trained model artifacts and plots.")
    parser.add_argument("--epochs", type=int, default=15, 
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=256, 
                        help="Mini-batch size for training.")
    parser.add_argument("--lr", type=float, default=0.001, 
                        help="Learning rate for Adam optimizer.")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128, 64],
                        help="Hidden layer dimensions (e.g. 256 128 64).")
    parser.add_argument("--dropout", type=float, default=0.3, 
                        help="Dropout probability.")
    parser.add_argument("--val_split", type=float, default=0.2, 
                        help="Validation split ratio on Day 1.")
    
    return parser.parse_args()


def load_dataset(csv_path: str, feature_cols: list, label_col: str = 'Label_Binary'):
    """
    Loads dataset, ensures correct feature ordering, and returns X and y.
    """
    print(f"[DATA] Loading '{csv_path}'...", flush=True)
    df = pd.read_csv(csv_path)
    
    # Missing columns check
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.float32)
    raw_labels = df['Label'].values if 'Label' in df.columns else None
    
    print(f"  Shape: {X.shape} | Attack ratio: {(y == 1).mean() * 100:.2f}%", flush=True)
    return X, y, raw_labels


def train_model(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    scheduler, 
    device: torch.device, 
    epochs: int,
    model_save_path: str
):
    print("\n" + "=" * 65, flush=True)
    print(">>> STARTING DNN MODEL TRAINING (PyTorch) <<<", flush=True)
    print("=" * 65, flush=True)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_f1': []
    }
    
    best_val_f1 = 0.0
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_X.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_y.size(0)
            
        train_loss = running_loss / total_train
        train_acc = correct_train / total_train
        
        # --- VALIDATION PHASE ---
        model.eval()
        running_val_loss = 0.0
        val_preds_list = []
        val_targets_list = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device).unsqueeze(1)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                running_val_loss += loss.item() * batch_X.size(0)
                
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                
                val_preds_list.extend(preds.cpu().numpy().flatten())
                val_targets_list.extend(batch_y.cpu().numpy().flatten())
                
        val_loss = running_val_loss / len(val_targets_list)
        val_acc = accuracy_score(val_targets_list, val_preds_list)
        val_f1 = f1_score(val_targets_list, val_preds_list, zero_division=0)
        
        if scheduler:
            scheduler.step(val_loss)
            
        epoch_dur = time.time() - epoch_start
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Checkpoint best model
        saved_mark = ""
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            saved_mark = " [BEST SAVED]"
            
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_dur:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc * 100:.2f}% | Val F1: {val_f1:.4f}{saved_mark}", flush=True)
              
    total_time = time.time() - start_time
    print(f"\n[SUCCESS] Training finished in {total_time:.1f}s. Best Validation F1: {best_val_f1:.4f}")
    return history


def evaluate_on_test_day2(model: nn.Module, X_test, y_test, raw_labels_test, device: torch.device):
    """
    Comprehensive evaluation on unseen Day 2 test set (Cross-day generalization & False Positive Rate).
    """
    print("\n" + "=" * 65, flush=True)
    print(">>> EVALUATING ON INDEPENDENT DAY 2 (03-11) TEST SET <<<", flush=True)
    print("=" * 65, flush=True)
    
    model.eval()
    test_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_loader = DataLoader(TensorDataset(test_tensor), batch_size=1024, shuffle=False)
    
    all_probs = []
    inference_times = []
    
    with torch.no_grad():
        for batch in test_loader:
            b_x = batch[0].to(device)
            t0 = time.time()
            outputs = model(b_x)
            t1 = time.time()
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_probs.extend(probs)
            inference_times.append((t1 - t0) / len(b_x))
            
    all_probs = np.array(all_probs)
    preds = (all_probs >= 0.5).astype(int)
    
    # Calculate detection metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, all_probs)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    avg_latency_ms = np.mean(inference_times) * 1000.0
    throughput_fps = 1.0 / np.mean(inference_times) if np.mean(inference_times) > 0 else 0.0
    
    results = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'false_positive_rate': float(fpr),
        'confusion_matrix': {
            'TN': int(tn), 'FP': int(fp),
            'FN': int(fn), 'TP': int(tp)
        },
        'avg_inference_latency_ms': float(avg_latency_ms),
        'throughput_flows_per_sec': float(throughput_fps),
        'total_test_samples': int(len(y_test)),
        'benign_test_samples': int(tn + fp),
        'attack_test_samples': int(fn + tp)
    }
    
    print(f"Test Accuracy       : {acc * 100:.2f}%")
    print(f"Precision           : {prec * 100:.2f}%")
    print(f"Recall (Sensitivity): {rec * 100:.2f}%")
    print(f"F1-Score            : {f1:.4f}")
    print(f"ROC-AUC Score       : {roc_auc:.4f}")
    print(f"False Positive Rate : {fpr * 100:.4f}% ({fp:,} / {tn + fp:,} Benign flows falsely flagged)")
    print(f"Avg Inference Time  : {avg_latency_ms:.4f} ms/flow (~{throughput_fps:,.0f} flows/sec)")
    print("\nConfusion Matrix:")
    print(f"  TN (Benign -> Benign)   : {tn:,}")
    print(f"  FP (Benign -> Attack)   : {fp:,}  (Crucial for SDN False Alarm Mitigation)")
    print(f"  FN (Attack -> Benign)   : {fn:,}")
    print(f"  TP (Attack -> Attack)   : {tp:,}")
    
    # Per attack type recall analysis
    if raw_labels_test is not None:
        print("\n=== PER-CLASS DETECTION ACCURACY ON DAY 2 ===")
        df_eval = pd.DataFrame({'Label': raw_labels_test, 'Actual': y_test, 'Pred': preds})
        per_class = []
        for lbl, group in df_eval.groupby('Label'):
            group_acc = float((group['Actual'] == group['Pred']).mean())
            count = int(len(group))
            correct = int((group['Actual'] == group['Pred']).sum())
            per_class.append({'Class': str(lbl), 'Samples': count, 'Correct': correct, 'Accuracy': group_acc})
            print(f"  - {lbl:<12}: {group_acc * 100:6.2f}% ({correct:,}/{count:,})")
        results['per_class_breakdown'] = per_class
        
    return results, cm, all_probs


def plot_and_save_charts(history: dict, results: dict, cm: np.ndarray, y_test, all_probs, output_dir: str):
    """
    Generates high-aesthetic visual evaluation charts for thesis and presentation slides.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), dpi=120)
    plt.subplots_adjust(wspace=0.3)
    
    # 1. Loss & F1 Curve
    epochs_range = range(1, len(history['train_loss']) + 1)
    ax1 = axes[0]
    ax1.plot(epochs_range, history['train_loss'], label='Train Loss', color='#2b5c8f', lw=2)
    ax1.plot(epochs_range, history['val_loss'], label='Val Loss', color='#d96b27', lw=2, linestyle='--')
    ax1.plot(epochs_range, history['val_f1'], label='Val F1', color='#107c41', lw=2)
    ax1.set_title("Training & Validation Curves", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("Epochs", fontsize=11)
    ax1.set_ylabel("Loss / Score", fontsize=11)
    ax1.legend(loc='center right', frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 2. Confusion Matrix Heatmap
    ax2 = axes[1]
    labels_text = ['BENIGN (0)', 'DDoS Attack (1)']
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', cbar=False, ax=ax2,
                xticklabels=labels_text, yticklabels=labels_text, annot_kws={'size': 13, 'weight': 'bold'})
    ax2.set_title(f"Test Confusion Matrix (Day 2)\nFPR: {results['false_positive_rate']*100:.3f}% | Acc: {results['accuracy']*100:.2f}%", 
                  fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel("Predicted Label", fontsize=11)
    ax2.set_ylabel("Ground Truth Label", fontsize=11)
    
    # 3. ROC Curve
    ax3 = axes[2]
    fpr_curve, tpr_curve, _ = roc_curve(y_test, all_probs)
    ax3.plot(fpr_curve, tpr_curve, color='#7c3aed', lw=2.5, label=f"DNN (AUC = {results['roc_auc']:.4f})")
    ax3.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', lw=1.5)
    ax3.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel("False Positive Rate", fontsize=11)
    ax3.set_ylabel("True Positive Rate", fontsize=11)
    ax3.legend(loc='lower right', frameon=True)
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    chart_path = os.path.join(output_dir, "evaluation_summary.png")
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()
    print(f"\n[SAVED] Visual charts saved to: '{chart_path}'", flush=True)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load feature columns
    if not os.path.exists(args.features_path):
        raise FileNotFoundError(f"Feature columns schema not found at: {args.features_path}")
        
    with open(args.features_path, 'r') as f:
        feature_cols = json.load(f)
    print(f"[INFO] Using {len(feature_cols)} standard features for DNN input.", flush=True)
    
    # Load Day 1 (Train + Val)
    X_day1, y_day1, _ = load_dataset(args.train_data, feature_cols)
    
    # Stratified Train/Val Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_day1, y_day1, test_size=args.val_split, random_state=SEED, stratify=y_day1
    )
    print(f"[SPLIT] Train: {len(X_train):,} samples | Val: {len(X_val):,} samples", flush=True)
    
    # PyTorch DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Build DNN Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Training on device: {device}", flush=True)
    
    model = DDoSDeepNeuralNetwork(
        input_dim=len(feature_cols),
        hidden_dims=args.hidden_dims,
        output_dim=1,
        dropout_rate=args.dropout
    ).to(device)
    
    print("\n=== DNN MODEL ARCHITECTURE ===")
    print(model)
    
    # Loss function with positive class weighting for high precision
    num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_weights_path = os.path.join(args.output_dir, "best_dnn_weights.pt")
    full_model_path = os.path.join(args.output_dir, "dnn_ddos_model.pt")
    
    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        model_save_path=best_weights_path
    )
    
    # Load best weights
    model.load_state_dict(torch.load(best_weights_path))
    
    # Save standalone deployment package for Ryu Controller
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': len(feature_cols),
        'hidden_dims': args.hidden_dims,
        'feature_columns': feature_cols,
        'dropout': args.dropout,
        'architecture': 'DDoSDeepNeuralNetwork'
    }, full_model_path)
    print(f"[SAVED] Deployment package saved to: '{full_model_path}'", flush=True)
    
    # Evaluate on Unseen Day 2 Test Set
    if os.path.exists(args.test_data):
        X_test, y_test, raw_labels_test = load_dataset(args.test_data, feature_cols)
        results, cm, all_probs = evaluate_on_test_day2(model, X_test, y_test, raw_labels_test, device)
        
        # Save results JSON
        results_json_path = os.path.join(args.output_dir, "evaluation_report.json")
        with open(results_json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[SAVED] Evaluation metrics saved to: '{results_json_path}'", flush=True)
        
        # Plot evaluation summary charts
        plot_and_save_charts(history, results, cm, y_test, all_probs, args.output_dir)
        
    print("\n" + "=" * 65)
    print(">>> STEP 3: DNN MODEL TRAINING & EVALUATION COMPLETED! <<<")
    print("=" * 65)


if __name__ == '__main__':
    main()
