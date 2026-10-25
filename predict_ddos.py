
import json
import joblib
import numpy as np
import pandas as pd
import torch

from train_dnn import DDoSDeepNeuralNetwork

MODEL_PATH = "models/dnn_ddos_model.pt"
SCALER_PATH = "cleaned_dataset/scaler.pkl"
FEATURE_PATH = "cleaned_dataset/feature_columns.json"


def load_inference_components():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    with open(FEATURE_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    scaler = joblib.load(SCALER_PATH)

    model = DDoSDeepNeuralNetwork(
        input_dim=checkpoint["input_dim"],
        hidden_dims=checkpoint["hidden_dims"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, scaler, feature_columns


def predict_ddos(raw_features, model, scaler, feature_columns):
    # ตรวจสอบว่าได้รับครบทั้ง 67 Features
    missing = set(feature_columns) - set(raw_features)

    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    # จัดลำดับ Features ให้ตรงกับตอน Train
    data = pd.DataFrame(
        [{name: raw_features[name] for name in feature_columns}]
    )

    # ป้องกันข้อมูลที่ไม่ใช่ตัวเลขและค่า Inf/NaN
    data = data.apply(pd.to_numeric, errors="raise")

    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Input contains NaN or Infinity")

    # ใช้ Scaler เดิมที่ Fit จาก Training Data
    scaled = scaler.transform(data)

    # แปลงเป็น Tensor
    x = torch.tensor(scaled, dtype=torch.float32)

    # ทำนายโดยไม่คำนวณ Gradient
    with torch.inference_mode():
        logits = model(x)
        probability = torch.sigmoid(logits).item()

    prediction = int(probability >= 0.5)

    return {
        "prediction": prediction,
        "label": "DDoS" if prediction == 1 else "BENIGN",
        "ddos_probability": probability
    }


if __name__ == "__main__":
    model, scaler, feature_columns = load_inference_components()

    print("DNN model loaded successfully!")
    print("Input features:", len(feature_columns))
    print("Model is ready for inference.")
