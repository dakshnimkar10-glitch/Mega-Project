"""
CALCE Battery Prediction Script
===============================
Loads trained CALCE battery models (SOH, SOC, Anomaly Detector) and runs sample predictions.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def load_soh_model():
    path = BASE_DIR / "calce_soh_model.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["feature_cols"]


def load_soc_model():
    path = BASE_DIR / "calce_soc_model.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["feature_cols"]


def load_anomaly_detector():
    path = BASE_DIR / "calce_anomaly_detector.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["iso_forest"], data["feature_cols"], data.get("rules", {})


def main():
    print("=" * 60)
    print(" CALCE Battery Models - Inference Demo")
    print("=" * 60)

    # 1. State of Health (SOH) Prediction
    soh_model, soh_features = load_soh_model()
    sample_soh = pd.DataFrame([{
        "cycle_number": 100,
        "mean_voltage": 3.75,
        "min_voltage": 3.0,
        "max_voltage": 4.2,
        "voltage_std": 0.15,
        "mean_current": 1.1,
        "mean_temperature": 25.0
    }])
    pred_soh = soh_model.predict(sample_soh[soh_features])[0]
    print(f"\n[1] State of Health (SOH):")
    print(f"    Input Features: {sample_soh.to_dict(orient='records')[0]}")
    print(f"    Predicted SOH:  {pred_soh:.4f}")

    # 2. State of Charge (SOC) Prediction
    soc_model, soc_features = load_soc_model()
    sample_soc = pd.DataFrame([{
        "voltage_V": 3.82,
        "current_A": -1.0,
        "temperature_C": 25.4,
        "time_s": 3600,
        "cycle_number": 100
    }])
    pred_soc = soc_model.predict(sample_soc[soc_features])[0]
    print(f"\n[2] State of Charge (SOC):")
    print(f"    Input Features: {sample_soc.to_dict(orient='records')[0]}")
    print(f"    Predicted SOC:  {pred_soc:.4f}")

    # 3. Anomaly Detection
    detector, detector_features, rules = load_anomaly_detector()
    sample_anomaly = pd.DataFrame([{
        "voltage_V": 3.82,
        "current_A": 0.5,
        "temperature_C": 25.4,
        "cycle_number": 100
    }])
    is_anomaly = detector.predict(sample_anomaly[detector_features])[0]
    status = "Normal" if is_anomaly == 1 else "Anomaly Detected"
    print(f"\n[3] Anomaly Detection:")
    print(f"    Input Features: {sample_anomaly.to_dict(orient='records')[0]}")
    print(f"    Safety Rules:   {rules}")
    print(f"    Status:         {status} (Code: {is_anomaly})")

    print("\n" + "=" * 60)
    print(" Inference completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
