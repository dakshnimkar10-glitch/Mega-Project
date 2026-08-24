"""
Test / Evaluate Trained Battery ML Model
========================================
Loads the trained model (trained_models/xgb_nasa.pkl) and tests it
on the test set, outputting error metrics and sample predictions.

Usage:
    python scripts/test_model.py [--config configs/custom/xgb_nasa.yaml] [--model trained_models/xgb_nasa.pkl]
"""

import sys
import pickle
import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import batteryml  # noqa: F401
import batteryml.preprocess  # noqa: F401
from batteryml.pipeline import load_config, build_dataset


def main():
    parser = argparse.ArgumentParser(description='Test trained BatteryML model')
    parser.add_argument('--config', type=str, default='configs/custom/xgb_nasa.yaml',
                        help='Path to dataset config YAML')
    parser.add_argument('--model', type=str, default='trained_models/xgb_nasa.pkl',
                        help='Path to saved model .pkl file')
    args = parser.parse_args()

    model_path = ROOT / args.model
    config_path = ROOT / args.config

    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        print("Please train the model first by running scripts/train_and_evaluate.py")
        sys.exit(1)

    print("=" * 50)
    print("  Testing Battery ML Model")
    print("=" * 50)
    print(f"  Model file:  {args.model}")
    print(f"  Config file: {args.config}")
    print("=" * 50)

    # 1. Load trained model
    print("\n[1/3] Loading saved model...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("  -> Model loaded successfully!")

    # 2. Build dataset split (uses cache if already processed)
    print("\n[2/3] Loading dataset...")
    config = load_config(args.config, workspace=None)
    dataset, _ = build_dataset(config, device='cpu')

    # 3. Predict & Evaluate
    print("\n[3/3] Running predictions on test data...")
    prediction = model.predict(dataset)
    
    metrics = ['RMSE', 'MAE', 'MAPE']
    scores = {m: float(dataset.evaluate(prediction, m)) for m in metrics}

    # Extract true & predicted labels in original units
    y_true = dataset.test_data.label
    y_pred = prediction
    if dataset.label_transformation is not None:
        y_true = dataset.label_transformation.inverse_transform(y_true)
        y_pred = dataset.label_transformation.inverse_transform(y_pred)

    y_true = y_true.cpu().numpy().flatten()
    y_pred = np.array(y_pred).flatten()

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    scores['R2'] = float(r2)

    print("\n" + "=" * 50)
    print("  TEST RESULTS")
    print("=" * 50)
    for metric_name, val in scores.items():
        if metric_name == 'MAPE':
            print(f"  {metric_name:<6}: {val * 100:.2f}%")
        else:
            print(f"  {metric_name:<6}: {val:.4f}")

    print("\n  Sample True vs Predicted RUL (Cycles):")
    print("  --------------------------------------")
    print(f"  {'Index':<8} {'True RUL':<12} {'Predicted RUL':<15} {'Error':<10}")
    print("  --------------------------------------")
    for i in range(min(10, len(y_true))):
        err = abs(y_true[i] - y_pred[i])
        print(f"  {i:<8} {y_true[i]:<12.1f} {y_pred[i]:<15.1f} {err:<10.2f}")
    print("=" * 50)


if __name__ == '__main__':
    main()
