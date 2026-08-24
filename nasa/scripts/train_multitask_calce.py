"""
Train Multi-Task Models on CALCE Dataset
=========================================
Trains and evaluates 3 dedicated machine learning models:
  1. SOC Model: Predicts State of Charge (%) from sensor telemetry.
  2. SOH Model: Predicts State of Health (%) from cycle degradation features.
  3. Anomaly Detector: IsolationForest + Rule Engine for Voltage & Current abnormalities.

Saves trained model artifacts to trained_models/.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.calce_multitask_data import load_calce_cells, extract_soc_dataset, extract_soh_dataset, generate_anomaly_dataset

MODELS_DIR = ROOT / 'trained_models'
MODELS_DIR.mkdir(exist_ok=True)


def train_soc_model(soc_df):
    print("\n" + "=" * 55)
    print("  [1/3] Training SOC (State of Charge %) Regressor")
    print("=" * 55)

    feature_cols = ['voltage_V', 'current_A', 'temperature_C', 'time_s', 'cycle_number']
    X = soc_df[feature_cols].values
    y = soc_df['soc_pct'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    mape = float(np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-5))) * 100.0)

    print(f"  SOC Model Evaluation Results:")
    print(f"    - MAE  : {mae:.2f}%")
    print(f"    - RMSE : {rmse:.2f}%")
    print(f"    - MAPE : {mape:.2f}%")
    print(f"    - R²   : {r2:.4f}")

    save_path = MODELS_DIR / 'calce_soc_model.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump({'model': model, 'feature_cols': feature_cols}, f)
    print(f"  [SUCCESS] Saved SOC Model -> {save_path.name}")

    return model, {'MAE': mae, 'RMSE': rmse, 'R2': r2}


def train_soh_model(soh_df):
    print("\n" + "=" * 55)
    print("  [2/3] Training SOH (State of Health %) Regressor")
    print("=" * 55)

    feature_cols = ['cycle_number', 'mean_voltage', 'min_voltage', 'max_voltage', 'voltage_std', 'mean_current', 'mean_temperature']
    X = soh_df[feature_cols].values
    y = soh_df['soh_pct'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=120, max_depth=5, learning_rate=0.08, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100.0)

    print(f"  SOH Model Evaluation Results:")
    print(f"    - MAE  : {mae:.2f}%")
    print(f"    - RMSE : {rmse:.2f}%")
    print(f"    - MAPE : {mape:.2f}%")
    print(f"    - R²   : {r2:.4f}")

    save_path = MODELS_DIR / 'calce_soh_model.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump({'model': model, 'feature_cols': feature_cols}, f)
    print(f"  [SUCCESS] Saved SOH Model -> {save_path.name}")

    return model, {'MAE': mae, 'RMSE': rmse, 'R2': r2}


def train_anomaly_detector(anom_df):
    print("\n" + "=" * 55)
    print("  [3/3] Training Voltage & Current Abnormality Detector")
    print("=" * 55)

    feature_cols = ['voltage_V', 'current_A', 'temperature_C', 'cycle_number']

    # Train IsolationForest on normal baseline data
    normal_df = anom_df[anom_df['is_anomaly'] == 0]
    X_normal = normal_df[feature_cols].values

    iso_forest = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    iso_forest.fit(X_normal)

    # Test on full labeled dataset with synthetic anomalies
    X_full = anom_df[feature_cols].values
    y_true = anom_df['is_anomaly'].values

    # IsolationForest returns -1 for anomaly, 1 for normal
    iso_preds = iso_forest.predict(X_full)
    iso_anom_binary = np.where(iso_preds == -1, 1, 0)

    # Domain rule threshold checks for deterministic boundary protection
    rule_anom_binary = np.where(
        (anom_df['voltage_V'] > 4.25) | 
        (anom_df['voltage_V'] < 2.50) | 
        (np.abs(anom_df['current_A']) > 3.5) | 
        (anom_df['temperature_C'] > 45.0),
        1, 0
    )

    # Combined diagnostic prediction
    final_preds = np.maximum(iso_anom_binary, rule_anom_binary)

    acc = float(accuracy_score(y_true, final_preds))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, final_preds, average='binary')

    print(f"  Abnormality Detector Evaluation Results:")
    print(f"    - Accuracy  : {acc * 100:.2f}%")
    print(f"    - Precision : {prec * 100:.2f}%")
    print(f"    - Recall    : {rec * 100:.2f}%")
    print(f"    - F1-Score  : {f1 * 100:.2f}%")

    save_path = MODELS_DIR / 'calce_anomaly_detector.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump({
            'iso_forest': iso_forest,
            'feature_cols': feature_cols,
            'rules': {
                'v_max': 4.25,
                'v_min': 2.50,
                'i_max': 3.5,
                't_max': 45.0
            }
        }, f)
    print(f"  [SUCCESS] Saved Abnormality Detector -> {save_path.name}")

    return iso_forest, {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}


def main():
    print("=" * 55)
    print("  CALCE Multi-Task Training Pipeline (SOC, SOH, Anomalies)")
    print("=" * 55)

    print("\nLoading CALCE raw pickles & building dataset tables...")
    cells = load_calce_cells()
    
    soc_df = extract_soc_dataset(cells)
    soh_df = extract_soh_dataset(cells)
    anom_df = generate_anomaly_dataset(soc_df)

    train_soc_model(soc_df)
    train_soh_model(soh_df)
    train_anomaly_detector(anom_df)

    print("\n" + "=" * 55)
    print("  [COMPLETE] All 3 Multi-Task Models Trained & Saved!")
    print("=" * 55)


if __name__ == '__main__':
    main()
