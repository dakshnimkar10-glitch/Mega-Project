# CALCE Battery Models

This directory contains trained Machine Learning models for State of Health (SOH), State of Charge (SOC), Anomaly Detection, and Remaining Useful Life (RUL) estimation on the **CALCE (Center for Advanced Life Cycle Engineering)** battery dataset.

## Models Included

| Model File | Type | Algorithm | Input Features |
| :--- | :--- | :--- | :--- |
| `calce_soh_model.pkl` | SOH Estimation | `GradientBoostingRegressor` | `['cycle_number', 'mean_voltage', 'min_voltage', 'max_voltage', 'voltage_std', 'mean_current', 'mean_temperature']` |
| `calce_soc_model.pkl` | SOC Estimation | `RandomForestRegressor` | `['voltage_V', 'current_A', 'temperature_C', 'time_s', 'cycle_number']` |
| `calce_anomaly_detector.pkl` | Anomaly Detection | `IsolationForest` | `['voltage_V', 'current_A', 'temperature_C', 'cycle_number']` |
| `xgb_calce.pkl` | RUL Prediction | `XGBoostRULPredictor` (`batteryml`) | Extracted discharge features |

## Requirements

Install standard scientific computing packages:
```bash
pip install scikit-learn pandas numpy
```

*(Note: `xgb_calce.pkl` additionally requires `xgboost` and `batteryml`)*

## Quick Start / Inference

Run the demo script to verify all models:
```bash
python predict.py
```
