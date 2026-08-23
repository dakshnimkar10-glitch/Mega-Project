# 🔋 NASA Li-ion Battery SOH/RUL Predictor

This repository contains a pre-trained, high-performance XGBoost model for predicting the **State of Health (SOH)** of Li-ion batteries using the NASA Battery Aging Dataset.

---

## 📊 Results

| Model | Test Dataset | MAE | RMSE | R² |
|---|---|---|---|---|
| **SOH Predictor** | NASA_B0018 | 0.0086 | 0.0103 | **0.9822** |

> The SOH model achieves an **R² of 0.98** on the unseen battery cell — predicting health to within ~1% average error.

---

## 📁 Project Structure

```
.
├── configs/
│   └── custom/
│       └── xgb_nasa.yaml      # Model Configuration File
├── trained_models/
│   ├── xgb_nasa_soh.pkl       # ⭐ Pre-trained SOH Model (R²=0.98)
│   └── xgb_nasa_rul.pkl       # Pre-trained RUL Model
├── preprocess_pipeline.py     # Data Preprocessing Script (.mat -> .pkl)
├── train_and_evaluate.py      # Standard Training Script
├── train_nasa.py              # Custom Training Script with engineered features
├── predict.py                 # ⭐ Prediction script to directly use the model
└── README.md                  # Documentation
```

---

## 🚀 How to Run

### 1. Install Dependencies
Make sure you have the required dependencies and the `BatteryML` package installed:
```bash
pip install xgboost scikit-learn scipy numpy
pip install -e git+https://github.com/microsoft/BatteryML#egg=batteryml
```

### 2. Preprocess the Data (If retraining)
If you want to run the prediction on new cells or retrain the model, place the raw `.mat` files in `data/raw/NASA/` and run:
```bash
python preprocess_pipeline.py --datasets NASA
```

### 3. Run Predictions Directly (Using the pre-trained model)
You can directly run predictions using the pre-trained model on any of the processed cells (`NASA_B0005`, `NASA_B0006`, `NASA_B0007`, `NASA_B0018`):
```bash
python predict.py --cell NASA_B0005
```

### 4. Retrain the Models
To train the SOH and RUL XGBoost models from scratch:
```bash
python train_nasa.py --seed 42
```
