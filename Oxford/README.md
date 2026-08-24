# Oxford Battery ML System: SOC, SOH & V/I/T Validation

This project contains a complete Machine Learning pipeline and interactive Streamlit dashboard for estimating battery **State of Charge (SOC)**, **State of Health (SOH)**, and performing real-time **Sensor Validation** using the Oxford Battery Degradation Dataset.

## Features
- **SOC Estimation**: XGBoost model to predict State of Charge based on V, I, T and their derivatives.
- **SOH Estimation**: Model to predict capacity degradation over aging cycles.
- **Sensor Validation**: Dual-layer (Physical rules + Isolation Forest) anomaly detection to verify if Voltage, Current, and Temperature readings are "RIGHT" or anomalous.
- **Streamlit Dashboard**: Interactive web app for real-time inference, dataset visualization, and batch CSV testing.

## Setup & Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the interactive dashboard:
   ```bash
   streamlit run app.py
   ```
3. Test inference programmatically:
   ```bash
   python predict_example.py
   ```

*(Note: The main `Oxford_Battery_Degradation_Dataset_1.mat` file is large and is excluded from git. The `train_all.py` script automatically downloads it if missing).*
