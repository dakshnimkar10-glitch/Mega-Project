"""
Simple Inference Script to load and use the Oxford Battery Trained Models.
"""
import sys
import os
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.models.soc_model import SOCEstimator
from src.models.soh_model import SOHEstimator
from src.models.validation_model import BatterySensorValidator

def main():
    print("=" * 60)
    print("       OXFORD BATTERY TRAINED MODELS INFERENCE EXAMPLE       ")
    print("=" * 60)

    base_dir = os.path.abspath(os.path.dirname(__file__))
    # 1. Load Trained Models from disk
    soc_model = SOCEstimator.load(os.path.join(base_dir, "models", "soc_model.joblib"))
    soh_model = SOHEstimator.load(os.path.join(base_dir, "models", "soh_model.joblib"))
    validator = BatterySensorValidator.load(os.path.join(base_dir, "models", "validation_model.joblib"))

    print("\nModels loaded successfully!")
    print(f"  - SOC Model Type: {soc_model.model_type}")
    print(f"  - SOH Model Type: {soh_model.model_type}")

    # 2. Example Input Sensor Readings
    v_in = 3.85       # Voltage in Volts
    i_in = 740.0      # Current in mA
    t_in = 40.0       # Temperature in Deg C
    cycle_count = 200 # Battery Aging Cycle Count

    print(f"\n--- Testing Inputs: V = {v_in}V, I = {i_in}mA, T = {t_in} deg C, Cycle = {cycle_count} ---")

    # A. Validate Sensor Readings ("Is V, I, T Right or Not?")
    val_res = validator.validate_sample(voltage_v=v_in, current_ma=i_in, temp_c=t_in)
    print(f"\n[1] Sensor Validation Status: {val_res['status']}")
    print(f"    - Valid/Right: {val_res['is_valid']}")
    print(f"    - Anomaly Risk Score: {val_res['anomaly_score_pct']:.1f}%")
    if val_res['faults']:
        print("    - Faults:", val_res['faults'])

    # B. Predict State of Charge (SOC %)
    df_soc_sample = pd.DataFrame([{
        'voltage_v': v_in,
        'current_ma': i_in,
        'temp_c': t_in,
        'power_mw': v_in * i_in,
        'dv_dt': 0.0,
        'di_dt': 0.0,
        'dt_dt': 0.0,
        'v_roll_mean': v_in,
        'v_roll_std': 0.0,
        'i_roll_mean': i_in,
        't_roll_mean': t_in
    }])
    pred_soc = soc_model.predict(df_soc_sample)[0]
    print(f"\n[2] Predicted State of Charge (SOC): {pred_soc:.2f} %")

    # C. Predict State of Health (SOH %)
    df_soh_sample = pd.DataFrame([{
        'cycle_num': cycle_count,
        'avg_ch_temp': t_in,
        'max_ch_temp': t_in + 1.0,
        'temp_rise': 1.0,
        'time_to_4v': 2200.0
    }])
    pred_soh = soh_model.predict(df_soh_sample)[0]
    print(f"\n[3] Predicted State of Health (SOH): {pred_soh:.2f} %")

    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
