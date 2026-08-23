"""
NASA Battery ML – Predict State of Health (SOH) & RUL
======================================================
Loads a pre-trained SOH model and evaluates it on preprocessed NASA battery cells.

Usage:
    python predict.py [--cell NASA_B0005]
"""

import sys
import pickle
import argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import batteryml  # noqa
from batteryml.data.battery_data import BatteryData

def load_model(model_path: Path):
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    return bundle['model'], bundle.get('feature_names', [])

def get_discharge_capacities(cell: BatteryData):
    caps = []
    for c in cell.cycle_data:
        if c.discharge_capacity_in_Ah:
            caps.append(max(c.discharge_capacity_in_Ah))
        else:
            caps.append(np.nan)
    return np.array(caps)

def main():
    parser = argparse.ArgumentParser(description='Predict SOH for NASA Batteries')
    parser.add_argument('--cell', type=str, default='NASA_B0005', 
                        choices=['NASA_B0005', 'NASA_B0006', 'NASA_B0007', 'NASA_B0018'],
                        help='Cell to run predictions on')
    args = parser.parse_args()

    model_path = ROOT / 'trained_models' / 'xgb_nasa_soh.pkl'
    if not model_path.exists():
        print(f"[ERROR] Trained model not found at {model_path}. Please run train_nasa.py first.")
        return

    print("Loading model...")
    model, feature_names = load_model(model_path)
    
    cell_path = ROOT / 'data' / 'processed' / 'NASA' / f'{args.cell}.pkl'
    if not cell_path.exists():
        print(f"[ERROR] Processed cell data not found at {cell_path}.")
        print("Please place the raw NASA dataset in data/raw/NASA and run preprocess_pipeline.py first.")
        return

    print(f"Loading cell data for {args.cell}...")
    cell = BatteryData.load(str(cell_path))
    caps = get_discharge_capacities(cell)
    nominal = cell.nominal_capacity_in_Ah or 2.0

    print("\nExtracting features and predicting SOH...")
    # SOH = capacity / nominal_capacity
    y_true = caps / nominal
    
    # Extract cycle features
    rows_X = []
    for i, cap in enumerate(caps):
        if np.isnan(cap):
            continue
        window = caps[max(0, i-10):i+1]
        window_valid = window[~np.isnan(window)]
        
        row = {
            'cycle_idx': float(i),
            'cap': float(cap),
            'cap_rolling_mean': float(np.nanmean(window_valid)) if len(window_valid) else cap,
            'cap_rolling_var': float(np.nanvar(window_valid)) if len(window_valid) > 1 else 0.,
            'cap_trend': float(window_valid[-1] - window_valid[0]) if len(window_valid) > 1 else 0.,
            'nominal_cap': float(nominal),
        }
        rows_X.append([row[f] for f in feature_names])
    
    X = np.array(rows_X, dtype=np.float32)
    y_pred = model.predict(X)
    
    # Show predictions for every 20 cycles
    print(f"\nPredictions for {args.cell}:")
    print(f"{'Cycle':<10}{'Capacity (Ah)':<20}{'True SOH (%)':<15}{'Predicted SOH (%)':<20}{'Error (%)':<10}")
    print("-" * 75)
    
    errors = []
    for idx, (t, p, c) in enumerate(zip(y_true, y_pred, caps)):
        error = abs(t - p) * 100
        errors.append(error)
        if idx % 20 == 0 or idx == len(y_pred) - 1:
            print(f"{idx:<10}{c:<20.4f}{t*100:<15.2f}{p*100:<20.2f}{error:<10.2f}")
            
    print("-" * 75)
    print(f"Mean Absolute Error (MAE): {np.mean(errors):.3f}% SOH")

if __name__ == '__main__':
    main()
