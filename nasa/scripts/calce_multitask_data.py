"""
CALCE Multi-Task Data Extractor & Feature Builder
==================================================
Loads processed CALCE battery files and generates tabular datasets for:
  1. SOC Prediction (sample-level time-series measurements)
  2. SOH Prediction (cycle-level battery degradation metrics)
  3. Abnormality Detection (voltage & current sensor profiles)
"""

import glob
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data' / 'processed' / 'CALCE'
NOMINAL_CAPACITY = 1.1  # Ah for CALCE CS2 cells


def load_calce_cells():
    pkl_files = sorted(glob.glob(str(DATA_DIR / '*.pkl')))
    if not pkl_files:
        raise FileNotFoundError(f"No CALCE pickle files found in {DATA_DIR}")
    
    cells_data = {}
    for filepath in pkl_files:
        filename = Path(filepath).stem
        with open(filepath, 'rb') as f:
            cells_data[filename] = pickle.load(f)
    return cells_data


def extract_soc_dataset(cells_data, max_samples_per_cycle=100):
    """
    Extract sample-level feature matrix for SOC prediction:
    Features: [voltage_in_V, current_in_A, temperature_in_C, time_in_s, cycle_number]
    Target: SOC (%)
    """
    records = []
    
    for cell_id, cell_obj in cells_data.items():
        for cyc in cell_obj.cycle_data:
            c_num = cyc.cycle_number
            v_arr = np.array(cyc.voltage_in_V)
            i_arr = np.array(cyc.current_in_A)
            t_arr = np.array(cyc.temperature_in_C)
            time_arr = np.array(cyc.time_in_s)
            cap_arr = np.array(cyc.discharge_capacity_in_Ah)
            
            n_pts = len(v_arr)
            if n_pts == 0 or len(cap_arr) == 0:
                continue
                
            max_cap = np.max(cap_arr)
            if max_cap <= 0:
                max_cap = NOMINAL_CAPACITY
                
            for idx in range(0, n_pts, 5):
                # Calculate instantaneous State of Charge (%)
                curr_cap = cap_arr[idx] if idx < len(cap_arr) else cap_arr[-1]
                # For discharge cycle: SOC = (1 - curr_cap / max_cap) * 100
                soc_pct = max(0.0, min(100.0, (1.0 - (curr_cap / max_cap)) * 100.0))
                
                records.append({
                    'cell_id': cell_id,
                    'cycle_number': c_num,
                    'voltage_V': v_arr[idx],
                    'current_A': i_arr[idx],
                    'temperature_C': t_arr[idx],
                    'time_s': time_arr[idx],
                    'soc_pct': soc_pct
                })
                
    df = pd.DataFrame(records)
    return df


def extract_soh_dataset(cells_data):
    """
    Extract cycle-level feature matrix for SOH prediction:
    Features: [cycle_number, mean_voltage, min_voltage, max_voltage, mean_current, mean_temperature, voltage_std]
    Target: SOH (%) = (discharge_capacity / nominal_capacity) * 100
    """
    records = []
    
    for cell_id, cell_obj in cells_data.items():
        for cyc in cell_obj.cycle_data:
            c_num = cyc.cycle_number
            v_arr = np.array(cyc.voltage_in_V)
            i_arr = np.array(cyc.current_in_A)
            t_arr = np.array(cyc.temperature_in_C)
            cap_arr = np.array(cyc.discharge_capacity_in_Ah)
            
            if len(v_arr) == 0 or len(cap_arr) == 0:
                continue
                
            max_cap = np.max(cap_arr)
            soh_pct = (max_cap / NOMINAL_CAPACITY) * 100.0
            soh_pct = max(0.0, min(120.0, soh_pct))
            
            records.append({
                'cell_id': cell_id,
                'cycle_number': c_num,
                'capacity_Ah': max_cap,
                'mean_voltage': np.mean(v_arr),
                'min_voltage': np.min(v_arr),
                'max_voltage': np.max(v_arr),
                'voltage_std': np.std(v_arr),
                'mean_current': np.mean(i_arr),
                'mean_temperature': np.mean(t_arr),
                'soh_pct': soh_pct
            })
            
    df = pd.DataFrame(records)
    return df


def generate_anomaly_dataset(soc_df, synthetic_anomalies_ratio=0.08, seed=42):
    """
    Build anomaly detection dataset with baseline sensor data and labeled synthetic abnormalities
    (Overvoltage, Undervoltage, Current Spikes, Extreme Temperature).
    """
    np.random.seed(seed)
    df = soc_df.copy()
    df['is_anomaly'] = 0
    df['anomaly_type'] = 'NORMAL'
    
    n_anomalies = int(len(df) * synthetic_anomalies_ratio)
    anomaly_indices = np.random.choice(df.index, size=n_anomalies, replace=False)
    
    types = ['OVERVOLTAGE', 'UNDERVOLTAGE', 'CURRENT_SPIKE', 'THERMAL_SPIKE']
    
    for idx in anomaly_indices:
        atype = np.random.choice(types)
        df.loc[idx, 'is_anomaly'] = 1
        df.loc[idx, 'anomaly_type'] = atype
        
        if atype == 'OVERVOLTAGE':
            df.loc[idx, 'voltage_V'] += np.random.uniform(0.6, 1.2)  # Exceed 4.2V max
        elif atype == 'UNDERVOLTAGE':
            df.loc[idx, 'voltage_V'] -= np.random.uniform(0.8, 1.5)  # Below 2.5V min
        elif atype == 'CURRENT_SPIKE':
            df.loc[idx, 'current_A'] += np.random.uniform(4.0, 8.0)  # Abnormal current draw
        elif atype == 'THERMAL_SPIKE':
            df.loc[idx, 'temperature_C'] += np.random.uniform(20.0, 35.0)  # Thermal surge
            
    return df


if __name__ == '__main__':
    cells = load_calce_cells()
    print(f"Loaded {len(cells)} CALCE battery cells.")
    
    soc_df = extract_soc_dataset(cells)
    print(f"SOC Dataset: {len(soc_df)} time-series samples extracted.")
    print("SOC Sample Head:\n", soc_df[['cell_id', 'cycle_number', 'voltage_V', 'current_A', 'soc_pct']].head())
    
    soh_df = extract_soh_dataset(cells)
    print(f"\nSOH Dataset: {len(soh_df)} cycle degradation records extracted.")
    print("SOH Sample Head:\n", soh_df[['cell_id', 'cycle_number', 'capacity_Ah', 'soh_pct']].head())
    
    anom_df = generate_anomaly_dataset(soc_df)
    print(f"\nAnomaly Dataset: {len(anom_df)} total records ({anom_df['is_anomaly'].sum()} anomalies).")
