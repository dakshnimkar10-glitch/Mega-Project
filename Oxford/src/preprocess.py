import numpy as np
import pandas as pd

def extract_soc_features(df):
    """
    Extracts features for State of Charge (SOC) estimation from raw V, I, T time series.
    Input DataFrame must have columns: ['time_s', 'voltage_v', 'current_ma', 'temp_c']
    Optionally 'soc' if target is present.
    """
    data = df.copy()
    
    # Ensure sorted by time
    data.sort_values('time_s', inplace=True)
    
    # Power (mW)
    data['power_mw'] = data['voltage_v'] * data['current_ma']
    
    # Derivatives with respect to time (handling 0 dt gracefully)
    dt = data['time_s'].diff().fillna(1.0)
    dt = np.where(dt <= 0, 1.0, dt)
    
    data['dv_dt'] = data['voltage_v'].diff().fillna(0.0) / dt
    data['di_dt'] = data['current_ma'].diff().fillna(0.0) / dt
    data['dt_dt'] = data['temp_c'].diff().fillna(0.0) / dt
    
    # Rolling features (window=5)
    data['v_roll_mean'] = data['voltage_v'].rolling(window=5, min_periods=1).mean()
    data['v_roll_std'] = data['voltage_v'].rolling(window=5, min_periods=1).std().fillna(0.0)
    data['i_roll_mean'] = data['current_ma'].rolling(window=5, min_periods=1).mean()
    data['t_roll_mean'] = data['temp_c'].rolling(window=5, min_periods=1).mean()
    
    feature_cols = [
        'voltage_v', 'current_ma', 'temp_c', 'power_mw',
        'dv_dt', 'di_dt', 'dt_dt',
        'v_roll_mean', 'v_roll_std', 'i_roll_mean', 't_roll_mean'
    ]
    
    X = data[feature_cols]
    y = data['soc'] if 'soc' in data.columns else None
    
    return X, y, data


def extract_soh_cycle_features(cells_data, nominal_capacity=740.0):
    """
    Extracts cycle-level features and SOH targets across all cells and characterisation cycles in Oxford dataset.
    Returns DataFrame where each row is a cycle test with extracted features and SOH (%).
    """
    rows = []
    
    for cell_id, cell_obj in cells_data.items():
        cell_num = int(cell_id.replace('Cell', '')) if cell_id.replace('Cell', '').isdigit() else 0
        
        for cyc_id, cyc_obj in cell_obj.items():
            cyc_num = int(cyc_id.replace('cyc', '')) if cyc_id.replace('cyc', '').isdigit() else 0
            
            # Look for 1C discharge phase to compute max discharge capacity
            c1dc_df = cyc_obj.get('C1dc', None)
            c1ch_df = cyc_obj.get('C1ch', None)
            
            if c1dc_df is None or len(c1dc_df) == 0:
                continue
                
            q_vals = c1dc_df['charge_mah'].values
            q_discharged = np.ptp(q_vals)
            
            # SOH (% capacity retention)
            soh = (q_discharged / nominal_capacity) * 100.0
            
            # Extract features from 1C charge curve if available
            time_to_4v = 0.0
            avg_ch_temp = 40.0
            max_ch_temp = 40.0
            
            if c1ch_df is not None and len(c1ch_df) > 0:
                avg_ch_temp = c1ch_df['temp_c'].mean()
                max_ch_temp = c1ch_df['temp_c'].max()
                
                # time to reach 4.0V
                ch_4v = c1ch_df[c1ch_df['voltage_v'] >= 4.0]
                if len(ch_4v) > 0:
                    time_to_4v = ch_4v['time_s'].iloc[0] - c1ch_df['time_s'].iloc[0]
                else:
                    time_to_4v = c1ch_df['time_s'].iloc[-1] - c1ch_df['time_s'].iloc[0]

            rows.append({
                'cell_id': cell_id,
                'cell_num': cell_num,
                'cycle_num': cyc_num,
                'q_discharged_mah': q_discharged,
                'soh': soh,
                'avg_ch_temp': avg_ch_temp,
                'max_ch_temp': max_ch_temp,
                'temp_rise': max_ch_temp - avg_ch_temp,
                'time_to_4v': time_to_4v,
                'v_start_dc': c1dc_df['voltage_v'].iloc[0],
                'v_end_dc': c1dc_df['voltage_v'].iloc[-1],
                'temp_dc_mean': c1dc_df['temp_c'].mean()
            })
            
    df_soh = pd.DataFrame(rows)
    return df_soh


if __name__ == '__main__':
    print("Testing preprocessing module...")
    from data_loader import load_example_dc
    ex = load_example_dc('ExampleDC_C1.mat')
    X, y, _ = extract_soc_features(ex['dc'])
    print("SOC Feature matrix shape:", X.shape)
    print("SOC Feature columns:", X.columns.tolist())
