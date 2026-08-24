import os
import scipy.io
import numpy as np
import pandas as pd

NOMINAL_CAPACITY_MAH = 740.0  # Kokam SLPB533459H4 cell nominal capacity (740 mAh)

def load_example_dc(mat_path='ExampleDC_C1.mat'):
    """
    Loads ExampleDC_C1.mat containing 1st cycle charge/discharge drive cycle data.
    Returns dictionary with 'ch' and 'dc' DataFrames.
    """
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"File {mat_path} not found.")
        
    mat = scipy.io.loadmat(mat_path)
    data = mat['ExampleDC_C1'][0, 0]
    
    result = {}
    for phase in ['ch', 'dc']:
        phase_data = data[phase][0, 0]
        df_dict = {}
        for field in phase_data.dtype.names:
            arr = phase_data[field].flatten()
            df_dict[field] = arr
        
        df = pd.DataFrame(df_dict)
        df.rename(columns={
            't': 'time_s',
            'v': 'voltage_v',
            'q': 'charge_mah',
            'T': 'temp_c',
            'i': 'current_ma'
        }, inplace=True)
        
        q_vals = df['charge_mah'].values
        q_range = np.ptp(q_vals) if len(q_vals) > 0 else NOMINAL_CAPACITY_MAH
        q_range = max(q_range, 1.0)
        
        if phase == 'ch':
            df['soc'] = np.clip((q_vals - np.min(q_vals)) / q_range * 100.0, 0.0, 100.0)
        else: # dc
            df['soc'] = np.clip((q_vals - np.min(q_vals)) / q_range * 100.0, 0.0, 100.0)
            
        result[phase] = df
        
    return result


def load_oxford_dataset(mat_path='Oxford_Battery_Degradation_Dataset_1.mat'):
    """
    Loads full Oxford Battery Degradation Dataset 1.
    Returns dictionary structured as:
    {
       'Cell1': {
           'cyc0000': {
               'C1ch': DataFrame(time_s, voltage_v, charge_mah, temp_c, current_ma, soc),
               'C1dc': DataFrame(...),
               'OCVch': DataFrame(...),
               'OCVdc': DataFrame(...)
           },
           ...
       },
       ...
    }
    """
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"File {mat_path} not found.")

    mat = scipy.io.loadmat(mat_path, simplify_cells=True)
    cells_data = {}

    for key in mat.keys():
        if key.startswith('__'):
            continue
        cell_obj = mat[key]
        if not isinstance(cell_obj, dict):
            continue
            
        cell_dict = {}
        for cyc_key, cyc_val in cell_obj.items():
            if not isinstance(cyc_val, dict):
                continue
            cyc_dict = {}
            for phase_key, phase_val in cyc_val.items():
                if isinstance(phase_val, dict):
                    df_dict = {}
                    for field in ['t', 'v', 'q', 'T']:
                        if field in phase_val:
                            df_dict[field] = np.array(phase_val[field]).flatten()
                    
                    if not df_dict:
                        continue
                        
                    df = pd.DataFrame(df_dict)
                    df.rename(columns={
                        't': 'time_s',
                        'v': 'voltage_v',
                        'q': 'charge_mah',
                        'T': 'temp_c'
                    }, inplace=True)

                    # Infer current for constant current phases
                    # C1ch: +740 mA, C1dc: -740 mA, OCVch: +40 mA, OCVdc: -40 mA
                    if 'ch' in phase_key:
                        c_val = 740.0 if 'C1' in phase_key else 40.0
                    else:
                        c_val = -740.0 if 'C1' in phase_key else -40.0
                    df['current_ma'] = c_val

                    # Calculate SOC
                    q_vals = df['charge_mah'].values
                    q_range = np.ptp(q_vals) if len(q_vals) > 0 else NOMINAL_CAPACITY_MAH
                    q_range = max(q_range, 1.0)
                    
                    df['soc'] = np.clip((q_vals - np.min(q_vals)) / q_range * 100.0, 0.0, 100.0)

                    cyc_dict[phase_key] = df
            cell_dict[cyc_key] = cyc_dict
        cells_data[key] = cell_dict

    return cells_data


if __name__ == '__main__':
    print("Testing data loader SOC calculation...")
    ex_dc = load_example_dc('ExampleDC_C1.mat')
    print("Discharge SOC range:", ex_dc['dc']['soc'].min(), "to", ex_dc['dc']['soc'].max())
    print("Discharge head:\n", ex_dc['dc'][['time_s', 'voltage_v', 'current_ma', 'charge_mah', 'soc']].head())
    print("Discharge tail:\n", ex_dc['dc'][['time_s', 'voltage_v', 'current_ma', 'charge_mah', 'soc']].tail())
