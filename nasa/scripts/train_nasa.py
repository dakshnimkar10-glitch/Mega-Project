"""
NASA Li-ion Battery SOH/RUL Predictor
======================================
A complete, standalone training pipeline for NASA Battery Dataset #5.

Features extracted per battery cell:
  - Discharge capacity fade statistics (mean, variance, slope, etc.)
  - dQ/dV curve features at key cycles
  - Voltage statistics per cycle
  - Temperature integration features
  - Capacity at early cycles

Labels:
  - SOH (State of Health) per cycle  = capacity / nominal_capacity
  - RUL (Remaining Useful Life)       = cycles until EOL (30% fade)

Models trained:
  - XGBoost SOH predictor (per-cycle regression)
  - XGBoost RUL predictor (per-cell regression)

Usage:
    python scripts/train_nasa.py [--seed 42] [--eol-threshold 0.8]
"""

import sys
import argparse
import pickle
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import scipy.interpolate
import scipy.stats
import xgboost as xgb
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import batteryml  # noqa
import batteryml.preprocess  # noqa
from batteryml.data.battery_data import BatteryData

PROCESSED_DIR = ROOT / 'data' / 'processed' / 'NASA'
MODELS_DIR = ROOT / 'trained_models'
MODELS_DIR.mkdir(exist_ok=True)

NASA_CELLS = ['NASA_B0005', 'NASA_B0006', 'NASA_B0007', 'NASA_B0018']

# =============================================================================
# Feature Engineering
# =============================================================================

def get_discharge_capacities(cell: BatteryData):
    """Return per-cycle peak discharge capacity (Ah)."""
    caps = []
    for c in cell.cycle_data:
        if c.discharge_capacity_in_Ah:
            caps.append(max(c.discharge_capacity_in_Ah))
        else:
            caps.append(np.nan)
    return np.array(caps)


def get_qdlin(cell: BatteryData, cycle_idx: int, n_points: int = 1000):
    """
    Interpolate discharge capacity vs voltage curve (dQ/dV style).
    Returns a 1D array of shape (n_points,) or all-NaN if unavailable.
    """
    c = cell.cycle_data[cycle_idx]
    if not c.discharge_capacity_in_Ah or not c.voltage_in_V:
        return np.full(n_points, np.nan)

    v = np.array(c.voltage_in_V)
    q = np.array(c.discharge_capacity_in_Ah)
    curr = np.array(c.current_in_A) if c.current_in_A else np.zeros_like(v)

    # Keep only discharge segments (current < 0 or capacity increasing)
    discharge_mask = (curr < 0) | (q > 0)
    v_d, q_d = v[discharge_mask], q[discharge_mask]

    if len(v_d) < 10:
        return np.full(n_points, np.nan)

    v_min = max(cell.min_voltage_limit_in_V or 2.7, v_d.min())
    v_max = min(cell.max_voltage_limit_in_V or 4.2, v_d.max())
    if v_max <= v_min:
        return np.full(n_points, np.nan)

    v_lin = np.linspace(v_min, v_max, n_points)
    try:
        # Sort by voltage for interpolation
        sort_idx = np.argsort(v_d)
        v_d_s, q_d_s = v_d[sort_idx], q_d[sort_idx]
        # Remove duplicates
        _, unique = np.unique(v_d_s, return_index=True)
        v_d_s, q_d_s = v_d_s[unique], q_d_s[unique]
        f = scipy.interpolate.interp1d(v_d_s, q_d_s, bounds_error=False, fill_value=np.nan)
        return f(v_lin)
    except Exception:
        return np.full(n_points, np.nan)


def extract_cell_features(cell: BatteryData) -> dict:
    """
    Extract a rich set of features from a single battery cell.
    Returns a dict of feature_name -> value.
    """
    caps = get_discharge_capacities(cell)
    valid = ~np.isnan(caps)
    caps_valid = caps[valid]
    n_cycles = len(caps_valid)

    features = {}

    # === Capacity statistics ===
    features['cap_initial'] = caps_valid[0] if n_cycles > 0 else np.nan
    features['cap_cycle5'] = np.nanmean(caps_valid[:5]) if n_cycles >= 5 else (caps_valid[0] if n_cycles > 0 else np.nan)
    features['cap_cycle10'] = np.nanmean(caps_valid[:10]) if n_cycles >= 10 else features['cap_cycle5']
    features['cap_max'] = np.nanmax(caps_valid) if n_cycles > 0 else np.nan
    features['cap_variance'] = np.nanvar(caps_valid) if n_cycles > 0 else 0.
    features['total_cycles'] = float(n_cycles)

    # === Capacity fade rate ===
    if n_cycles >= 20:
        x = np.arange(n_cycles).astype(float)
        slope, intercept, r, p, se = scipy.stats.linregress(x, caps_valid)
        features['cap_fade_slope'] = slope
        features['cap_fade_intercept'] = intercept
        features['cap_fade_r2'] = r ** 2
    else:
        features['cap_fade_slope'] = 0.
        features['cap_fade_intercept'] = caps_valid[0] if n_cycles > 0 else np.nan
        features['cap_fade_r2'] = 0.

    # === Early capacity drop ===
    if n_cycles >= 10:
        features['cap_drop_early'] = caps_valid[0] - np.nanmean(caps_valid[5:10])
    else:
        features['cap_drop_early'] = 0.

    # === dQ/dV curve difference (Severson-style) ===
    cycle_early = min(9, n_cycles - 1)
    cycle_late = min(99, n_cycles - 1)
    qdlin_early = get_qdlin(cell, cycle_early)
    qdlin_late = get_qdlin(cell, cycle_late)
    diff_qdlin = qdlin_late - qdlin_early
    diff_qdlin = diff_qdlin[~np.isnan(diff_qdlin)]

    if len(diff_qdlin) > 1:
        features['dqdv_min'] = np.log10(abs(diff_qdlin.min()) + 1e-8)
        features['dqdv_var'] = np.log10(diff_qdlin.var() + 1e-8)
        features['dqdv_skew'] = float(scipy.stats.skew(diff_qdlin))
        features['dqdv_kurtosis'] = float(scipy.stats.kurtosis(diff_qdlin))
        features['dqdv_mean'] = float(diff_qdlin.mean())
    else:
        features['dqdv_min'] = 0.
        features['dqdv_var'] = 0.
        features['dqdv_skew'] = 0.
        features['dqdv_kurtosis'] = 0.
        features['dqdv_mean'] = 0.

    # === Voltage statistics ===
    v_means, v_mins, v_vars = [], [], []
    for c in cell.cycle_data[:50]:
        if c.voltage_in_V and len(c.voltage_in_V) > 5:
            v = np.array(c.voltage_in_V)
            v_means.append(v.mean())
            v_mins.append(v.min())
            v_vars.append(v.var())
    features['volt_mean_early'] = np.mean(v_means) if v_means else np.nan
    features['volt_min_early'] = np.mean(v_mins) if v_mins else np.nan
    features['volt_var_early'] = np.mean(v_vars) if v_vars else 0.

    # === Temperature features ===
    temps = []
    for c in cell.cycle_data:
        if c.temperature_in_C:
            temps.extend(c.temperature_in_C)
    features['temp_mean'] = np.nanmean(temps) if temps else np.nan
    features['temp_max'] = np.nanmax(temps) if temps else np.nan
    features['temp_var'] = np.nanvar(temps) if temps else 0.

    # === Nominal capacity ===
    features['nominal_capacity'] = cell.nominal_capacity_in_Ah or 2.0

    return features


def extract_per_cycle_features(cell: BatteryData, eol_threshold: float = 0.8):
    """
    For SOH prediction: build one row per cycle.
    Features are rolling statistics up to that cycle.
    Returns X (n_cycles, n_features), y_soh, y_rul arrays.
    """
    caps = get_discharge_capacities(cell)
    nominal = cell.nominal_capacity_in_Ah or 2.0
    soh = caps / nominal  # State of Health per cycle

    eol_cap = nominal * eol_threshold
    eol_cycle = len(caps)
    for i, c in enumerate(caps):
        if not np.isnan(c) and c < eol_cap:
            eol_cycle = i
            break

    rows_X, rows_soh, rows_rul = [], [], []
    for i, (cap, soh_val) in enumerate(zip(caps, soh)):
        if np.isnan(cap):
            continue
        rul = max(0, eol_cycle - i)

        # Rolling features up to cycle i
        window = caps[max(0, i-10):i+1]
        window_valid = window[~np.isnan(window)]

        row = {
            'cycle_idx': float(i),
            'soh': float(soh_val),
            'cap': float(cap),
            'cap_rolling_mean': float(np.nanmean(window_valid)) if len(window_valid) else cap,
            'cap_rolling_var': float(np.nanvar(window_valid)) if len(window_valid) > 1 else 0.,
            'cap_trend': float(window_valid[-1] - window_valid[0]) if len(window_valid) > 1 else 0.,
            'nominal_cap': float(nominal),
        }
        rows_X.append(row)
        rows_soh.append(float(soh_val))
        rows_rul.append(float(rul))

    feature_names = ['cycle_idx', 'cap', 'cap_rolling_mean', 'cap_rolling_var', 'cap_trend', 'nominal_cap']
    X = np.array([[r[f] for f in feature_names] for r in rows_X])
    return X, np.array(rows_soh), np.array(rows_rul), feature_names


# =============================================================================
# Training
# =============================================================================

def print_metrics(name: str, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f'    {name}:  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}')
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def train_rul_model(cells: list, seed: int, eol_threshold: float):
    """Train a cell-level RUL predictor using cell-level features."""
    print('\n  [1/2] Training RUL predictor (cell-level features)...')

    X_rows, y_rul = [], []
    for cell in cells:
        feats = extract_cell_features(cell)
        caps = get_discharge_capacities(cell)
        nominal = cell.nominal_capacity_in_Ah or 2.0
        eol_cap = nominal * eol_threshold
        eol_cycle = len(caps)
        for i, c in enumerate(caps):
            if not np.isnan(c) and c < eol_cap:
                eol_cycle = i
                break

        feat_vals = [
            feats['cap_initial'], feats['cap_cycle5'], feats['cap_cycle10'],
            feats['cap_max'], feats['cap_variance'], feats['total_cycles'],
            feats['cap_fade_slope'], feats['cap_fade_intercept'], feats['cap_fade_r2'],
            feats['cap_drop_early'],
            feats['dqdv_min'], feats['dqdv_var'], feats['dqdv_skew'],
            feats['dqdv_kurtosis'], feats['dqdv_mean'],
            feats['volt_mean_early'], feats['volt_min_early'], feats['volt_var_early'],
            feats['temp_mean'], feats['temp_max'], feats['temp_var'],
            feats['nominal_capacity'],
        ]
        X_rows.append(feat_vals)
        y_rul.append(float(eol_cycle))

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rul, dtype=np.float32)

    # Replace NaN/Inf
    X = np.nan_to_num(X, nan=0., posinf=0., neginf=0.)

    print(f'    Cells: {len(cells)}, Features: {X.shape[1]}, RUL range: {y.min():.0f}–{y.max():.0f} cycles')

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbosity=0,
        objective='reg:squarederror',
    )

    if len(cells) >= 3:
        loo = LeaveOneOut()
        y_pred_cv = cross_val_predict(model, X, y, cv=loo)
        print('    Leave-One-Out CV:')
        print_metrics('RUL (cycles)', y, y_pred_cv)

    model.fit(X, y)

    model_path = MODELS_DIR / 'xgb_nasa_rul.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'type': 'rul'}, f)
    print(f'    Saved -> {model_path}')
    return model


def train_soh_model(cells: list, seed: int, eol_threshold: float):
    """Train a per-cycle SOH predictor."""
    print('\n  [2/2] Training SOH predictor (per-cycle features)...')

    all_X, all_soh, all_rul = [], [], []
    cell_ids = []
    for cell in cells:
        X, soh, rul, feat_names = extract_per_cycle_features(cell, eol_threshold)
        all_X.append(X)
        all_soh.append(soh)
        all_rul.append(rul)
        cell_ids.extend([cell.cell_id] * len(soh))

    X_all = np.vstack(all_X).astype(np.float32)
    y_soh = np.concatenate(all_soh).astype(np.float32)
    X_all = np.nan_to_num(X_all, nan=0., posinf=0., neginf=0.)

    print(f'    Total samples: {len(y_soh)}, Features: {X_all.shape[1]}')
    print(f'    SOH range: {y_soh.min():.4f} – {y_soh.max():.4f}')

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        min_child_weight=5,
        random_state=seed,
        verbosity=0,
        objective='reg:squarederror',
    )

    # Stratified split: hold out one cell as test
    test_cell_id = cells[-1].cell_id
    train_mask = np.array(cell_ids) != test_cell_id
    test_mask = ~train_mask

    X_train, y_train = X_all[train_mask], y_soh[train_mask]
    X_test, y_test = X_all[test_mask], y_soh[test_mask]

    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    print(f'    Train (cells: B0005/B0006/B0007):')
    print_metrics('SOH', y_train, y_pred_train)
    print(f'    Test  (cell: {test_cell_id}):')
    print_metrics('SOH', y_test, y_pred_test)

    # Refit on all data
    model.fit(X_all, y_soh)

    model_path = MODELS_DIR / 'xgb_nasa_soh.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'feature_names': feat_names, 'type': 'soh'}, f)
    print(f'    Saved -> {model_path}')
    return model, feat_names


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train NASA Battery ML models')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eol-threshold', type=float, default=0.8,
                        help='End-of-life capacity fraction (default 0.8 = 80% of nominal)')
    args = parser.parse_args()

    print('\n' + '=' * 60)
    print('  NASA Li-ion Battery ML Training Pipeline')
    print('=' * 60)
    print(f'  Seed: {args.seed}  |  EOL threshold: {args.eol_threshold*100:.0f}% of nominal capacity')

    # Load cells
    print(f'\n  Loading {len(NASA_CELLS)} battery cells from {PROCESSED_DIR}...')
    cells = []
    for name in NASA_CELLS:
        pkl = PROCESSED_DIR / f'{name}.pkl'
        if not pkl.exists():
            print(f'  [SKIP] {name} not found')
            continue
        cell = BatteryData.load(str(pkl))
        caps = get_discharge_capacities(cell)
        valid_caps = caps[~np.isnan(caps)]
        nominal = cell.nominal_capacity_in_Ah or 2.0
        soh_range = f'{(valid_caps.min()/nominal)*100:.1f}%–{(valid_caps.max()/nominal)*100:.1f}%'
        print(f'  Loaded {name}: {len(valid_caps)} cycles, SOH range {soh_range}')
        cells.append(cell)

    if not cells:
        print('[ERROR] No cells loaded. Run preprocessing first.')
        return

    np.random.seed(args.seed)

    # Train models
    rul_model = train_rul_model(cells, args.seed, args.eol_threshold)
    soh_model, feat_names = train_soh_model(cells, args.seed, args.eol_threshold)

    print('\n' + '=' * 60)
    print('  Training Complete!')
    print('  Models saved:')
    print(f'    - {MODELS_DIR / "xgb_nasa_rul.pkl"}  (RUL predictor)')
    print(f'    - {MODELS_DIR / "xgb_nasa_soh.pkl"}  (SOH predictor)')
    print('=' * 60 + '\n')


if __name__ == '__main__':
    main()
