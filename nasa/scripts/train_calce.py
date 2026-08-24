"""
Battery ML – Train & Save CALCE XGBoost Model
==============================================
Trains the XGBoost SOH/RUL predictor on the CALCE battery dataset
and saves the trained model as trained_models/xgb_calce.pkl.

Usage:
    python scripts/train_calce.py [--seed 0]
"""

import sys
import pickle
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import batteryml  # noqa: F401
import batteryml.preprocess  # noqa: F401
from batteryml.pipeline import Pipeline

CONFIG_PATH = 'configs/custom/xgb_calce.yaml'
MODELS_DIR = ROOT / 'trained_models'
MODELS_DIR.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Train CALCE Battery ML model')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = parser.parse_args()

    print('\n' + '=' * 55)
    print('  Battery ML – Training Pipeline (CALCE Dataset)')
    print('=' * 55)

    config_p = ROOT / CONFIG_PATH
    processed_dir = ROOT / 'data' / 'processed' / 'CALCE'

    if not processed_dir.exists() or not list(processed_dir.glob('*.pkl')):
        raise FileNotFoundError(
            f'No preprocessed CALCE data found in {processed_dir}.'
        )

    print(f'  Config: {config_p}')
    print(f'  Training XGBoost model on CALCE dataset (seed={args.seed})...')
    
    workspace = ROOT / 'workspaces' / 'custom' / 'xgb_calce'
    workspace.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(config_p, str(workspace))
    model, dataset = pipeline.train(seed=args.seed)

    # Save trained model
    model_path = MODELS_DIR / 'xgb_calce.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f'\n  [SUCCESS] CALCE Model trained and saved -> {model_path}')
    print('=' * 55)


if __name__ == '__main__':
    main()
