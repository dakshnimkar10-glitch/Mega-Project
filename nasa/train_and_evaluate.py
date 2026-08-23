"""
NASA Battery ML – Train & Save Model
======================================
Trains an XGBoost SOH predictor on the NASA battery dataset and saves
the trained model as a .pkl file.

Usage:
    python scripts/train_and_evaluate.py [--seed 0]
"""

import sys
import pickle
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Ensure all batteryml modules are registered
import batteryml  # noqa: F401
import batteryml.preprocess  # noqa: F401

from batteryml.pipeline import Pipeline


CONFIG_PATH = ROOT / 'configs' / 'custom' / 'xgb_nasa.yaml'

MODELS_DIR = ROOT / 'trained_models'
MODELS_DIR.mkdir(exist_ok=True)


def run_pipeline(seed: int):
    processed_dir = ROOT / 'data' / 'processed' / 'NASA'
    if not processed_dir.exists() or not list(processed_dir.glob('*.pkl')):
        raise FileNotFoundError(
            f'No processed data found in {processed_dir}. '
            'Run scripts/preprocess_pipeline.py --datasets NASA first.'
        )

    workspace = ROOT / 'workspaces' / 'custom' / 'xgb_nasa'
    workspace.mkdir(parents=True, exist_ok=True)

    print(f'\n{"=" * 50}')
    print(f'  Training NASA model  (seed={seed})')
    print(f'{"=" * 50}')

    pipeline = Pipeline(CONFIG_PATH, str(workspace))
    model, dataset = pipeline.train(seed=seed)

    model_path = MODELS_DIR / 'xgb_nasa.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f'  Model saved -> {model_path}')


def main():
    parser = argparse.ArgumentParser(description='Train NASA Battery ML model')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()

    print('\n' + '=' * 50)
    print('  NASA Battery ML – Training Pipeline')
    print('=' * 50)

    try:
        run_pipeline(args.seed)
    except Exception as e:
        print(f'  [ERROR] {e}')

    print('\nDone! Model saved to:', MODELS_DIR)


if __name__ == '__main__':
    main()
