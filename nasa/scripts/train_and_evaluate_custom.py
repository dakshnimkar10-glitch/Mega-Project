"""
Battery ML – Train & Save Models
=================================
Trains XGBoost SOH predictors on all processed battery datasets and saves
the trained models as .pkl files.

Usage:
    python scripts/train_and_evaluate_custom.py [--configs CONFIG [CONFIG ...]]
                                         [--skip-missing]
                                         [--seed 0]
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Ensure all batteryml modules are registered
import batteryml  # noqa: F401
import batteryml.preprocess  # noqa: F401

from batteryml.pipeline import Pipeline


ALL_CONFIGS = {
    'CALCE':  'configs/custom/xgb_calce.yaml',
    'NASA':   'configs/custom/xgb_nasa.yaml',
    'HUST':   'configs/custom/xgb_hust.yaml',
    'Oxford': 'configs/custom/xgb_oxford.yaml',
}

MODELS_DIR = ROOT / 'trained_models'
MODELS_DIR.mkdir(exist_ok=True)


def run_pipeline(name: str, config_path: str, seed: int, skip_missing: bool):
    config_p = ROOT / config_path

    dataset_name = name.upper() if name.upper() in ('CALCE', 'NASA', 'HUST') else name
    processed_dir = ROOT / 'data' / 'processed' / dataset_name
    if not processed_dir.exists() or not list(processed_dir.glob('*.pkl')):
        if skip_missing:
            print(f'  [{name}] No processed data found – skipping.')
            return
        else:
            raise FileNotFoundError(
                f'No processed data for {name} in {processed_dir}. '
                'Run scripts/preprocess_pipeline.py first.'
            )

    print(f'\n{"=" * 50}')
    print(f'  Training on: {name}  (seed={seed})')
    print(f'{"=" * 50}')

    workspace = ROOT / 'workspaces' / 'custom' / f'xgb_{name.lower()}'
    workspace.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(config_p, str(workspace))
    model, dataset = pipeline.train(seed=seed)

    # Save model
    import pickle
    model_path = MODELS_DIR / f'xgb_{name.lower()}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f'  [{name}] Model saved -> {model_path}')


def main():
    parser = argparse.ArgumentParser(description='Train Battery ML models')
    parser.add_argument('--configs', nargs='+', choices=list(ALL_CONFIGS.keys()),
                        default=list(ALL_CONFIGS.keys()),
                        help='Datasets to run (default: all)')
    parser.add_argument('--skip-missing', action='store_true',
                        help='Skip datasets whose processed data is not available')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()

    print('\n' + '=' * 50)
    print('  Battery ML – Training Pipeline')
    print('=' * 50)

    for name in args.configs:
        config_path = ALL_CONFIGS[name]
        try:
            run_pipeline(name, config_path, args.seed, args.skip_missing)
        except Exception as e:
            print(f'  [ERROR] {name}: {e}')

    print('\nDone! Trained models saved to:', MODELS_DIR)


if __name__ == '__main__':
    main()
