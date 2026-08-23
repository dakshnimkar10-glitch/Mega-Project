"""
Battery Dataset Download & Preprocessing Pipeline
==================================================
Downloads and preprocesses all supported battery aging datasets:
  - CALCE  (calce.umd.edu)
  - NASA   (data.nasa.gov)
  - Oxford (ora.ox.ac.uk)
  - HUST   (data.mendeley.com - HUST LFP)
  - MATR   (data.matr.io)

Usage:
    python preprocess_pipeline.py [--datasets CALCE NASA OX HUST MATR] [--skip-download]
"""

import os
import sys
import argparse
import zipfile
import urllib.request
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from batteryml.preprocess.download import download_file, DOWNLOAD_LINKS
from batteryml.builders import PREPROCESSORS

# Make sure all preprocessors are registered
import batteryml.preprocess  # noqa: F401


# ─── Dataset download configurations ────────────────────────────────────────

NASA_CELLS = ['B0005', 'B0006', 'B0007', 'B0018']
NASA_BASE_URL = (
    'https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip'
)

OX_DATASET_URL = (
    'https://ora.ox.ac.uk/objects/'
    'uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac/download_file?file_format=zip'
)

# ─── Helper utilities ────────────────────────────────────────────────────────

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_with_progress(url: str, dest: Path):
    """Download a file using urllib with a simple progress indicator."""
    if dest.exists():
        print(f'  [skip] {dest.name} already exists.')
        return

    print(f'  Downloading {dest.name} ...')
    dest.parent.mkdir(parents=True, exist_ok=True)

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size * 100 // total_size, 100)
            print(f'\r  Progress: {pct}%', end='', flush=True)

    try:
        urllib.request.urlretrieve(url, str(dest), reporthook)
        print()  # newline after progress
    except Exception as e:
        print(f'\n  [ERROR] Failed to download {url}: {e}')
        if dest.exists():
            dest.unlink()
        raise


def extract_zip(zip_path: Path, dest_dir: Path):
    print(f'  Extracting {zip_path.name} ...')
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest_dir)
    print(f'  Extracted to {dest_dir}')


# ─── Per-dataset downloaders ─────────────────────────────────────────────────

def download_calce(raw_dir: Path):
    print('\n[CALCE] Downloading CS2/CX2 battery cells ...')
    calce_dir = ensure_dir(raw_dir / 'CALCE')
    for url, filename in DOWNLOAD_LINKS['CALCE']:
        download_with_progress(url, calce_dir / filename)
    print('[CALCE] Download complete.')


def download_hust(raw_dir: Path):
    print('\n[HUST] Downloading HUST Mendeley dataset ...')
    hust_dir = ensure_dir(raw_dir / 'HUST')
    url, filename = DOWNLOAD_LINKS['HUST'][0]
    download_with_progress(url, hust_dir / filename)
    print('[HUST] Download complete.')


def download_matr(raw_dir: Path):
    print('\n[MATR] Downloading MATR batches ...')
    matr_dir = ensure_dir(raw_dir / 'MATR')
    for url, filename in DOWNLOAD_LINKS['MATR']:
        download_with_progress(url, matr_dir / filename)
    print('[MATR] Download complete.')


def download_nasa(raw_dir: Path):
    """Download NASA battery dataset zip and extract individual cell .mat files."""
    print('\n[NASA] Downloading NASA Li-ion Aging dataset ...')
    nasa_dir = ensure_dir(raw_dir / 'NASA')
    zip_path = nasa_dir / 'nasa_battery.zip'
    download_with_progress(NASA_BASE_URL, zip_path)
    if zip_path.exists():
        extract_zip(zip_path, nasa_dir)
        # Move .mat files to nasa_dir root for easy discovery
        for mat_file in nasa_dir.rglob('*.mat'):
            target = nasa_dir / mat_file.name
            if not target.exists():
                mat_file.rename(target)
    print('[NASA] Download complete.')


def download_oxford(raw_dir: Path):
    """Download Oxford battery degradation dataset."""
    print('\n[Oxford] Downloading Oxford battery dataset ...')
    ox_dir = ensure_dir(raw_dir / 'OX')
    zip_path = ox_dir / 'oxford_battery.zip'
    download_with_progress(OX_DATASET_URL, zip_path)
    if zip_path.exists():
        extract_zip(zip_path, ox_dir)
    print('[Oxford] Download complete.')


# ─── Preprocessing runner ─────────────────────────────────────────────────────

def preprocess_dataset(name: str, raw_subdir: Path, processed_dir: Path, silent: bool = False):
    out_dir = ensure_dir(processed_dir / name)
    print(f'\n[{name}] Preprocessing raw data from {raw_subdir} ...')

    if not raw_subdir.exists():
        print(f'  [SKIP] Raw directory not found: {raw_subdir}')
        return 0, 0

    processor = PREPROCESSORS.build(dict(
        name=f'{name}Preprocessor',
        output_dir=out_dir,
        silent=silent,
    ))
    result = processor(raw_subdir)
    if isinstance(result, tuple):
        processed, skipped = result
    else:
        processed, skipped = result, 0
    print(f'  Done: {processed} cells processed, {skipped} skipped.')
    return processed, skipped


# ─── Main entry point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Battery Dataset Pipeline')
    parser.add_argument(
        '--datasets', nargs='+',
        choices=['CALCE', 'NASA', 'OX', 'HUST', 'MATR'],
        default=['CALCE', 'NASA', 'OX', 'HUST', 'MATR'],
        help='Datasets to process'
    )
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip downloading and go straight to preprocessing')
    parser.add_argument('--silent', action='store_true',
                        help='Suppress verbose preprocessing output')
    args = parser.parse_args()

    raw_dir = ROOT / 'data' / 'raw'
    processed_dir = ROOT / 'data' / 'processed'
    ensure_dir(raw_dir)
    ensure_dir(processed_dir)

    download_fns = {
        'CALCE': download_calce,
        'HUST': download_hust,
        'MATR': download_matr,
        'NASA': download_nasa,
        'OX': download_oxford,
    }

    print('=' * 60)
    print('Battery ML Preprocessing Pipeline')
    print('=' * 60)
    print(f'Datasets: {args.datasets}')
    print(f'Raw dir:  {raw_dir}')
    print(f'Output:   {processed_dir}')
    print('=' * 60)

    total_processed = 0
    total_skipped = 0

    for dataset in args.datasets:
        # Download
        if not args.skip_download:
            try:
                download_fns[dataset](raw_dir)
            except Exception as e:
                print(f'  [WARNING] Download failed for {dataset}: {e}')
                print('  Attempting to preprocess existing data anyway ...')

        # Preprocess
        raw_subdir = raw_dir / dataset
        try:
            p, s = preprocess_dataset(dataset, raw_subdir, processed_dir, args.silent)
            total_processed += p
            total_skipped += s
        except Exception as e:
            print(f'  [ERROR] Preprocessing failed for {dataset}: {e}')

    print('\n' + '=' * 60)
    print(f'Pipeline complete: {total_processed} cells processed, {total_skipped} skipped.')
    print('=' * 60)


if __name__ == '__main__':
    main()
