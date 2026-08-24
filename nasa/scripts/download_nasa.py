"""
NASA Battery Download helper
Downloads individual .mat files from the NASA Prognostics Center of Excellence 
Dataset Repository (Battery Data Set #5).

Cells: B0005, B0006, B0007, B0018
"""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# NASA battery dataset - individual cell files via direct links
# Source: https://data.nasa.gov/dataset/li-ion-battery-aging-datasets  
# Hosted at: https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip

NASA_CELL_URLS = {
    'B0005': 'https://data.nasa.gov/api/views/ufpe-8yvy/files/071f0e46-38c3-4b9f-a8a9-5d2e44c87b3b?download=true&filename=B0005.mat',
    'B0006': 'https://data.nasa.gov/api/views/ufpe-8yvy/files/c1cac7da-e7e4-4f7c-9b07-9de5ef9b0e74?download=true&filename=B0006.mat',
    'B0007': 'https://data.nasa.gov/api/views/ufpe-8yvy/files/ff06e2f3-e543-47e5-83d8-1451e1aa01fe?download=true&filename=B0007.mat',
    'B0018': 'https://data.nasa.gov/api/views/ufpe-8yvy/files/bca25e62-3c4d-4571-9897-e5f0cc97ac04?download=true&filename=B0018.mat',
}

# Alternative fallback URLs (direct S3 links from NASA Prognostics Center)
NASA_S3_FALLBACK = {
    'B0005': 'https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set/B0005.mat',
    'B0006': 'https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set/B0006.mat',
    'B0007': 'https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set/B0007.mat',
    'B0018': 'https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set/B0018.mat',
}


def download_nasa():
    nasa_dir = ROOT / 'data' / 'raw' / 'NASA'
    nasa_dir.mkdir(parents=True, exist_ok=True)

    print('Downloading NASA Battery Dataset cells ...')
    for cell, url in NASA_CELL_URLS.items():
        dest = nasa_dir / f'{cell}.mat'
        if dest.exists():
            print(f'  [skip] {cell}.mat already exists ({dest.stat().st_size / 1e6:.1f}MB)')
            continue

        print(f'  Trying primary URL for {cell} ...')
        success = try_download(url, dest)

        if not success:
            # Try S3 fallback
            fallback_url = NASA_S3_FALLBACK.get(cell)
            if fallback_url:
                print(f'  Trying S3 fallback for {cell} ...')
                success = try_download(fallback_url, dest)

        if not success:
            print(f'  [WARNING] Could not download {cell}. Please download manually from:')
            print(f'    https://data.nasa.gov/dataset/li-ion-battery-aging-datasets')
            print(f'    and place {cell}.mat in {nasa_dir}')


def try_download(url: str, dest: Path) -> bool:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; BatteryML/1.0)',
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as response:
            total = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            chunk_size = 65536
            with open(dest, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        print(f'\r  {dest.name}: {pct}%  ({downloaded/1e6:.1f}MB)', end='', flush=True)
            print(f'\r  {dest.name}: Done ({downloaded/1e6:.1f}MB)     ')
        return True
    except Exception as e:
        print(f'\r  [ERROR] {e}')
        if dest.exists():
            dest.unlink()
        return False


if __name__ == '__main__':
    download_nasa()
