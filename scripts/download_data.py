"""
Download HR-ThermalPV dataset from Zenodo
"""

import argparse
import requests
from pathlib import Path
import zipfile
import hashlib
from tqdm import tqdm
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Zenodo record information
ZENODO_RECORD_ID = "XXXXXX"  # UPDATE THIS after Zenodo upload
ZENODO_BASE_URL = f"https://zenodo.org/record/{ZENODO_RECORD_ID}/files"

# Dataset file information
DATASET_FILES = {
    'raw': {
        'filename': 'hr-thermalpv-raw.zip',
        'url': f"{ZENODO_BASE_URL}/hr-thermalpv-raw.zip",
        'size_gb': 8.0,
        'md5': None,  # UPDATE after upload
        'description': 'Raw 14-bit thermal images (12,460 images, 640×512)'
    },
    'preprocessed': {
        'filename': 'hr-thermalpv-preprocessed.zip',
        'url': f"{ZENODO_BASE_URL}/hr-thermalpv-preprocessed.zip",
        'size_gb': 4.0,
        'md5': None,  # UPDATE after upload
        'description': 'Preprocessed 8-bit patches (49,840 patches, 320×256)'
    },
    'homography_pairs': {
        'filename': 'hr-thermalpv-homography-pairs.zip',
        'url': f"{ZENODO_BASE_URL}/hr-thermalpv-homography-pairs.zip",
        'size_gb': 20.0,
        'md5': None,  # UPDATE after upload
        'description': 'Training homography pairs (99,680 pairs)'
    }
}


def compute_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 checksum of file"""
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def download_file(url: str, output_path: Path, expected_size_gb: float = None) -> bool:
    """Download file with progress bar"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Stream download
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        
        if expected_size_gb:
            expected_bytes = int(expected_size_gb * 1024**3)
            if abs(total_size - expected_bytes) > expected_bytes * 0.1:  # 10% tolerance
                logger.warning(f"File size mismatch. Expected ~{expected_size_gb:.1f}GB, got {total_size/(1024**3):.1f}GB")
        
        # Download with progress bar
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"Downloaded: {output_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def verify_checksum(filepath: Path, expected_md5: str) -> bool:
    """Verify file integrity using MD5"""
    if not expected_md5:
        logger.warning(f"No checksum available for {filepath.name}, skipping verification")
        return True
    
    logger.info(f"Verifying checksum for {filepath.name}...")
    actual_md5 = compute_md5(filepath)
    
    if actual_md5 == expected_md5:
        logger.info("✓ Checksum verified")
        return True
    else:
        logger.error(f"✗ Checksum mismatch!")
        logger.error(f"  Expected: {expected_md5}")
        logger.error(f"  Got:      {actual_md5}")
        return False


def extract_zip(zip_path: Path, output_dir: Path) -> bool:
    """Extract zip file with progress"""
    try:
        logger.info(f"Extracting {zip_path.name}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            
            with tqdm(total=len(members), desc="Extracting") as pbar:
                for member in members:
                    zip_ref.extract(member, output_dir)
                    pbar.update(1)
        
        logger.info(f"Extracted to: {output_dir}")
        return True
        
    except zipfile.BadZipFile:
        logger.error(f"Corrupted zip file: {zip_path}")
        return False
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def download_dataset(dataset_type: str, output_dir: Path, keep_zip: bool = False):
    """Download and extract dataset"""
    
    if dataset_type == 'all':
        types_to_download = ['raw', 'preprocessed', 'homography_pairs']
    else:
        types_to_download = [dataset_type]
    
    for dtype in types_to_download:
        if dtype not in DATASET_FILES:
            logger.error(f"Unknown dataset type: {dtype}")
            continue
        
        file_info = DATASET_FILES[dtype]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading: {dtype}")
        logger.info(f"Description: {file_info['description']}")
        logger.info(f"Size: ~{file_info['size_gb']:.1f} GB")
        logger.info(f"{'='*60}\n")
        
        # Download
        zip_path = output_dir / file_info['filename']
        
        if zip_path.exists():
            logger.info(f"File already exists: {zip_path}")
            user_input = input("Re-download? (y/n): ").lower()
            if user_input != 'y':
                logger.info("Skipping download")
            else:
                zip_path.unlink()
                if not download_file(file_info['url'], zip_path, file_info['size_gb']):
                    continue
        else:
            if not download_file(file_info['url'], zip_path, file_info['size_gb']):
                continue
        
        # Verify checksum
        if file_info['md5']:
            if not verify_checksum(zip_path, file_info['md5']):
                logger.error("Checksum verification failed. File may be corrupted.")
                logger.error("Please re-download or contact support.")
                continue
        
        # Extract
        extract_dir = output_dir / dtype
        if not extract_zip(zip_path, extract_dir):
            continue
        
        # Clean up zip file
        if not keep_zip:
            logger.info(f"Removing zip file: {zip_path}")
            zip_path.unlink()
        
        logger.info(f"✓ {dtype} dataset ready at: {extract_dir}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download HR-ThermalPV dataset from Zenodo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download raw data only
  python scripts/download_data.py --type raw --output ./data

  # Download all datasets
  python scripts/download_data.py --type all --output ./data

  # Keep zip files after extraction
  python scripts/download_data.py --type preprocessed --output ./data --keep-zip
        """
    )
    
    parser.add_argument('--type', type=str, required=True,
                        choices=['raw', 'preprocessed', 'homography_pairs', 'all'],
                        help='Dataset type to download')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for downloaded data')
    parser.add_argument('--keep-zip', action='store_true',
                        help='Keep zip files after extraction (default: delete)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check available disk space
    import shutil
    total, used, free = shutil.disk_usage(output_dir)
    free_gb = free / (1024**3)
    
    required_gb = {
        'raw': 16,          # 8GB compressed + 8GB extracted
        'preprocessed': 8,
        'homography_pairs': 40,
        'all': 64
    }
    
    if free_gb < required_gb[args.type]:
        logger.error(f"Insufficient disk space!")
        logger.error(f"  Available: {free_gb:.1f} GB")
        logger.error(f"  Required:  {required_gb[args.type]:.1f} GB")
        sys.exit(1)
    
    logger.info(f"Available disk space: {free_gb:.1f} GB")
    
    # Download
    download_dataset(args.type, output_dir, args.keep_zip)
    
    logger.info("\n" + "="*60)
    logger.info("Download complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()