#!/usr/bin/env python3
"""
Download THED-PV dataset from Zenodo.

This script downloads the THED-PV dataset files from Zenodo record 17404247.
It supports downloading specific file types or all files with progress tracking
and automatic checksum verification.

Usage:
    python download_data.py --output ./data --type all
    python download_data.py --output ./data --type raw
    python download_data.py --output ./data --type preprocessed
    python download_data.py --output ./data --type metadata
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Error: Required packages not installed.")
    print("Please install: pip install requests tqdm")
    sys.exit(1)

# Zenodo API configuration
ZENODO_RECORD_ID = "17404247"
ZENODO_API_BASE = "https://zenodo.org/api/records/"
ZENODO_RECORD_URL = f"{ZENODO_API_BASE}{ZENODO_RECORD_ID}"

# File type mappings - updated based on actual Zenodo structure
FILE_TYPES = {
    "all": None,  # Download everything
    "raw": ["raw_thermal_images"],  # Raw TIFF images
    "preprocessed": ["preprocessed_patches"],  # Preprocessed patches
    "metadata": ["environmental_metadata", "Code.zip"],  # Metadata and code
    "code": ["Code.zip"]  # Just the code/scripts
}


def get_zenodo_files():
    """
    Fetch file information from Zenodo API.
    
    Returns:
        list: List of file dictionaries with 'filename', 'size', 'checksum', and 'links'
    """
    print(f"Fetching file information from Zenodo record {ZENODO_RECORD_ID}...")
    
    try:
        response = requests.get(ZENODO_RECORD_URL, timeout=30)
        response.raise_for_status()
        record_data = response.json()
        
        files_info = []
        for file_entry in record_data.get("files", []):
            files_info.append({
                "filename": file_entry["key"],
                "size": file_entry["size"],
                "checksum": file_entry["checksum"],
                "links": file_entry["links"]["self"]
            })
        
        return files_info
    
    except requests.RequestException as e:
        print(f"Error fetching file information: {e}")
        sys.exit(1)


def compute_md5(filepath, chunk_size=8192):
    """
    Compute MD5 checksum of a file.
    
    Args:
        filepath: Path to the file
        chunk_size: Size of chunks to read (default 8KB)
    
    Returns:
        str: MD5 checksum as hex string
    """
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def verify_checksum(filepath, expected_checksum):
    """
    Verify file integrity using MD5 checksum.
    
    Args:
        filepath: Path to the downloaded file
        expected_checksum: Expected MD5 checksum (format: "md5:hash")
    
    Returns:
        bool: True if checksum matches, False otherwise
    """
    if not expected_checksum.startswith("md5:"):
        print(f"Warning: Unexpected checksum format: {expected_checksum}")
        return True  # Skip verification if format is unexpected
    
    expected_md5 = expected_checksum.split(":", 1)[1]
    computed_md5 = compute_md5(filepath)
    
    return computed_md5 == expected_md5


def download_file(url, output_path, filename, expected_size, checksum):
    """
    Download a file from URL with progress bar and checksum verification.
    
    Args:
        url: Download URL
        output_path: Directory to save the file
        filename: Name of the file
        expected_size: Expected file size in bytes
        checksum: Expected MD5 checksum
    
    Returns:
        bool: True if download successful, False otherwise
    """
    filepath = output_path / filename
    
    # Check if file already exists and is valid
    if filepath.exists():
        if filepath.stat().st_size == expected_size:
            print(f"File already exists: {filename}")
            if verify_checksum(filepath, checksum):
                print(f"✓ Checksum verified: {filename}")
                return True
            else:
                print(f"✗ Checksum mismatch: {filename}. Re-downloading...")
                filepath.unlink()
        else:
            print(f"Incomplete file found: {filename}. Re-downloading...")
            filepath.unlink()
    
    # Download the file
    print(f"Downloading: {filename} ({expected_size / (1024**3):.2f} GB)")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Create progress bar
        progress_bar = tqdm(
            total=expected_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=filename
        )
        
        # Download in chunks
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        progress_bar.close()
        
        # Verify checksum
        print(f"Verifying checksum for {filename}...")
        if verify_checksum(filepath, checksum):
            print(f"✓ Download complete and verified: {filename}")
            return True
        else:
            print(f"✗ Checksum verification failed: {filename}")
            filepath.unlink()
            return False
    
    except requests.RequestException as e:
        print(f"Error downloading {filename}: {e}")
        if filepath.exists():
            filepath.unlink()
        return False


def filter_files_by_type(files, file_type):
    """
    Filter files based on requested type.
    
    Args:
        files: List of file dictionaries
        file_type: Type of files to download ('all', 'raw', 'preprocessed', 'metadata')
    
    Returns:
        list: Filtered list of files
    """
    if file_type == "all":
        return files
    
    patterns = FILE_TYPES.get(file_type, [])
    if not patterns:
        print(f"Warning: Unknown file type '{file_type}'. Downloading all files.")
        return files
    
    filtered = []
    for file_info in files:
        filename = file_info["filename"]
        if any(pattern in filename for pattern in patterns):
            filtered.append(file_info)
    
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Download THED-PV dataset from Zenodo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all files
  python download_data.py --output ./data --type all
  
  # Download only raw thermal images
  python download_data.py --output ./data --type raw
  
  # Download preprocessed patches
  python download_data.py --output ./data --type preprocessed
  
  # Download metadata and code
  python download_data.py --output ./data --type metadata

File Types:
  all           : Download all dataset files (~26.6 GB total)
  raw           : Raw thermal images (640×512, 14-bit TIFF)
  preprocessed  : Preprocessed patches (320×256)
  metadata      : Environmental metadata CSV files and code
  code          : Code and scripts only
        """
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory to save downloaded files"
    )
    
    parser.add_argument(
        "--type",
        type=str,
        default="all",
        choices=["all", "raw", "preprocessed", "metadata", "code"],
        help="Type of files to download (default: all)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("THED-PV Dataset Downloader")
    print("=" * 70)
    print(f"Zenodo Record: {ZENODO_RECORD_ID}")
    print(f"Output Directory: {output_path.absolute()}")
    print(f"Download Type: {args.type}")
    print("=" * 70)
    
    # Fetch file information from Zenodo
    all_files = get_zenodo_files()
    
    if not all_files:
        print("Error: No files found in Zenodo record.")
        sys.exit(1)
    
    # Filter files based on type
    files_to_download = filter_files_by_type(all_files, args.type)
    
    if not files_to_download:
        print(f"No files found matching type '{args.type}'")
        sys.exit(1)
    
    # Calculate total size
    total_size = sum(f["size"] for f in files_to_download)
    print(f"\nFound {len(files_to_download)} file(s) to download")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    print()
    
    # Download files
    successful = 0
    failed = 0
    
    for idx, file_info in enumerate(files_to_download, 1):
        print(f"\n[{idx}/{len(files_to_download)}]")
        
        success = download_file(
            url=file_info["links"],
            output_path=output_path,
            filename=file_info["filename"],
            expected_size=file_info["size"],
            checksum=file_info["checksum"]
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)
    print(f"Successful: {successful}/{len(files_to_download)}")
    print(f"Failed: {failed}/{len(files_to_download)}")
    
    if failed > 0:
        print("\nSome downloads failed. Please check the errors above and retry.")
        sys.exit(1)
    else:
        print("\n✓ All files downloaded successfully!")
        print(f"Files saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()