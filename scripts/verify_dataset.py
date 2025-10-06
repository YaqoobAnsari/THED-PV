"""
Verify HR-ThermalPV dataset integrity and structure
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import json
from collections import defaultdict
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetVerifier:
    """Verify dataset integrity and structure"""
    
    def __init__(self, data_dir: Path, dataset_type: str):
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(int)
    
    def verify_raw_dataset(self):
        """Verify raw thermal images"""
        logger.info("Verifying raw dataset structure...")
        
        expected_structure = {
            'days': ['2024-12-21', '2024-12-22', '2024-12-23', '2024-12-24', '2024-12-25'],
            'times': ['8am', '10am', '12pm', '2pm'],
            'heights': ['10cm', '20cm', '30cm', '40cm'],
            'angles': ['30', '60']
        }
        
        # Check directory structure
        for day in expected_structure['days']:
            day_path = self.data_dir / day
            if not day_path.exists():
                self.errors.append(f"Missing day directory: {day}")
                continue
            
            for time in expected_structure['times']:
                time_path = day_path / time
                if not time_path.exists():
                    self.warnings.append(f"Missing time directory: {day}/{time}")
                    continue
                
                for height in expected_structure['heights']:
                    height_path = time_path / height
                    if not height_path.exists():
                        self.warnings.append(f"Missing height directory: {day}/{time}/{height}")
                        continue
                    
                    # Check for TIFF files
                    tiff_files = list(height_path.glob("*.tiff")) + list(height_path.glob("*.tif"))
                    
                    if not tiff_files:
                        self.warnings.append(f"No TIFF files in: {day}/{time}/{height}")
                    
                    # Verify each image
                    for tiff_file in tiff_files:
                        self.verify_raw_image(tiff_file)
        
        logger.info(f"Total raw images found: {self.stats['raw_images']}")
    
    def verify_raw_image(self, filepath: Path):
        """Verify single raw thermal image"""
        try:
            img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
            
            if img is None:
                self.errors.append(f"Failed to read: {filepath}")
                return
            
            # Check dimensions
            if img.shape != (512, 640):
                self.errors.append(f"Wrong dimensions {img.shape} (expected 512×640): {filepath.name}")
            
            # Check bit depth
            if img.dtype != np.uint16:
                self.errors.append(f"Wrong dtype {img.dtype} (expected uint16): {filepath.name}")
            
            # Check value range (14-bit should be 0-16383)
            max_val = img.max()
            if max_val > 16383:
                self.warnings.append(f"Pixel values exceed 14-bit range ({max_val}): {filepath.name}")
            
            # Check for all-black images
            if img.max() == 0:
                self.errors.append(f"Image is all black: {filepath.name}")
            
            self.stats['raw_images'] += 1
            
        except Exception as e:
            self.errors.append(f"Error verifying {filepath}: {e}")
    
    def verify_preprocessed_dataset(self):
        """Verify preprocessed patches"""
        logger.info("Verifying preprocessed dataset...")
        
        # Find all patches
        patch_files = list(self.data_dir.rglob("*_patch*.tiff"))
        
        if not patch_files:
            self.errors.append("No preprocessed patches found")
            return
        
        for patch_file in tqdm(patch_files, desc="Verifying patches"):
            self.verify_preprocessed_patch(patch_file)
        
        logger.info(f"Total preprocessed patches found: {self.stats['preprocessed_patches']}")
        
        # Check patch count (should be 4× raw image count)
        expected_patches = self.stats.get('raw_images', 12460) * 4
        if self.stats['preprocessed_patches'] < expected_patches * 0.95:
            self.warnings.append(f"Fewer patches than expected: {self.stats['preprocessed_patches']} (expected ~{expected_patches})")
    
    def verify_preprocessed_patch(self, filepath: Path):
        """Verify single preprocessed patch"""
        try:
            img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                self.errors.append(f"Failed to read: {filepath}")
                return
            
            # Check dimensions
            if img.shape != (256, 320):
                self.errors.append(f"Wrong dimensions {img.shape} (expected 256×320): {filepath.name}")
            
            # Check bit depth
            if img.dtype != np.uint8:
                self.errors.append(f"Wrong dtype {img.dtype} (expected uint8): {filepath.name}")
            
            # Check patch index in filename
            if '_patch' not in filepath.stem:
                self.warnings.append(f"Missing patch index in filename: {filepath.name}")
            else:
                patch_idx = filepath.stem.split('_patch')[-1]
                if patch_idx not in ['0', '1', '2', '3']:
                    self.errors.append(f"Invalid patch index '{patch_idx}': {filepath.name}")
            
            self.stats['preprocessed_patches'] += 1
            
        except Exception as e:
            self.errors.append(f"Error verifying {filepath}: {e}")
    
    def verify_homography_pairs(self):
        """Verify homography pair samples"""
        logger.info("Verifying homography pairs...")
        
        # Find all sample directories
        sample_dirs = [d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('sample_')]
        
        if not sample_dirs:
            self.errors.append("No sample directories found")
            return
        
        for sample_dir in tqdm(sample_dirs, desc="Verifying samples"):
            self.verify_homography_sample(sample_dir)
        
        logger.info(f"Total homography pairs found: {self.stats['homography_pairs']}")
    
    def verify_homography_sample(self, sample_dir: Path):
        """Verify single homography sample"""
        required_files = ['patch_A.tiff', 'patch_B.tiff', 'homography_matrix.npy', '4point_params.npy']
        
        # Check required files exist
        for filename in required_files:
            if not (sample_dir / filename).exists():
                self.errors.append(f"Missing {filename} in {sample_dir.name}")
                return
        
        try:
            # Verify patches
            patch_A = cv2.imread(str(sample_dir / 'patch_A.tiff'), cv2.IMREAD_GRAYSCALE)
            patch_B = cv2.imread(str(sample_dir / 'patch_B.tiff'), cv2.IMREAD_GRAYSCALE)
            
            if patch_A is None or patch_B is None:
                self.errors.append(f"Failed to read patches in {sample_dir.name}")
                return
            
            # Check dimensions
            if patch_A.shape != (256, 256) or patch_B.shape != (256, 256):
                self.errors.append(f"Wrong patch dimensions in {sample_dir.name}")
            
            # Check for black padding
            if np.any(patch_B == 0):
                self.warnings.append(f"patch_B contains zeros (possible padding) in {sample_dir.name}")
            
            # Verify homography matrix
            H = np.load(sample_dir / 'homography_matrix.npy')
            if H.shape != (3, 3):
                self.errors.append(f"Wrong homography shape {H.shape} in {sample_dir.name}")
            
            # Verify 4-point params
            params = np.load(sample_dir / '4point_params.npy')
            if params.shape != (8,):
                self.errors.append(f"Wrong 4point_params shape {params.shape} in {sample_dir.name}")
            
            # Check metadata if exists
            metadata_path = sample_dir / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    if 'sample_id' not in metadata:
                        self.warnings.append(f"Missing sample_id in metadata: {sample_dir.name}")
            
            self.stats['homography_pairs'] += 1
            
        except Exception as e:
            self.errors.append(f"Error verifying {sample_dir.name}: {e}")
    
    def verify(self):
        """Run verification based on dataset type"""
        if self.dataset_type == 'raw':
            self.verify_raw_dataset()
        elif self.dataset_type == 'preprocessed':
            self.verify_preprocessed_dataset()
        elif self.dataset_type == 'homography_pairs':
            self.verify_homography_pairs()
        else:
            self.errors.append(f"Unknown dataset type: {self.dataset_type}")
    
    def print_report(self):
        """Print verification report"""
        print("\n" + "="*60)
        print("VERIFICATION REPORT")
        print("="*60)
        
        print(f"\nDataset Type: {self.dataset_type}")
        print(f"Location: {self.data_dir}")
        
        print("\nStatistics:")
        for key, value in self.stats.items():
            print(f"  {key}: {value}")
        
        if self.warnings:
            print(f"\n⚠ Warnings ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings[:10], 1):
                print(f"  {i}. {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more")
        
        if self.errors:
            print(f"\n✗ Errors ({len(self.errors)}):")
            for i, error in enumerate(self.errors[:10], 1):
                print(f"  {i}. {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more")
        else:
            print("\n✓ No errors found!")
        
        print("\n" + "="*60)
        
        # Write detailed report to file
        report_path = self.data_dir / "verification_report.txt"
        with open(report_path, 'w') as f:
            f.write("HR-ThermalPV Dataset Verification Report\n")
            f.write("="*60 + "\n\n")
            f.write(f"Dataset Type: {self.dataset_type}\n")
            f.write(f"Location: {self.data_dir}\n\n")
            
            f.write("Statistics:\n")
            for key, value in self.stats.items():
                f.write(f"  {key}: {value}\n")
            
            if self.warnings:
                f.write(f"\nWarnings ({len(self.warnings)}):\n")
                for i, warning in enumerate(self.warnings, 1):
                    f.write(f"  {i}. {warning}\n")
            
            if self.errors:
                f.write(f"\nErrors ({len(self.errors)}):\n")
                for i, error in enumerate(self.errors, 1):
                    f.write(f"  {i}. {error}\n")
        
        print(f"Detailed report saved to: {report_path}")
        
        return len(self.errors) == 0


def main():
    parser = argparse.ArgumentParser(description="Verify HR-ThermalPV dataset integrity")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--type', type=str, 
                        choices=['raw', 'preprocessed', 'homography_pairs'],
                        help='Dataset type (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        logger.error(f"Directory does not exist: {data_dir}")
        return 1
    
    # Auto-detect type if not specified
    if args.type:
        dataset_type = args.type
    else:
        if any((data_dir / day).exists() for day in ['2024-12-21', '2024-12-22']):
            dataset_type = 'raw'
        elif list(data_dir.rglob("*_patch*.tiff")):
            dataset_type = 'preprocessed'
        elif list(data_dir.glob("sample_*")):
            dataset_type = 'homography_pairs'
        else:
            logger.error("Could not auto-detect dataset type. Please specify --type")
            return 1
        
        logger.info(f"Auto-detected dataset type: {dataset_type}")
    
    # Run verification
    verifier = DatasetVerifier(data_dir, dataset_type)
    verifier.verify()
    success = verifier.print_report()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())