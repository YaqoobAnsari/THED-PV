"""
Data validation utilities for HR-ThermalPV dataset
Validates images, homography pairs, and dataset structure
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import json
import logging

logger = logging.getLogger(__name__)


class ImageValidator:
    """Validate thermal images at different processing stages"""
    
    @staticmethod
    def validate_raw_image(img_path: Path) -> Dict:
        """
        Validate raw 14-bit thermal TIFF image
        
        Returns dict with 'valid' boolean and 'errors' list
        """
        result = {'valid': True, 'errors': []}
        
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            
            if img is None:
                result['errors'].append(f"Cannot read image: {img_path.name}")
                result['valid'] = False
                return result
            
            # Check dimensions (640×512)
            if img.shape != (512, 640):
                result['errors'].append(f"Wrong dimensions {img.shape}, expected (512, 640)")
                result['valid'] = False
            
            # Check dtype (uint16 for 14-bit)
            if img.dtype != np.uint16:
                result['errors'].append(f"Wrong dtype {img.dtype}, expected uint16")
                result['valid'] = False
            
            # Check value range (14-bit: 0-16383)
            if img.max() > 16383:
                result['errors'].append(f"Values exceed 14-bit range: {img.max()}")
                result['valid'] = False
            
            # Check not blank
            if img.max() == 0:
                result['errors'].append("Image is blank (all zeros)")
                result['valid'] = False
            
        except Exception as e:
            result['errors'].append(f"Validation exception: {str(e)}")
            result['valid'] = False
        
        return result
    
    @staticmethod
    def validate_preprocessed_patch(patch_path: Path) -> Dict:
        """
        Validate preprocessed 8-bit patch (320×256)
        
        Returns dict with 'valid' boolean and 'errors' list
        """
        result = {'valid': True, 'errors': []}
        
        try:
            img = cv2.imread(str(patch_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                result['errors'].append(f"Cannot read patch: {patch_path.name}")
                result['valid'] = False
                return result
            
            # Check dimensions (320×256)
            if img.shape != (256, 320):
                result['errors'].append(f"Wrong dimensions {img.shape}, expected (256, 320)")
                result['valid'] = False
            
            # Check dtype (uint8)
            if img.dtype != np.uint8:
                result['errors'].append(f"Wrong dtype {img.dtype}, expected uint8")
                result['valid'] = False
            
            # Check not blank
            if img.max() == 0:
                result['errors'].append("Patch is blank")
                result['valid'] = False
            
            # Check has contrast
            if img.std() < 5:
                result['errors'].append(f"Very low contrast (std={img.std():.2f})")
                result['valid'] = False
            
        except Exception as e:
            result['errors'].append(f"Validation exception: {str(e)}")
            result['valid'] = False
        
        return result


class HomographyValidator:
    """Validate homography pair samples"""
    
    @staticmethod
    def validate_sample(sample_dir: Path) -> Dict:
        """
        Validate complete homography sample directory
        
        Returns dict with 'valid' boolean and 'errors' list
        """
        result = {'valid': True, 'errors': []}
        
        # Check required files exist
        required = ['patch_A.tiff', 'patch_B.tiff', 'homography_matrix.npy', '4point_params.npy']
        for filename in required:
            if not (sample_dir / filename).exists():
                result['errors'].append(f"Missing file: {filename}")
                result['valid'] = False
        
        if not result['valid']:
            return result
        
        try:
            # Load components
            patch_A = cv2.imread(str(sample_dir / 'patch_A.tiff'), cv2.IMREAD_GRAYSCALE)
            patch_B = cv2.imread(str(sample_dir / 'patch_B.tiff'), cv2.IMREAD_GRAYSCALE)
            H = np.load(sample_dir / 'homography_matrix.npy')
            params = np.load(sample_dir / '4point_params.npy')
            
            # Validate patches
            if patch_A is None or patch_B is None:
                result['errors'].append("Cannot read patches")
                result['valid'] = False
                return result
            
            # Check patch dimensions (256×256)
            if patch_A.shape != (256, 256):
                result['errors'].append(f"patch_A wrong size: {patch_A.shape}")
                result['valid'] = False
            
            if patch_B.shape != (256, 256):
                result['errors'].append(f"patch_B wrong size: {patch_B.shape}")
                result['valid'] = False
            
            # Check no black padding in patch_B
            if np.any(patch_B == 0):
                black_count = np.sum(patch_B == 0)
                result['errors'].append(f"patch_B has {black_count} black pixels (padding)")
                result['valid'] = False
            
            # Check homography matrix
            if H.shape != (3, 3):
                result['errors'].append(f"H wrong shape: {H.shape}")
                result['valid'] = False
            
            # Check H is invertible
            try:
                H_inv = np.linalg.inv(H)
                if np.any(np.isnan(H_inv)) or np.any(np.isinf(H_inv)):
                    result['errors'].append("H inverse contains NaN/Inf")
                    result['valid'] = False
            except np.linalg.LinAlgError:
                result['errors'].append("H is not invertible")
                result['valid'] = False
            
            # Check 4-point params
            if params.shape != (8,):
                result['errors'].append(f"4point_params wrong shape: {params.shape}")
                result['valid'] = False
            
        except Exception as e:
            result['errors'].append(f"Validation exception: {str(e)}")
            result['valid'] = False
        
        return result


class DatasetValidator:
    """Validate entire dataset structure and contents"""
    
    @staticmethod
    def validate_raw_dataset(data_dir: Path) -> Dict:
        """
        Validate raw dataset directory structure
        
        Expected structure:
        data_dir/
        ├── 2024-12-21/ ... 2024-12-25/
        │   └── {8am,10am,12pm,2pm}/
        │       └── {10cm,20cm,30cm,40cm}/
        │           └── *.tiff
        """
        result = {
            'valid': True,
            'total_images': 0,
            'valid_images': 0,
            'missing_dirs': [],
            'invalid_images': [],
            'errors': []
        }
        
        expected_days = ['2024-12-21', '2024-12-22', '2024-12-23', '2024-12-24', '2024-12-25']
        expected_times = ['8am', '10am', '12pm', '2pm']
        expected_heights = ['10cm', '20cm', '30cm', '40cm']
        
        # Check directory structure
        for day in expected_days:
            if not (data_dir / day).exists():
                result['missing_dirs'].append(day)
                continue
            
            for time in expected_times:
                if not (data_dir / day / time).exists():
                    result['missing_dirs'].append(f"{day}/{time}")
                    continue
                
                for height in expected_heights:
                    height_path = data_dir / day / time / height
                    if not height_path.exists():
                        result['missing_dirs'].append(f"{day}/{time}/{height}")
                        continue
                    
                    # Count and validate images
                    tiffs = list(height_path.glob("*.tiff")) + list(height_path.glob("*.tif"))
                    result['total_images'] += len(tiffs)
        
        if result['missing_dirs']:
            result['errors'].append(f"{len(result['missing_dirs'])} missing directories")
            result['valid'] = False
        
        return result
    
    @staticmethod
    def validate_preprocessed_dataset(data_dir: Path, expected_count: Optional[int] = None) -> Dict:
        """Validate preprocessed dataset (patches)"""
        result = {
            'valid': True,
            'total_patches': 0,
            'valid_patches': 0,
            'invalid_patches': [],
            'errors': []
        }
        
        patches = list(data_dir.rglob("*_patch*.tiff"))
        result['total_patches'] = len(patches)
        
        if expected_count and len(patches) < expected_count * 0.95:
            result['errors'].append(f"Too few patches: {len(patches)} (expected ~{expected_count})")
            result['valid'] = False
        
        return result
    
    @staticmethod
    def validate_homography_dataset(data_dir: Path, expected_count: Optional[int] = None) -> Dict:
        """Validate homography pairs dataset"""
        result = {
            'valid': True,
            'total_samples': 0,
            'valid_samples': 0,
            'incomplete_samples': [],
            'errors': []
        }
        
        sample_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('sample_')]
        result['total_samples'] = len(sample_dirs)
        
        required_files = ['patch_A.tiff', 'patch_B.tiff', 'homography_matrix.npy', '4point_params.npy']
        
        for sample_dir in sample_dirs:
            missing = [f for f in required_files if not (sample_dir / f).exists()]
            if missing:
                result['incomplete_samples'].append({
                    'sample': sample_dir.name,
                    'missing': missing
                })
        
        result['valid_samples'] = len(sample_dirs) - len(result['incomplete_samples'])
        
        if result['incomplete_samples']:
            result['errors'].append(f"{len(result['incomplete_samples'])} incomplete samples")
            result['valid'] = False
        
        if expected_count and len(sample_dirs) < expected_count * 0.95:
            result['errors'].append(f"Too few samples: {len(sample_dirs)} (expected ~{expected_count})")
            result['valid'] = False
        
        return result


def compute_image_stats(img: np.ndarray) -> Dict:
    """Compute basic statistics for an image"""
    return {
        'shape': img.shape,
        'dtype': str(img.dtype),
        'min': float(img.min()),
        'max': float(img.max()),
        'mean': float(img.mean()),
        'std': float(img.std())
    }


def validate_preprocessing_output(raw_path: Path, processed_path: Path) -> Dict:
    """
    Compare raw and preprocessed images to ensure quality improvement
    
    Returns metrics showing preprocessing effectiveness
    """
    result = {'valid': True, 'errors': []}
    
    try:
        raw = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)
        processed = cv2.imread(str(processed_path), cv2.IMREAD_GRAYSCALE)
        
        if raw is None or processed is None:
            result['errors'].append("Cannot read images")
            result['valid'] = False
            return result
        
        # Normalize raw to 8-bit for comparison
        raw_norm = ((raw - raw.min()) / (raw.max() - raw.min()) * 255).astype(np.uint8)
        
        # Compute entropy (should increase after preprocessing)
        def entropy(img):
            hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
            hist = hist[hist > 0] / hist.sum()
            return -np.sum(hist * np.log2(hist))
        
        entropy_raw = entropy(raw_norm)
        entropy_processed = entropy(processed)
        
        result['entropy_raw'] = entropy_raw
        result['entropy_processed'] = entropy_processed
        result['entropy_gain'] = entropy_processed - entropy_raw
        
        # Check entropy increased (indicates better feature distribution)
        if entropy_processed <= entropy_raw:
            result['errors'].append("Preprocessing did not increase entropy")
            result['valid'] = False
        
    except Exception as e:
        result['errors'].append(f"Validation exception: {str(e)}")
        result['valid'] = False
    
    return result


# Export main classes
__all__ = [
    'ImageValidator',
    'HomographyValidator', 
    'DatasetValidator',
    'compute_image_stats',
    'validate_preprocessing_output'
]