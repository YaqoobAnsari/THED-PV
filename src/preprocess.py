"""
Thermal Image Preprocessing Pipeline for HR-ThermalPV Dataset

Implements the exact preprocessing methodology from the paper:
- Paper Section: "Image Preprocessing" (Lines 114-143, Page 5-6)
- Pipeline: Normalize → Glare Suppression → CLAHE → Bilateral → Shadow Correction → Sharpening
- Output: 320×256 patches from 640×512 raw images
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm
import logging
from typing import Tuple, Optional, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ThermalPreprocessor:
    """Preprocessing pipeline following paper specifications (Lines 122-143, Page 6)"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize preprocessor with config file or paper defaults.
        
        Args:
            config_path: Path to YAML config file. Uses paper defaults if None.
        """
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded config from: {config_path}")
        else:
            self.config = self._get_paper_defaults()
            logger.info("Using paper default parameters")
        
        self._log_config()
    
    def _get_paper_defaults(self) -> dict:
        """Default parameters from paper (Lines 123-142, Page 6)"""
        return {
            'clahe': {
                'clip_limit': 0.018,       # Paper Line 123
                'tile_grid_size': [8, 8]   # Paper Line 124
            },
            'bilateral': {
                'd': 9,                     # Paper Line 128
                'sigma_color': 0.015,       # Paper Line 129 (scaled internally)
                'sigma_space': 8            # Paper Line 128
            },
            'glare': {
                'threshold_factor': 2.0     # Paper Line 136: "mean + 2× standard deviation"
            },
            'shadow': {
                'enabled': True,
                'kernel_divisor': 20        # Kernel size = image_size / 20
            },
            'sharpening': {
                'enabled': True,
                'sigma': 2.0,
                'alpha': 1.5,               # Weight for original
                'beta': -0.5                # Weight for blurred (negative = subtract)
            },
            'output': {
                'patch_size': [320, 256]    # Quarter from 640×512
            }
        }
    
    def _log_config(self):
        """Log configuration for reproducibility"""
        logger.info("Preprocessing Configuration:")
        logger.info(f"  CLAHE clip_limit: {self.config['clahe']['clip_limit']}")
        logger.info(f"  CLAHE tile_grid: {self.config['clahe']['tile_grid_size']}")
        logger.info(f"  Bilateral d: {self.config['bilateral']['d']}")
        logger.info(f"  Bilateral σ_color: {self.config['bilateral']['sigma_color']}")
        logger.info(f"  Bilateral σ_space: {self.config['bilateral']['sigma_space']}")
        logger.info(f"  Glare threshold: mean + {self.config['glare']['threshold_factor']}σ")
    
    def read_thermal_image(self, path: Path) -> np.ndarray:
        """
        Read 14-bit thermal TIFF image.
        
        Args:
            path: Path to TIFF file
            
        Returns:
            Image array (dtype: uint16 for 14-bit)
            
        Raises:
            ValueError: If image cannot be read
        """
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        return img
    
    def normalize_to_8bit(self, img: np.ndarray) -> np.ndarray:
        """
        Step 1: Normalize 14-bit to 8-bit (Paper Lines 122-126, Page 6)
        
        Converts 14-bit thermal data (0-16383) to 8-bit (0-255).
        """
        img_min, img_max = img.min(), img.max()
        
        if img_max <= img_min:
            logger.warning("Image has uniform intensity, returning zeros")
            return np.zeros_like(img, dtype=np.uint8)
        
        img_norm = ((img.astype(np.float32) - img_min) / (img_max - img_min) * 255.0)
        return img_norm.astype(np.uint8)
    
    def suppress_glare(self, img: np.ndarray) -> np.ndarray:
        """
        Step 2: Adaptive glare suppression (Paper Lines 134-136, Page 6)
        
        Quote: "Glare suppression was achieved by clamping pixel intensities
        exceeding mean + 2× standard deviation to the glare threshold"
        """
        mean = np.mean(img)
        std = np.std(img)
        threshold = mean + (self.config['glare']['threshold_factor'] * std)
        
        # Clamp and renormalize
        img_clamped = np.clip(img, 0, threshold)
        
        # Renormalize to full 0-255 range
        if img_clamped.max() > 0:
            img_norm = (img_clamped / img_clamped.max() * 255.0).astype(np.uint8)
        else:
            img_norm = img_clamped.astype(np.uint8)
        
        return img_norm
    
    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """
        Step 3: CLAHE enhancement (Paper Lines 122-126, Page 6)
        
        Quote: "CLAHE implementation used a clip limit of 0.018 and operated
        on small contextual regions"
        """
        clahe = cv2.createCLAHE(
            clipLimit=self.config['clahe']['clip_limit'],
            tileGridSize=tuple(self.config['clahe']['tile_grid_size'])
        )
        return clahe.apply(img)
    
    def apply_bilateral_filter(self, img: np.ndarray) -> np.ndarray:
        """
        Step 4: Bilateral filtering (Paper Lines 127-131, Page 6)
        
        Quote: "Bilateral filter configured with spatial smoothing parameter
        σ_spatial = 8 and radiometric smoothing parameter σ_color = 0.015"
        """
        # σ_color in config is 0-1 range, scale to 0-255 for OpenCV
        sigma_color = self.config['bilateral']['sigma_color'] * 255
        
        return cv2.bilateralFilter(
            img,
            d=self.config['bilateral']['d'],
            sigmaColor=sigma_color,
            sigmaSpace=self.config['bilateral']['sigma_space']
        )
    
    def correct_shadows(self, img: np.ndarray) -> np.ndarray:
        """
        Step 5: Shadow correction (Paper Lines 136-138, Page 6)
        
        Quote: "Shadowed regions...were corrected via localized intensity
        normalization to restore thermal consistency"
        
        Uses morphological opening to estimate background.
        """
        if not self.config['shadow']['enabled']:
            return img
        
        # Convert to float [0, 1]
        img_float = img.astype(np.float32) / 255.0
        
        # Adaptive kernel size based on image dimensions
        kernel_size = max(img.shape) // self.config['shadow']['kernel_divisor']
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size  # Ensure odd
        
        # Estimate background via morphological opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        background = cv2.morphologyEx(img_float, cv2.MORPH_OPEN, kernel)
        
        # Normalize by background
        corrected = img_float / (background + 1e-6)
        corrected = np.clip(corrected, 0, 1)
        
        return (corrected * 255).astype(np.uint8)
    
    def sharpen_image(self, img: np.ndarray) -> np.ndarray:
        """
        Step 6: Sharpening via unsharp mask (Paper Lines 139-143, Page 6)
        
        Quote: "Feature-space optimization through...keypoint enhancement strategy"
        
        Final preprocessing step.
        """
        if not self.config['sharpening']['enabled']:
            return img
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(img, (0, 0), self.config['sharpening']['sigma'])
        
        # Unsharp mask: sharp = α×original - β×blurred
        sharpened = cv2.addWeighted(
            img, self.config['sharpening']['alpha'],
            blurred, self.config['sharpening']['beta'],
            0
        )
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Full 6-step preprocessing pipeline (Paper Figure 5, Page 5)
        
        Pipeline order:
        1. Normalize to 8-bit
        2. Glare suppression  
        3. CLAHE enhancement
        4. Bilateral filtering
        5. Shadow correction
        6. Sharpening
        
        Args:
            img: Raw thermal image (14-bit or 8-bit)
            
        Returns:
            Preprocessed 8-bit image
        """
        # Step 1: Normalize
        img = self.normalize_to_8bit(img)
        
        # Step 2: Glare suppression
        img = self.suppress_glare(img)
        
        # Step 3: CLAHE
        img = self.apply_clahe(img)
        
        # Step 4: Bilateral filter
        img = self.apply_bilateral_filter(img)
        
        # Step 5: Shadow correction
        img = self.correct_shadows(img)
        
        # Step 6: Sharpening
        img = self.sharpen_image(img)
        
        return img
    
    def quarter_image(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Divide 640×512 image into four 320×256 patches (Paper Line 183, Page 7)
        
        Quote: "Images were patched from their original 640x512 resolution
        to a more manageable 320x256 resolution"
        
        Returns patches in order: [top-left, top-right, bottom-left, bottom-right]
        """
        h, w = img.shape[:2]
        
        if (h, w) != (512, 640):
            logger.warning(f"Expected 512×640, got {h}×{w}. Attempting to quarter anyway.")
        
        patches = []
        for i in range(2):
            for j in range(2):
                patch = img[i*256:(i+1)*256, j*320:(j+1)*320]
                
                # Validate patch size
                if patch.shape != (256, 320):
                    logger.error(f"Patch {len(patches)} has wrong shape: {patch.shape}")
                    # Pad if needed
                    padded = np.zeros((256, 320), dtype=img.dtype)
                    padded[:patch.shape[0], :patch.shape[1]] = patch
                    patch = padded
                
                patches.append(patch)
        
        return patches
    
    def process_file(self, input_path: Path, output_dir: Path) -> List[Path]:
        """
        Process single thermal image: preprocess + quarter + save
        
        Args:
            input_path: Path to input TIFF
            output_dir: Directory to save patches
            
        Returns:
            List of saved patch paths
        """
        # Read
        img = self.read_thermal_image(input_path)
        
        # Preprocess
        img_processed = self.preprocess_image(img)
        
        # Quarter
        patches = self.quarter_image(img_processed)
        
        # Save
        saved_paths = []
        for idx, patch in enumerate(patches):
            output_name = f"{input_path.stem}_patch{idx}.tiff"
            output_path = output_dir / output_name
            
            if cv2.imwrite(str(output_path), patch):
                saved_paths.append(output_path)
            else:
                logger.error(f"Failed to save: {output_path}")
        
        return saved_paths


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    config_path: Optional[str] = None
):
    """
    Process entire dataset with progress tracking
    
    Args:
        input_dir: Root directory of raw images
        output_dir: Root directory for preprocessed output
        config_path: Optional config file path
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        raise ValueError(f"Input directory not found: {input_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize
    preprocessor = ThermalPreprocessor(config_path)
    
    # Find all TIFF files
    tiff_files = sorted(list(input_dir.rglob("*.tiff")) + list(input_dir.rglob("*.tif")))
    
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {input_dir}")
    
    logger.info(f"\nFound {len(tiff_files)} thermal images")
    logger.info(f"Expected output: {len(tiff_files) * 4} patches\n")
    
    # Process
    total_patches = 0
    failed = []
    
    for tiff_path in tqdm(tiff_files, desc="Preprocessing"):
        # Maintain directory structure
        rel_path = tiff_path.relative_to(input_dir)
        output_subdir = output_dir / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        try:
            saved = preprocessor.process_file(tiff_path, output_subdir)
            total_patches += len(saved)
        except Exception as e:
            logger.error(f"Failed {tiff_path.name}: {e}")
            failed.append(str(tiff_path))
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Preprocessing Complete")
    logger.info(f"{'='*60}")
    logger.info(f"  Processed: {len(tiff_files) - len(failed)}/{len(tiff_files)} images")
    logger.info(f"  Output patches: {total_patches}")
    logger.info(f"  Output directory: {output_dir}")
    
    if failed:
        logger.warning(f"  Failed: {len(failed)} images")
        fail_log = output_dir / "failed_files.txt"
        fail_log.write_text('\n'.join(failed))
        logger.info(f"  Failed files logged: {fail_log}")
    
    logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="HR-ThermalPV Image Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline (Paper Figure 5, Page 5):
  1. Normalize 14-bit → 8-bit
  2. Glare suppression (mean + 2σ threshold)
  3. CLAHE (clip=0.018, tiles=8×8)
  4. Bilateral filter (d=9, σ_color=0.015, σ_space=8)
  5. Shadow correction (morphological)
  6. Sharpening (unsharp mask)
  7. Quarter to 320×256 patches

Example:
  python src/preprocess.py \\
      --input_dir ./data/raw \\
      --output_dir ./data/preprocessed
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory with raw thermal images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for preprocessed patches')
    parser.add_argument('--config', type=str, default=None,
                        help='Config YAML file (optional, uses paper defaults)')
    
    args = parser.parse_args()
    
    process_dataset(args.input_dir, args.output_dir, args.config)


if __name__ == "__main__":
    main()