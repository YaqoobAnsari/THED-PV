"""
Homography Pair Generation for HR-ThermalPV Dataset
Generates training pairs with controlled geometric transformations
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import logging
from typing import Tuple, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HomographyPairGenerator:
    """Generate homography pairs from thermal images"""
    
    def __init__(
        self,
        patch_size: int = 256,
        perturbation_range: Tuple[int, int] = (15, 50),
        overlap_range: Tuple[float, float] = (0.55, 0.85),
        center_bias: float = 0.2,
        max_attempts: int = 100
    ):
        """
        Args:
            patch_size: Size of patches to extract (default 256x256)
            perturbation_range: Min and max pixel perturbation for corners
            overlap_range: Min and max overlap ratio between patches
            center_bias: Bias towards center of image (0=uniform, 1=center only)
            max_attempts: Maximum attempts to generate valid pair
        """
        self.patch_size = patch_size
        self.perturbation_range = perturbation_range
        self.overlap_range = overlap_range
        self.center_bias = center_bias
        self.max_attempts = max_attempts
        
        logger.info(f"Initialized HomographyPairGenerator:")
        logger.info(f"  Patch size: {patch_size}x{patch_size}")
        logger.info(f"  Perturbation range: {perturbation_range}")
        logger.info(f"  Overlap range: {overlap_range}")
    
    def sample_patch_location(self, img_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Sample patch top-left corner with center bias"""
        h, w = img_shape
        
        # Available space for top-left corner
        max_y = h - self.patch_size
        max_x = w - self.patch_size
        
        if max_y <= 0 or max_x <= 0:
            raise ValueError(f"Image too small for patch size {self.patch_size}")
        
        # Center-biased sampling using normal distribution
        if self.center_bias > 0:
            center_y = max_y / 2
            center_x = max_x / 2
            std_y = max_y / 2 * (1 - self.center_bias)
            std_x = max_x / 2 * (1 - self.center_bias)
            
            y = int(np.clip(np.random.normal(center_y, std_y), 0, max_y))
            x = int(np.clip(np.random.normal(center_x, std_x), 0, max_x))
        else:
            y = np.random.randint(0, max_y + 1)
            x = np.random.randint(0, max_x + 1)
        
        return y, x
    
    def get_patch_corners(self, top_left: Tuple[int, int]) -> np.ndarray:
        """Get four corners of patch as [x, y] coordinates"""
        y, x = top_left
        corners = np.array([
            [x, y],                                    # Top-left
            [x + self.patch_size, y],                  # Top-right
            [x + self.patch_size, y + self.patch_size],# Bottom-right
            [x, y + self.patch_size]                   # Bottom-left
        ], dtype=np.float32)
        return corners
    
    def perturb_corners(
        self,
        corners: np.ndarray,
        img_shape: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """Randomly perturb corner points within valid image bounds"""
        h, w = img_shape
        rho_min, rho_max = self.perturbation_range
        
        perturbed = corners.copy()
        
        for i in range(4):
            # Random perturbation in x and y
            delta_x = np.random.randint(-rho_max, rho_max + 1)
            delta_y = np.random.randint(-rho_max, rho_max + 1)
            
            # Ensure perturbation is at least rho_min
            if abs(delta_x) < rho_min:
                delta_x = rho_min if delta_x >= 0 else -rho_min
            if abs(delta_y) < rho_min:
                delta_y = rho_min if delta_y >= 0 else -rho_min
            
            perturbed[i, 0] += delta_x
            perturbed[i, 1] += delta_y
            
            # Check bounds
            if not (0 <= perturbed[i, 0] < w and 0 <= perturbed[i, 1] < h):
                return None  # Out of bounds
        
        return perturbed
    
    def slide_patch(
        self,
        top_left: Tuple[int, int],
        img_shape: Tuple[int, int],
        direction: Optional[str] = None
    ) -> Optional[Tuple[int, int]]:
        """Slide patch in a direction maintaining overlap"""
        h, w = img_shape
        y, x = top_left
        
        # Target overlap ratio
        target_overlap = np.random.uniform(*self.overlap_range)
        slide_distance = int(self.patch_size * (1 - target_overlap))
        
        # Random direction if not specified
        if direction is None:
            direction = np.random.choice(['up', 'down', 'left', 'right', 
                                         'up-left', 'up-right', 
                                         'down-left', 'down-right'])
        
        # Apply slide
        new_y, new_x = y, x
        if 'up' in direction:
            new_y -= slide_distance
        if 'down' in direction:
            new_y += slide_distance
        if 'left' in direction:
            new_x -= slide_distance
        if 'right' in direction:
            new_x += slide_distance
        
        # Check bounds
        if (new_y < 0 or new_y + self.patch_size > h or
            new_x < 0 or new_x + self.patch_size > w):
            return None
        
        return new_y, new_x
    
    def compute_homography(
        self,
        corners1: np.ndarray,
        corners2: np.ndarray
    ) -> np.ndarray:
        """Compute homography matrix from corner correspondences"""
        H, _ = cv2.findHomography(corners1, corners2, 0)
        if H is None:
            raise ValueError("Failed to compute homography")
        return H
    
    def compute_4point_params(
        self,
        corners1: np.ndarray,
        corners2: np.ndarray
    ) -> np.ndarray:
        """Compute 4-point parameterization (displacement vector)"""
        # This is what HomographyNet uses
        delta = corners2 - corners1
        return delta.flatten()  # Shape: (8,)
    
    def warp_image(
        self,
        img: np.ndarray,
        H: np.ndarray
    ) -> np.ndarray:
        """Apply homography transformation to image"""
        h, w = img.shape[:2]
        warped = cv2.warpPerspective(img, H, (w, h))
        return warped
    
    def validate_pair(
        self,
        img: np.ndarray,
        patch1_tl: Tuple[int, int],
        patch2_tl: Tuple[int, int],
        corners2: np.ndarray
    ) -> bool:
        """Validate that warped patch has no black padding"""
        y, x = patch2_tl
        
        # Extract patch region
        patch = img[y:y+self.patch_size, x:x+self.patch_size]
        
        # Check for black padding (zeros)
        if patch.shape != (self.patch_size, self.patch_size):
            return False
        
        # Check if any pixel is exactly zero (padding)
        if np.any(patch == 0):
            return False
        
        # Check corner points are within patch bounds
        for corner in corners2:
            cx, cy = corner
            if not (x <= cx < x + self.patch_size and y <= cy < y + self.patch_size):
                return False
        
        return True
    
    def generate_pair(
        self,
        img: np.ndarray
    ) -> Optional[dict]:
        """Generate a single valid homography pair"""
        
        for attempt in range(self.max_attempts):
            try:
                # 1. Sample initial patch location
                patch1_tl = self.sample_patch_location(img.shape)
                corners1 = self.get_patch_corners(patch1_tl)
                
                # 2. Slide to get second patch with overlap
                patch2_tl = self.slide_patch(patch1_tl, img.shape)
                if patch2_tl is None:
                    continue
                
                # 3. Perturb corners
                corners2 = self.get_patch_corners(patch2_tl)
                corners2_perturbed = self.perturb_corners(corners2, img.shape)
                if corners2_perturbed is None:
                    continue
                
                # 4. Compute homography
                H = self.compute_homography(corners1, corners2_perturbed)
                H_inv = np.linalg.inv(H)
                
                # 5. Warp image
                warped_img = self.warp_image(img, H_inv)
                
                # 6. Validate no black padding
                if not self.validate_pair(warped_img, patch1_tl, patch2_tl, corners2_perturbed):
                    continue
                
                # 7. Extract patches
                y1, x1 = patch1_tl
                y2, x2 = patch2_tl
                
                patch_A = img[y1:y1+self.patch_size, x1:x1+self.patch_size]
                patch_B = warped_img[y2:y2+self.patch_size, x2:x2+self.patch_size]
                
                # 8. Compute 4-point parameterization
                delta_4pt = self.compute_4point_params(corners1, corners2_perturbed)
                
                return {
                    'patch_A': patch_A,
                    'patch_B': patch_B,
                    'homography_matrix': H,
                    '4point_params': delta_4pt,
                    'corners1': corners1,
                    'corners2': corners2_perturbed,
                    'patch1_tl': patch1_tl,
                    'patch2_tl': patch2_tl,
                    'attempts': attempt + 1
                }
                
            except Exception as e:
                logger.debug(f"Attempt {attempt+1} failed: {e}")
                continue
        
        logger.warning(f"Failed to generate valid pair after {self.max_attempts} attempts")
        return None
    
    def save_pair(
        self,
        pair_data: dict,
        output_dir: Path,
        sample_id: int
    ):
        """Save homography pair to disk"""
        sample_dir = output_dir / f"sample_{sample_id:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Save patches
        cv2.imwrite(str(sample_dir / "patch_A.tiff"), pair_data['patch_A'])
        cv2.imwrite(str(sample_dir / "patch_B.tiff"), pair_data['patch_B'])
        
        # Save homography matrix
        np.save(sample_dir / "homography_matrix.npy", pair_data['homography_matrix'])
        
        # Save 4-point params (for HomographyNet)
        np.save(sample_dir / "4point_params.npy", pair_data['4point_params'])
        
        # Save metadata
        metadata = {
            'sample_id': sample_id,
            'patch1_topleft': pair_data['patch1_tl'],
            'patch2_topleft': pair_data['patch2_tl'],
            'corners1': pair_data['corners1'].tolist(),
            'corners2': pair_data['corners2'].tolist(),
            'attempts': pair_data['attempts']
        }
        
        with open(sample_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    num_pairs_per_image: int = 2,
    **generator_kwargs
):
    """Generate homography pairs for entire dataset"""
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = HomographyPairGenerator(**generator_kwargs)
    
    # Find all preprocessed images
    image_files = sorted(list(input_dir.rglob("*.tiff")) + list(input_dir.rglob("*.tif")))
    
    if not image_files:
        raise ValueError(f"No images found in {input_dir}")
    
    logger.info(f"Found {len(image_files)} images")
    logger.info(f"Target: {num_pairs_per_image} pairs per image")
    logger.info(f"Total pairs to generate: {len(image_files) * num_pairs_per_image}")
    
    sample_counter = 0
    failed_images = []
    
    for img_path in tqdm(image_files, desc="Generating homography pairs"):
        # Read image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error(f"Failed to read {img_path}")
            failed_images.append(str(img_path))
            continue
        
        # Generate pairs for this image
        pairs_generated = 0
        for _ in range(num_pairs_per_image):
            pair_data = generator.generate_pair(img)
            
            if pair_data is not None:
                generator.save_pair(pair_data, output_dir, sample_counter)
                sample_counter += 1
                pairs_generated += 1
            else:
                logger.warning(f"Could not generate pair for {img_path.name}")
        
        if pairs_generated == 0:
            failed_images.append(str(img_path))
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Generation complete!")
    logger.info(f"Total pairs generated: {sample_counter}")
    logger.info(f"Images processed: {len(image_files) - len(failed_images)}/{len(image_files)}")
    if failed_images:
        logger.warning(f"Failed images ({len(failed_images)}): See failed_images.txt")
        with open(output_dir / "failed_images.txt", 'w') as f:
            f.write('\n'.join(failed_images))
    logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate homography pairs for HR-ThermalPV dataset"
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory with preprocessed images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for homography pairs')
    parser.add_argument('--num_pairs', type=int, default=2,
                        help='Number of pairs per image (default: 2)')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Patch size (default: 256)')
    parser.add_argument('--perturbation_range', type=int, nargs=2, default=[15, 50],
                        help='Min and max perturbation in pixels (default: 15 50)')
    parser.add_argument('--overlap_range', type=float, nargs=2, default=[0.55, 0.85],
                        help='Min and max overlap ratio (default: 0.55 0.85)')
    parser.add_argument('--center_bias', type=float, default=0.2,
                        help='Center bias factor 0-1 (default: 0.2)')
    
    args = parser.parse_args()
    
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_pairs_per_image=args.num_pairs,
        patch_size=args.patch_size,
        perturbation_range=tuple(args.perturbation_range),
        overlap_range=tuple(args.overlap_range),
        center_bias=args.center_bias
    )


if __name__ == "__main__":
    main()