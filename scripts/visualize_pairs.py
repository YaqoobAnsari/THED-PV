#!/usr/bin/env python3
"""
visualize_pairs.py - Diagnostic visualization for homography pairs and feature correspondences

This script generates comprehensive visualizations for homography estimation validation:
- Side-by-side image pair visualization
- Feature keypoint overlays
- Feature correspondence lines
- Homography warping results
- Inlier/outlier distinction
- Grid overlay for geometric distortion analysis

Author: THED-PV Team
Date: January 2025
Python: 3.8+
Dependencies: opencv-python, numpy, matplotlib
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, ConnectionPatch

# Use non-interactive backend for batch processing
matplotlib.use('Agg')

# Suppress warnings
warnings.filterwarnings('ignore')


class HomographyVisualizer:
    """
    Creates diagnostic visualizations for homography pairs and feature matching.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (16, 8), dpi: int = 150):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Figure size in inches (width, height)
            dpi: Resolution in dots per inch
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # Color schemes
        self.colors = {
            'inlier': (0, 255, 0),      # Green
            'outlier': (255, 0, 0),      # Red
            'keypoint': (255, 255, 0),   # Yellow
            'match_line': (0, 255, 255), # Cyan
            'grid': (255, 128, 0)        # Orange
        }
    
    def load_image_pair(self, img1_path: str, img2_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load an image pair.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
        
        Returns:
            img1, img2: Loaded images (BGR format)
        """
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))
        
        if img1 is None or img2 is None:
            raise ValueError(f"Failed to load images: {img1_path}, {img2_path}")
        
        return img1, img2
    
    def load_homography_data(self, pair_dir: str) -> Dict:
        """
        Load homography pair data including ground truth matrix and metadata.
        
        Args:
            pair_dir: Directory containing pair data
        
        Returns:
            dict: Pair data including images, homography matrix, metadata
        """
        pair_dir = Path(pair_dir)
        
        data = {}
        
        # Load images
        img_a_path = pair_dir / "patch_A.tiff"
        img_b_path = pair_dir / "patch_B.tiff"
        
        if not img_a_path.exists() or not img_b_path.exists():
            raise FileNotFoundError(f"Images not found in {pair_dir}")
        
        data['img1'], data['img2'] = self.load_image_pair(img_a_path, img_b_path)
        data['img1_path'] = str(img_a_path)
        data['img2_path'] = str(img_b_path)
        
        # Load homography matrix
        H_path = pair_dir / "H_matrix.npy"
        if H_path.exists():
            data['H_gt'] = np.load(H_path)
        
        # Load metadata
        metadata_path = pair_dir / "metadata.txt"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = {}
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        metadata[key.strip()] = value.strip()
                data['metadata'] = metadata
        
        return data
    
    def visualize_side_by_side(self, img1: np.ndarray, img2: np.ndarray,
                               title1: str = "Image A",
                               title2: str = "Image B",
                               output_path: Optional[str] = None):
        """
        Create side-by-side visualization of image pair.
        
        Args:
            img1: First image
            img2: Second image
            title1: Title for first image
            title2: Title for second image
            output_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)
        
        # Convert BGR to RGB for matplotlib
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if len(img1.shape) == 3 else img1
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if len(img2.shape) == 3 else img2
        
        axes[0].imshow(img1_rgb, cmap='gray' if len(img1.shape) == 2 else None)
        axes[0].set_title(title1, fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(img2_rgb, cmap='gray' if len(img2.shape) == 2 else None)
        axes[1].set_title(title2, fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=self.dpi)
            print(f"Saved: {output_path}")
        
        plt.close()
    
    def visualize_keypoints(self, img1: np.ndarray, img2: np.ndarray,
                           kp1: List, kp2: List,
                           output_path: Optional[str] = None):
        """
        Visualize detected keypoints on both images.
        
        Args:
            img1: First image
            img2: Second image
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            output_path: Path to save figure
        """
        # Draw keypoints
        img1_kp = cv2.drawKeypoints(
            img1, kp1, None,
            color=self.colors['keypoint'],
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        img2_kp = cv2.drawKeypoints(
            img2, kp2, None,
            color=self.colors['keypoint'],
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)
        
        # Convert BGR to RGB
        img1_rgb = cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB)
        
        axes[0].imshow(img1_rgb)
        axes[0].set_title(f'Image A: {len(kp1)} keypoints', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(img2_rgb)
        axes[1].set_title(f'Image B: {len(kp2)} keypoints', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=self.dpi)
            print(f"Saved: {output_path}")
        
        plt.close()
    
    def visualize_matches(self, img1: np.ndarray, img2: np.ndarray,
                         kp1: List, kp2: List, matches: List,
                         inlier_mask: Optional[np.ndarray] = None,
                         max_matches: int = 100,
                         output_path: Optional[str] = None):
        """
        Visualize feature matches between image pair.
        
        Args:
            img1: First image
            img2: Second image
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: List of matches
            inlier_mask: Binary mask indicating inliers (optional)
            max_matches: Maximum number of matches to display
            output_path: Path to save figure
        """
        # Limit number of matches for visualization
        if len(matches) > max_matches:
            # Sample matches uniformly
            indices = np.linspace(0, len(matches)-1, max_matches, dtype=int)
            matches_to_draw = [matches[i] for i in indices]
            if inlier_mask is not None:
                inlier_mask = inlier_mask[indices]
        else:
            matches_to_draw = matches
        
        # Draw matches
        if inlier_mask is not None and len(inlier_mask) > 0:
            # Draw with inlier/outlier distinction
            match_img = cv2.drawMatches(
                img1, kp1, img2, kp2, matches_to_draw, None,
                matchColor=self.colors['inlier'],
                singlePointColor=self.colors['keypoint'],
                matchesMask=inlier_mask.tolist(),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            
            # Draw outliers separately
            outlier_mask = (1 - inlier_mask).astype(bool)
            if np.any(outlier_mask):
                match_img = cv2.drawMatches(
                    img1, kp1, img2, kp2, matches_to_draw, match_img,
                    matchColor=self.colors['outlier'],
                    matchesMask=outlier_mask.tolist(),
                    flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG | 
                          cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
            
            num_inliers = int(np.sum(inlier_mask))
            title = f'Feature Matches: {len(matches)} total, {num_inliers} inliers (green), {len(matches)-num_inliers} outliers (red)'
        else:
            # Draw all matches without distinction
            match_img = cv2.drawMatches(
                img1, kp1, img2, kp2, matches_to_draw, None,
                matchColor=self.colors['match_line'],
                singlePointColor=self.colors['keypoint'],
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            title = f'Feature Matches: {len(matches)} total'
        
        # Create figure
        fig = plt.figure(figsize=(self.figsize[0], self.figsize[1]//1.5), dpi=self.dpi)
        
        # Convert BGR to RGB
        match_img_rgb = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
        
        plt.imshow(match_img_rgb)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=self.dpi)
            print(f"Saved: {output_path}")
        
        plt.close()
    
    def visualize_homography_warp(self, img1: np.ndarray, img2: np.ndarray,
                                 H: np.ndarray,
                                 output_path: Optional[str] = None):
        """
        Visualize homography warping result.
        
        Args:
            img1: Source image
            img2: Target image
            H: Homography matrix (3x3)
            output_path: Path to save figure
        """
        # Warp source image to target
        h, w = img2.shape[:2]
        img1_warped = cv2.warpPerspective(img1, H, (w, h))
        
        # Create difference image
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1_warped, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1_warped
            img2_gray = img2
        
        diff = cv2.absdiff(img1_gray, img2_gray)
        
        # Create blended overlay
        alpha = 0.5
        if len(img1_warped.shape) == 3 and len(img2.shape) == 3:
            overlay = cv2.addWeighted(img1_warped, alpha, img2, 1-alpha, 0)
        else:
            overlay = cv2.addWeighted(img1_gray, alpha, img2_gray, 1-alpha, 0)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        
        # Convert images to RGB for display
        img1_warped_rgb = cv2.cvtColor(img1_warped, cv2.COLOR_BGR2RGB) if len(img1_warped.shape) == 3 else img1_warped
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if len(img2.shape) == 3 else img2
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB) if len(overlay.shape) == 3 else overlay
        
        # Warped source
        axes[0, 0].imshow(img1_warped_rgb, cmap='gray' if len(img1_warped.shape) == 2 else None)
        axes[0, 0].set_title('Warped Image A', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Target
        axes[0, 1].imshow(img2_rgb, cmap='gray' if len(img2.shape) == 2 else None)
        axes[0, 1].set_title('Image B (Target)', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Overlay
        axes[1, 0].imshow(overlay_rgb, cmap='gray' if len(overlay.shape) == 2 else None)
        axes[1, 0].set_title('Overlay (50/50 blend)', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Difference
        axes[1, 1].imshow(diff, cmap='hot')
        axes[1, 1].set_title('Absolute Difference', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=self.dpi)
            print(f"Saved: {output_path}")
        
        plt.close()
    
    def visualize_grid_overlay(self, img1: np.ndarray, img2: np.ndarray,
                              H: np.ndarray,
                              grid_size: int = 20,
                              output_path: Optional[str] = None):
        """
        Visualize geometric distortion using grid overlay.
        
        Args:
            img1: Source image
            img2: Target image
            H: Homography matrix
            grid_size: Grid spacing in pixels
            output_path: Path to save figure
        """
        h, w = img1.shape[:2]
        
        # Create grid points
        x = np.arange(0, w, grid_size)
        y = np.arange(0, h, grid_size)
        xx, yy = np.meshgrid(x, y)
        
        # Flatten grid points
        points = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Transform grid points
        points_homogeneous = np.column_stack([points, np.ones(len(points))])
        transformed = (H @ points_homogeneous.T).T
        transformed = transformed[:, :2] / transformed[:, 2:3]
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)
        
        # Original grid on image 1
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if len(img1.shape) == 3 else img1
        axes[0].imshow(img1_rgb, cmap='gray' if len(img1.shape) == 2 else None)
        
        # Draw horizontal lines
        for i in range(len(y)):
            row_points = points[i*len(x):(i+1)*len(x)]
            axes[0].plot(row_points[:, 0], row_points[:, 1], 'c-', linewidth=1)
        
        # Draw vertical lines
        for j in range(len(x)):
            col_points = points[j::len(x)]
            axes[0].plot(col_points[:, 0], col_points[:, 1], 'c-', linewidth=1)
        
        axes[0].set_title('Original Grid (Image A)', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Transformed grid on image 2
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if len(img2.shape) == 3 else img2
        axes[1].imshow(img2_rgb, cmap='gray' if len(img2.shape) == 2 else None)
        
        # Draw transformed horizontal lines
        for i in range(len(y)):
            row_points = transformed[i*len(x):(i+1)*len(x)]
            axes[1].plot(row_points[:, 0], row_points[:, 1], 'lime', linewidth=1)
        
        # Draw transformed vertical lines
        for j in range(len(x)):
            col_points = transformed[j::len(x)]
            axes[1].plot(col_points[:, 0], col_points[:, 1], 'lime', linewidth=1)
        
        axes[1].set_title('Transformed Grid (Image B)', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=self.dpi)
            print(f"Saved: {output_path}")
        
        plt.close()
    
    def create_comprehensive_visualization(self, pair_dir: str,
                                         detector_type: str = 'ORB',
                                         output_dir: Optional[str] = None):
        """
        Create comprehensive visualization suite for a homography pair.
        
        Args:
            pair_dir: Directory containing pair data
            detector_type: Feature detector to use
            output_dir: Directory to save visualizations
        """
        # Load pair data
        pair_data = self.load_homography_data(pair_dir)
        img1 = pair_data['img1']
        img2 = pair_data['img2']
        H_gt = pair_data.get('H_gt')
        
        pair_name = Path(pair_dir).name
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating visualizations for {pair_name}...")
        
        # 1. Side-by-side comparison
        output_path = output_dir / f"{pair_name}_side_by_side.png" if output_dir else None
        self.visualize_side_by_side(img1, img2, output_path=output_path)
        
        # 2. Detect features
        if detector_type == 'ORB':
            detector = cv2.ORB_create(nfeatures=2000)
        elif detector_type == 'SIFT':
            detector = cv2.SIFT_create(nfeatures=2000)
        elif detector_type == 'AKAZE':
            detector = cv2.AKAZE_create()
        else:
            detector = cv2.ORB_create(nfeatures=2000)
        
        # Convert to grayscale if needed
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        kp1, desc1 = detector.detectAndCompute(gray1, None)
        kp2, desc2 = detector.detectAndCompute(gray2, None)
        
        # 3. Visualize keypoints
        output_path = output_dir / f"{pair_name}_keypoints.png" if output_dir else None
        self.visualize_keypoints(img1, img2, kp1, kp2, output_path=output_path)
        
        # 4. Match features
        if desc1 is not None and desc2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING if detector_type in ['ORB', 'AKAZE', 'BRISK'] else cv2.NORM_L2)
            matches = bf.knnMatch(desc1, desc2, k=2)
            
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            # 5. Estimate homography
            if len(good_matches) >= 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if H is not None:
                    # 6. Visualize matches
                    output_path = output_dir / f"{pair_name}_matches.png" if output_dir else None
                    self.visualize_matches(img1, img2, kp1, kp2, good_matches,
                                         inlier_mask.ravel() if inlier_mask is not None else None,
                                         output_path=output_path)
                    
                    # 7. Visualize warping
                    output_path = output_dir / f"{pair_name}_warp.png" if output_dir else None
                    self.visualize_homography_warp(img1, img2, H, output_path=output_path)
                    
                    # 8. Visualize grid
                    output_path = output_dir / f"{pair_name}_grid.png" if output_dir else None
                    self.visualize_grid_overlay(img1, img2, H, output_path=output_path)
                    
                    print(f"  ✓ Created {4} visualization types")
                else:
                    print(f"  ✗ Homography estimation failed")
            else:
                print(f"  ✗ Insufficient matches: {len(good_matches)}")
        else:
            print(f"  ✗ No descriptors computed")
        
        print(f"✓ Visualization complete for {pair_name}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate diagnostic visualizations for homography pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize single pair
  python visualize_pairs.py \\
    --pair_dir ./data/homography_pairs/sample_00001 \\
    --output_dir ./visualizations \\
    --detector ORB
  
  # Batch visualize multiple pairs
  python visualize_pairs.py \\
    --dataset_dir ./data/homography_pairs \\
    --output_dir ./visualizations \\
    --max_pairs 10
  
  # Custom visualization
  python visualize_pairs.py \\
    --image1 ./data/pairs/sample_001/patch_A.tiff \\
    --image2 ./data/pairs/sample_001/patch_B.tiff \\
    --output_dir ./output

Visualization types generated:
  - Side-by-side image comparison
  - Keypoint detection overlay
  - Feature correspondence lines (inliers in green, outliers in red)
  - Homography warping result with difference map
  - Grid overlay showing geometric distortion
        """
    )
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--pair_dir', type=str, help='Directory containing single pair')
    input_group.add_argument('--dataset_dir', type=str, help='Dataset directory with multiple pairs')
    input_group.add_argument('--image1', type=str, help='First image path (custom visualization)')
    
    parser.add_argument('--image2', type=str, help='Second image path (with --image1)')
    
    # Visualization parameters
    parser.add_argument(
        '--detector',
        type=str,
        default='ORB',
        choices=['ORB', 'SIFT', 'AKAZE', 'BRISK'],
        help='Feature detector (default: ORB)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for visualizations'
    )
    
    parser.add_argument(
        '--max_pairs',
        type=int,
        help='Maximum number of pairs to visualize (for batch processing)'
    )
    
    parser.add_argument(
        '--figsize',
        type=int,
        nargs=2,
        default=[16, 8],
        help='Figure size (width height) in inches'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='Resolution in DPI (default: 150)'
    )
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = HomographyVisualizer(figsize=tuple(args.figsize), dpi=args.dpi)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Single pair visualization
    if args.pair_dir:
        visualizer.create_comprehensive_visualization(
            pair_dir=args.pair_dir,
            detector_type=args.detector,
            output_dir=output_dir
        )
    
    # Custom image pair
    elif args.image1:
        if not args.image2:
            print("Error: --image2 required with --image1")
            sys.exit(1)
        
        img1, img2 = visualizer.load_image_pair(args.image1, args.image2)
        
        # Simple side-by-side
        visualizer.visualize_side_by_side(
            img1, img2,
            title1=Path(args.image1).name,
            title2=Path(args.image2).name,
            output_path=output_dir / "side_by_side.png"
        )
        
        print(f"✓ Visualization saved to {output_dir}")
    
    # Batch processing
    elif args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
        pair_dirs = sorted(list(dataset_dir.glob("sample_*")))
        
        if args.max_pairs:
            pair_dirs = pair_dirs[:args.max_pairs]
        
        print(f"Visualizing {len(pair_dirs)} pair(s)...")
        
        for idx, pair_dir in enumerate(pair_dirs, 1):
            try:
                print(f"\n[{idx}/{len(pair_dirs)}] Processing {pair_dir.name}...")
                visualizer.create_comprehensive_visualization(
                    pair_dir=str(pair_dir),
                    detector_type=args.detector,
                    output_dir=output_dir
                )
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        print(f"\n✓ Batch visualization complete! Saved to {output_dir}")


if __name__ == "__main__":
    main()