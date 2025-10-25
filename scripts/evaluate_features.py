#!/usr/bin/env python3
"""
evaluate_features.py - Feature matching metrics calculation for classical homography methods

This script evaluates feature detection and matching performance for classical computer vision
methods (ORB, SIFT, AKAZE, etc.) on thermal images. It calculates metrics including:
- Keypoint counts
- Inlier ratios
- Reprojection errors
- Homography estimation success rates

Author: THED-PV Team
Date: January 2025
Python: 3.8+
Dependencies: opencv-python, numpy, pandas, scipy
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy import stats

# Suppress warnings
warnings.filterwarnings('ignore')


class FeatureEvaluator:
    """
    Evaluates feature detection and matching performance for classical methods.
    
    Supports multiple feature detectors (ORB, SIFT, AKAZE, BRISK, etc.) and
    calculates comprehensive metrics for homography estimation validation.
    """
    
    def __init__(self, detector_type: str = 'ORB',
                 matcher_type: str = 'BF',
                 ransac_threshold: float = 5.0):
        """
        Initialize the FeatureEvaluator.
        
        Args:
            detector_type: Feature detector ('ORB', 'SIFT', 'AKAZE', 'BRISK', 'ORB-FAST')
            matcher_type: Matcher type ('BF' for BruteForce, 'FLANN')
            ransac_threshold: RANSAC reprojection threshold in pixels
        """
        self.detector_type = detector_type.upper()
        self.matcher_type = matcher_type.upper()
        self.ransac_threshold = ransac_threshold
        
        # Initialize detector and matcher
        self.detector = self._create_detector()
        self.matcher = self._create_matcher()
    
    def _create_detector(self):
        """Create feature detector based on type."""
        if self.detector_type == 'ORB':
            return cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
        
        elif self.detector_type == 'ORB-FAST':
            # ORB with FAST detector (faster but fewer features)
            return cv2.ORB_create(nfeatures=1500, fastThreshold=20)
        
        elif self.detector_type == 'SIFT':
            return cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.04, edgeThreshold=10)
        
        elif self.detector_type == 'AKAZE':
            return cv2.AKAZE_create()
        
        elif self.detector_type == 'BRISK':
            return cv2.BRISK_create()
        
        elif self.detector_type == 'KAZE':
            return cv2.KAZE_create()
        
        else:
            raise ValueError(f"Unsupported detector type: {self.detector_type}")
    
    def _create_matcher(self):
        """Create feature matcher based on type."""
        if self.matcher_type == 'BF':
            # Brute Force matcher
            if self.detector_type in ['ORB', 'ORB-FAST', 'BRISK', 'AKAZE']:
                # Binary descriptors (Hamming distance)
                return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            else:
                # Float descriptors (L2 distance)
                return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        elif self.matcher_type == 'FLANN':
            # FLANN matcher
            if self.detector_type in ['ORB', 'ORB-FAST', 'BRISK', 'AKAZE']:
                # LSH for binary descriptors
                index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                                   table_number=6,
                                   key_size=12,
                                   multi_probe_level=1)
            else:
                # KD-Tree for float descriptors
                index_params = dict(algorithm=1,  # FLANN_INDEX_KDTREE
                                   trees=5)
            
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        
        else:
            raise ValueError(f"Unsupported matcher type: {self.matcher_type}")
    
    def detect_and_compute(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Detect keypoints and compute descriptors.
        
        Args:
            image: Input image (grayscale)
        
        Returns:
            keypoints: List of cv2.KeyPoint objects
            descriptors: Descriptor array (N x D)
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        
        return keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray,
                      ratio_threshold: float = 0.75) -> List:
        """
        Match features using ratio test (Lowe's ratio test).
        
        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image
            ratio_threshold: Ratio test threshold (default: 0.75)
        
        Returns:
            good_matches: List of good cv2.DMatch objects
        """
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []
        
        # Match descriptors using KNN (k=2)
        try:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return []
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def estimate_homography(self, kp1: List, kp2: List,
                           matches: List) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Estimate homography using RANSAC.
        
        Args:
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: List of good matches
        
        Returns:
            H: Homography matrix (3x3) or None if estimation fails
            inlier_mask: Binary mask indicating inliers
        """
        if len(matches) < 4:
            return None, np.array([])
        
        # Extract matched keypoint coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Estimate homography with RANSAC
        try:
            H, inlier_mask = cv2.findHomography(
                src_pts, dst_pts,
                cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold,
                maxIters=2000,
                confidence=0.995
            )
            
            if H is None:
                return None, np.array([])
            
            return H, inlier_mask.ravel()
        
        except cv2.error:
            return None, np.array([])
    
    def compute_reprojection_error(self, kp1: List, kp2: List,
                                   matches: List, H: np.ndarray,
                                   inlier_mask: np.ndarray) -> Dict[str, float]:
        """
        Compute reprojection errors for matched points.
        
        Args:
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: List of matches
            H: Homography matrix
            inlier_mask: Binary mask indicating inliers
        
        Returns:
            dict: Statistics including mean, median, std, max error
        """
        if H is None or len(matches) == 0:
            return {
                'mean_error': np.nan,
                'median_error': np.nan,
                'std_error': np.nan,
                'max_error': np.nan,
                'rmse': np.nan
            }
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # Transform source points using homography
        src_pts_homogeneous = np.column_stack([src_pts, np.ones(len(src_pts))])
        transformed_pts = (H @ src_pts_homogeneous.T).T
        transformed_pts = transformed_pts[:, :2] / transformed_pts[:, 2:3]
        
        # Compute Euclidean distances
        errors = np.linalg.norm(transformed_pts - dst_pts, axis=1)
        
        # Compute statistics
        return {
            'mean_error': float(np.mean(errors)),
            'median_error': float(np.median(errors)),
            'std_error': float(np.std(errors)),
            'max_error': float(np.max(errors)),
            'rmse': float(np.sqrt(np.mean(errors**2)))
        }
    
    def evaluate_image_pair(self, img1_path: str, img2_path: str,
                           H_gt: Optional[np.ndarray] = None) -> Dict:
        """
        Evaluate feature matching for an image pair.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            H_gt: Ground truth homography (optional, for accuracy evaluation)
        
        Returns:
            dict: Comprehensive evaluation metrics
        """
        # Load images
        img1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(img2_path), cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            raise ValueError(f"Failed to load images: {img1_path}, {img2_path}")
        
        results = {
            'image1_path': str(img1_path),
            'image2_path': str(img2_path),
            'detector': self.detector_type,
            'matcher': self.matcher_type
        }
        
        # Detect and compute features
        kp1, desc1 = self.detect_and_compute(img1)
        kp2, desc2 = self.detect_and_compute(img2)
        
        results['keypoints_img1'] = len(kp1)
        results['keypoints_img2'] = len(kp2)
        
        # Match features
        matches = self.match_features(desc1, desc2)
        results['num_matches'] = len(matches)
        
        # Estimate homography
        H, inlier_mask = self.estimate_homography(kp1, kp2, matches)
        
        if H is not None and len(inlier_mask) > 0:
            num_inliers = int(np.sum(inlier_mask))
            results['num_inliers'] = num_inliers
            results['inlier_ratio'] = num_inliers / len(matches) if len(matches) > 0 else 0.0
            results['estimation_success'] = True
            
            # Compute reprojection errors
            error_stats = self.compute_reprojection_error(kp1, kp2, matches, H, inlier_mask)
            results.update(error_stats)
            
            # Compare with ground truth if provided
            if H_gt is not None:
                results['homography_matrix_error'] = float(np.linalg.norm(H - H_gt, 'fro'))
        else:
            results['num_inliers'] = 0
            results['inlier_ratio'] = 0.0
            results['estimation_success'] = False
            results['mean_error'] = np.nan
            results['median_error'] = np.nan
            results['std_error'] = np.nan
            results['max_error'] = np.nan
            results['rmse'] = np.nan
        
        return results
    
    def evaluate_dataset(self, dataset_dir: str,
                        pair_list_file: Optional[str] = None,
                        output_csv: Optional[str] = None,
                        max_pairs: Optional[int] = None) -> pd.DataFrame:
        """
        Evaluate feature matching on an entire dataset.
        
        Args:
            dataset_dir: Root directory containing image pairs
            pair_list_file: Optional CSV file listing image pairs and ground truth
            output_csv: Optional path to save results
            max_pairs: Maximum number of pairs to evaluate (for testing)
        
        Returns:
            pd.DataFrame: Evaluation results for all pairs
        """
        dataset_dir = Path(dataset_dir)
        
        # Load pair list if provided
        if pair_list_file:
            pairs_df = pd.read_csv(pair_list_file)
            pairs = list(zip(pairs_df['image1'], pairs_df['image2']))
            
            # Load ground truth if available
            if 'homography_matrix' in pairs_df.columns:
                gt_homographies = [
                    np.array(json.loads(h)) if pd.notna(h) else None
                    for h in pairs_df['homography_matrix']
                ]
            else:
                gt_homographies = [None] * len(pairs)
        else:
            # Find pairs automatically (assumes directory structure)
            pairs = self._find_image_pairs(dataset_dir)
            gt_homographies = [None] * len(pairs)
        
        if max_pairs:
            pairs = pairs[:max_pairs]
            gt_homographies = gt_homographies[:max_pairs]
        
        print(f"Evaluating {len(pairs)} image pair(s) with {self.detector_type}...")
        
        # Evaluate each pair
        results = []
        for idx, (pair, H_gt) in enumerate(zip(pairs, gt_homographies), 1):
            try:
                img1_path = dataset_dir / pair[0] if not Path(pair[0]).is_absolute() else pair[0]
                img2_path = dataset_dir / pair[1] if not Path(pair[1]).is_absolute() else pair[1]
                
                result = self.evaluate_image_pair(img1_path, img2_path, H_gt)
                results.append(result)
                
                if idx % 50 == 0:
                    print(f"  Evaluated {idx}/{len(pairs)} pairs...")
            
            except Exception as e:
                print(f"  ✗ Error evaluating pair {idx}: {e}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Print summary statistics
        self._print_summary(df)
        
        # Save results
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"\nResults saved to: {output_csv}")
        
        return df
    
    def _find_image_pairs(self, dataset_dir: Path) -> List[Tuple[str, str]]:
        """Find image pairs in dataset directory."""
        pairs = []
        
        # Look for homography pair directories
        pair_dirs = list(dataset_dir.glob("sample_*"))
        
        for pair_dir in sorted(pair_dirs):
            patch_a = pair_dir / "patch_A.tiff"
            patch_b = pair_dir / "patch_B.tiff"
            
            if patch_a.exists() and patch_b.exists():
                pairs.append((str(patch_a), str(patch_b)))
        
        return pairs
    
    def _print_summary(self, df: pd.DataFrame):
        """Print summary statistics."""
        print("\n" + "="*70)
        print(f"Evaluation Summary - {self.detector_type} Detector")
        print("="*70)
        
        if len(df) == 0:
            print("No results to summarize.")
            return
        
        # Success rate
        success_rate = df['estimation_success'].mean() * 100
        print(f"Homography Estimation Success Rate: {success_rate:.2f}%")
        
        # Keypoint statistics
        print(f"\nKeypoint Statistics:")
        print(f"  Mean keypoints (image 1): {df['keypoints_img1'].mean():.1f} ± {df['keypoints_img1'].std():.1f}")
        print(f"  Mean keypoints (image 2): {df['keypoints_img2'].mean():.1f} ± {df['keypoints_img2'].std():.1f}")
        
        # Matching statistics
        print(f"\nMatching Statistics:")
        print(f"  Mean matches: {df['num_matches'].mean():.1f} ± {df['num_matches'].std():.1f}")
        
        # Inlier statistics (only for successful estimations)
        success_df = df[df['estimation_success']]
        if len(success_df) > 0:
            print(f"  Mean inliers: {success_df['num_inliers'].mean():.1f} ± {success_df['num_inliers'].std():.1f}")
            print(f"  Mean inlier ratio: {success_df['inlier_ratio'].mean():.3f} ± {success_df['inlier_ratio'].std():.3f}")
            
            # Reprojection error statistics
            print(f"\nReprojection Error Statistics (successful estimations only):")
            print(f"  Mean RMSE: {success_df['rmse'].mean():.2f} ± {success_df['rmse'].std():.2f} pixels")
            print(f"  Median error: {success_df['median_error'].mean():.2f} ± {success_df['median_error'].std():.2f} pixels")


def compare_detectors(img1_path: str, img2_path: str,
                     detectors: List[str] = None,
                     output_json: Optional[str] = None) -> Dict:
    """
    Compare multiple feature detectors on the same image pair.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        detectors: List of detector names to compare
        output_json: Optional path to save comparison results
    
    Returns:
        dict: Comparison results for all detectors
    """
    if detectors is None:
        detectors = ['ORB', 'SIFT', 'AKAZE', 'BRISK']
    
    print(f"Comparing {len(detectors)} feature detectors...")
    print(f"Image 1: {Path(img1_path).name}")
    print(f"Image 2: {Path(img2_path).name}")
    print()
    
    results = {}
    
    for detector in detectors:
        try:
            print(f"Evaluating {detector}...")
            evaluator = FeatureEvaluator(detector_type=detector)
            result = evaluator.evaluate_image_pair(img1_path, img2_path)
            results[detector] = result
            
            # Print key metrics
            print(f"  Keypoints: {result['keypoints_img1']} / {result['keypoints_img2']}")
            print(f"  Matches: {result['num_matches']}")
            print(f"  Inliers: {result['num_inliers']}")
            print(f"  Inlier ratio: {result['inlier_ratio']:.3f}")
            print(f"  Success: {result['estimation_success']}")
            print()
        
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            continue
    
    # Save comparison
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Comparison results saved to: {output_json}")
    
    return results


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Evaluate feature detection and matching for classical homography methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single image pair
  python evaluate_features.py \\
    --image1 ./data/pairs/sample_001/patch_A.tiff \\
    --image2 ./data/pairs/sample_001/patch_B.tiff \\
    --detector ORB
  
  # Evaluate entire dataset
  python evaluate_features.py \\
    --dataset_dir ./data/homography_pairs \\
    --detector SIFT \\
    --output results_sift.csv
  
  # Compare multiple detectors
  python evaluate_features.py \\
    --image1 ./data/pairs/sample_001/patch_A.tiff \\
    --image2 ./data/pairs/sample_001/patch_B.tiff \\
    --compare
  
  # Evaluate with pair list
  python evaluate_features.py \\
    --dataset_dir ./data/splits/test \\
    --pair_list test_pairs.csv \\
    --detector ORB \\
    --output test_results.csv

Supported detectors: ORB, SIFT, AKAZE, BRISK, KAZE, ORB-FAST
        """
    )
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image1', type=str, help='First image path')
    input_group.add_argument('--dataset_dir', type=str, help='Dataset directory')
    
    parser.add_argument('--image2', type=str, help='Second image path (with --image1)')
    parser.add_argument('--pair_list', type=str, help='CSV file listing image pairs')
    
    # Detector parameters
    parser.add_argument(
        '--detector',
        type=str,
        default='ORB',
        choices=['ORB', 'SIFT', 'AKAZE', 'BRISK', 'KAZE', 'ORB-FAST'],
        help='Feature detector type (default: ORB)'
    )
    
    parser.add_argument(
        '--matcher',
        type=str,
        default='BF',
        choices=['BF', 'FLANN'],
        help='Matcher type (default: BF)'
    )
    
    parser.add_argument(
        '--ransac_threshold',
        type=float,
        default=5.0,
        help='RANSAC reprojection threshold in pixels (default: 5.0)'
    )
    
    # Comparison mode
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple detectors (requires --image1 and --image2)'
    )
    
    # Output arguments
    parser.add_argument('--output', type=str, help='Output CSV file')
    parser.add_argument('--max_pairs', type=int, help='Maximum pairs to evaluate (for testing)')
    
    args = parser.parse_args()
    
    # Comparison mode
    if args.compare:
        if not args.image1 or not args.image2:
            print("Error: --compare requires --image1 and --image2")
            sys.exit(1)
        
        compare_detectors(args.image1, args.image2, output_json=args.output)
        return
    
    # Single pair evaluation
    if args.image1:
        if not args.image2:
            print("Error: --image2 required with --image1")
            sys.exit(1)
        
        evaluator = FeatureEvaluator(
            detector_type=args.detector,
            matcher_type=args.matcher,
            ransac_threshold=args.ransac_threshold
        )
        
        result = evaluator.evaluate_image_pair(args.image1, args.image2)
        
        print("\n" + "="*70)
        print("Evaluation Results")
        print("="*70)
        for key, value in result.items():
            print(f"{key:30s}: {value}")
        
        if args.output:
            df = pd.DataFrame([result])
            df.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")
    
    # Dataset evaluation
    elif args.dataset_dir:
        evaluator = FeatureEvaluator(
            detector_type=args.detector,
            matcher_type=args.matcher,
            ransac_threshold=args.ransac_threshold
        )
        
        evaluator.evaluate_dataset(
            dataset_dir=args.dataset_dir,
            pair_list_file=args.pair_list,
            output_csv=args.output,
            max_pairs=args.max_pairs
        )


if __name__ == "__main__":
    main()