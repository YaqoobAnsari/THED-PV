"""
Utility functions for HR-ThermalPV dataset
Feature extraction, homography computation, and visualization tools
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt


def compute_homography_orb(
    img1: np.ndarray,
    img2: np.ndarray,
    max_features: int = 2000,
    match_ratio: float = 0.75
) -> Tuple[Optional[np.ndarray], int, int]:
    """
    Compute homography using ORB features
    
    Args:
        img1: First image
        img2: Second image
        max_features: Maximum number of ORB features
        match_ratio: Lowe's ratio test threshold
    
    Returns:
        H: Homography matrix (3x3) or None if failed
        num_inliers: Number of inlier matches
        num_total: Total number of matches
    """
    # Initialize ORB
    orb = cv2.ORB_create(nfeatures=max_features)
    
    # Detect and compute
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        return None, 0, 0
    
    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < match_ratio * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 4:
        return None, 0, len(good_matches)
    
    # Extract matched keypoint locations
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Compute homography with RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    num_inliers = int(mask.sum()) if mask is not None else 0
    
    return H, num_inliers, len(good_matches)


def compute_homography_sift(
    img1: np.ndarray,
    img2: np.ndarray,
    match_ratio: float = 0.75
) -> Tuple[Optional[np.ndarray], int, int]:
    """
    Compute homography using SIFT features
    
    Args:
        img1: First image
        img2: Second image
        match_ratio: Lowe's ratio test threshold
    
    Returns:
        H: Homography matrix (3x3) or None if failed
        num_inliers: Number of inlier matches
        num_total: Total number of matches
    """
    # Initialize SIFT
    sift = cv2.SIFT_create()
    
    # Detect and compute
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        return None, 0, 0
    
    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < match_ratio * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 4:
        return None, 0, len(good_matches)
    
    # Extract matched keypoint locations
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Compute homography with RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    num_inliers = int(mask.sum()) if mask is not None else 0
    
    return H, num_inliers, len(good_matches)


def compute_corner_error(
    H_pred: np.ndarray,
    H_gt: np.ndarray,
    img_size: Tuple[int, int] = (256, 256)
) -> float:
    """
    Compute Mean Average Corner Error (MACE)
    
    Args:
        H_pred: Predicted homography matrix
        H_gt: Ground truth homography matrix
        img_size: Image size (height, width)
    
    Returns:
        mace: Mean average corner error in pixels
    """
    h, w = img_size
    
    # Define corners
    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1]
    ], dtype=np.float32).T
    
    # Transform corners
    corners_pred = H_pred @ corners
    corners_gt = H_gt @ corners
    
    # Convert from homogeneous
    corners_pred = corners_pred[:2, :] / corners_pred[2, :]
    corners_gt = corners_gt[:2, :] / corners_gt[2, :]
    
    # Compute Euclidean distance
    errors = np.linalg.norm(corners_pred - corners_gt, axis=0)
    mace = np.mean(errors)
    
    return mace


def visualize_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    method: str = 'orb',
    save_path: Optional[Path] = None
):
    """Visualize feature matches between two images"""
    
    if method.lower() == 'orb':
        detector = cv2.ORB_create(nfeatures=2000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:  # SIFT
        detector = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Detect and compute
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        print("No features detected")
        return
    
    # Match
    matches = matcher.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good = []
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < 0.75 * n.distance:
                good.append(m)
    
    # Draw matches
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, good[:50],  # Limit to 50 for visibility
        None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    # Display or save
    plt.figure(figsize=(15, 6))
    plt.imshow(img_matches, cmap='gray')
    plt.title(f'{method.upper()} Matches ({len(good)} good matches)')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    else:
        plt.show()
    
    plt.close()


def load_homography_pair(sample_dir: Path) -> dict:
    """Load a homography pair sample"""
    
    sample_dir = Path(sample_dir)
    
    patch_A = cv2.imread(str(sample_dir / "patch_A.tiff"), cv2.IMREAD_GRAYSCALE)
    patch_B = cv2.imread(str(sample_dir / "patch_B.tiff"), cv2.IMREAD_GRAYSCALE)
    H_matrix = np.load(sample_dir / "homography_matrix.npy")
    params_4pt = np.load(sample_dir / "4point_params.npy")
    
    return {
        'patch_A': patch_A,
        'patch_B': patch_B,
        'homography_matrix': H_matrix,
        '4point_params': params_4pt
    }


def compute_image_quality_metrics(
    img1: np.ndarray,
    img2: np.ndarray
) -> dict:
    """Compute PSNR, SSIM between two images"""
    
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    
    # Ensure same size
    if img1.shape != img2.shape:
        raise ValueError("Images must have same dimensions")
    
    # Compute PSNR
    psnr = peak_signal_noise_ratio(img1, img2, data_range=255)
    
    # Compute SSIM
    ssim = structural_similarity(img1, img2, data_range=255)
    
    return {
        'psnr': psnr,
        'ssim': ssim
    }


def compute_entropy(img: np.ndarray) -> float:
    """Compute Shannon entropy of image"""
    histogram, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    histogram = histogram / histogram.sum()
    histogram = histogram[histogram > 0]  # Remove zeros
    entropy = -np.sum(histogram * np.log2(histogram))
    return entropy


def count_features(img: np.ndarray, method: str = 'orb') -> int:
    """Count number of detected features"""
    
    if method.lower() == 'orb':
        detector = cv2.ORB_create(nfeatures=2000)
    else:
        detector = cv2.SIFT_create()
    
    kp, _ = detector.detectAndCompute(img, None)
    return len(kp) if kp else 0


def visualize_homography_warp(
    img: np.ndarray,
    H: np.ndarray,
    save_path: Optional[Path] = None
):
    """Visualize original image and warped version"""
    
    h, w = img.shape[:2]
    warped = cv2.warpPerspective(img, H, (w, h))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(warped, cmap='gray')
    axes[1].set_title('Warped Image')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    else:
        plt.show()
    
    plt.close()