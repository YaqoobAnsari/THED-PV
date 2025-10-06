"""
HR-ThermalPV: High-Resolution Thermal Imaging Dataset for Photovoltaic Homography

This package provides tools for preprocessing, generating, and working with
the HR-ThermalPV dataset.
"""

__version__ = "1.0.0"
__author__ = "Mohammed Yaqoob, Mohammed Yusuf Ansari, Dhanup Somasekharan Pillai, Eduardo Feo Flushing"
__email__ = "yansari@tamu.edu"
__license__ = "CC BY 4.0"

# Import main classes for convenience
from .preprocess import ThermalPreprocessor
from .generate_homography_pairs import HomographyPairGenerator
from .split_dataset import DatasetSplitter

# Import utility functions
from .utils import (
    compute_homography_orb,
    compute_homography_sift,
    compute_corner_error,
    load_homography_pair,
    compute_entropy,
    count_features,
    visualize_matches,
    visualize_homography_warp
)

__all__ = [
    # Classes
    'ThermalPreprocessor',
    'HomographyPairGenerator',
    'DatasetSplitter',
    
    # Functions
    'compute_homography_orb',
    'compute_homography_sift',
    'compute_corner_error',
    'load_homography_pair',
    'compute_entropy',
    'count_features',
    'visualize_matches',
    'visualize_homography_warp',
]