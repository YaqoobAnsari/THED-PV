# HR-ThermalPV API Reference

## Module: `preprocess.py`

### Class: `ThermalPreprocessor`

Main preprocessing pipeline for thermal images.

#### Constructor

```python
ThermalPreprocessor(config_path: Optional[str] = None)
```

**Parameters:**
- `config_path` (str, optional): Path to YAML configuration file. Uses defaults if None.

**Example:**
```python
from preprocess import ThermalPreprocessor

preprocessor = ThermalPreprocessor('configs/preprocessing_config.yaml')
```

#### Methods

##### `read_thermal_image(path: Path) -> np.ndarray`

Read 14-bit thermal TIFF image.

**Parameters:**
- `path` (Path): Path to TIFF file

**Returns:**
- `np.ndarray`: Image array (dtype: uint16 for 14-bit)

##### `normalize_to_8bit(img: np.ndarray) -> np.ndarray`

Normalize 14-bit image to 8-bit range.

**Parameters:**
- `img` (np.ndarray): Input 14-bit image

**Returns:**
- `np.ndarray`: 8-bit normalized image (dtype: uint8)

##### `apply_clahe(img: np.ndarray) -> np.ndarray`

Apply Contrast Limited Adaptive Histogram Equalization.

**Parameters:**
- `img` (np.ndarray): Input 8-bit grayscale image

**Returns:**
- `np.ndarray`: CLAHE-enhanced image

##### `apply_bilateral_filter(img: np.ndarray) -> np.ndarray`

Apply bilateral filtering for edge-preserving denoising.

**Parameters:**
- `img` (np.ndarray): Input 8-bit grayscale image

**Returns:**
- `np.ndarray`: Filtered image

##### `suppress_glare(img: np.ndarray) -> np.ndarray`

Adaptive glare suppression using mean + k*std threshold.

**Parameters:**
- `img` (np.ndarray): Input 8-bit grayscale image

**Returns:**
- `np.ndarray`: Glare-suppressed image

##### `correct_shadows(img: np.ndarray) -> np.ndarray`

Localized intensity normalization for shadow correction.

**Parameters:**
- `img` (np.ndarray): Input 8-bit grayscale image

**Returns:**
- `np.ndarray`: Shadow-corrected image

##### `preprocess_image(img: np.ndarray) -> np.ndarray`

Complete preprocessing pipeline.

**Parameters:**
- `img` (np.ndarray): Raw thermal image (14-bit or 8-bit)

**Returns:**
- `np.ndarray`: Fully preprocessed 8-bit image

**Pipeline:**
1. Normalize to 8-bit
2. Glare suppression
3. CLAHE enhancement
4. Bilateral filtering
5. Shadow correction
6. Sharpening

##### `quarter_image(img: np.ndarray) -> List[np.ndarray]`

Divide 640×512 image into four 320×256 patches.

**Parameters:**
- `img` (np.ndarray): Input image (640×512)

**Returns:**
- `List[np.ndarray]`: List of 4 patches [top-left, top-right, bottom-left, bottom-right]

---

## Module: `generate_homography_pairs.py`

### Class: `HomographyPairGenerator`

Generate homography pairs with controlled transformations.

#### Constructor

```python
HomographyPairGenerator(
    patch_size: int = 256,
    perturbation_range: Tuple[int, int] = (15, 50),
    overlap_range: Tuple[float, float] = (0.55, 0.85),
    center_bias: float = 0.2,
    max_attempts: int = 100
)
```

**Parameters:**
- `patch_size` (int): Size of square patches (default: 256)
- `perturbation_range` (tuple): (min, max) pixel perturbation for corners
- `overlap_range` (tuple): (min, max) overlap ratio between patches
- `center_bias` (float): Bias towards image center, 0=uniform, 1=center only
- `max_attempts` (int): Maximum attempts to generate valid pair

#### Methods

##### `generate_pair(img: np.ndarray) -> Optional[dict]`

Generate a single valid homography pair.

**Parameters:**
- `img` (np.ndarray): Input preprocessed image (grayscale)

**Returns:**
- `dict` or `None`: Dictionary containing:
  - `patch_A`: Original patch (256×256)
  - `patch_B`: Warped patch (256×256)
  - `homography_matrix`: 3×3 homography matrix
  - `4point_params`: 8-element displacement vector
  - `corners1`: Original corner coordinates
  - `corners2`: Perturbed corner coordinates
  - `attempts`: Number of attempts taken

**Example:**
```python
generator = HomographyPairGenerator()
img = cv2.imread('image.tiff', cv2.IMREAD_GRAYSCALE)
pair = generator.generate_pair(img)

if pair:
    cv2.imwrite('patch_A.tiff', pair['patch_A'])
    np.save('homography.npy', pair['homography_matrix'])
```

##### `compute_homography(corners1: np.ndarray, corners2: np.ndarray) -> np.ndarray`

Compute 3×3 homography matrix from corner correspondences.

**Parameters:**
- `corners1` (np.ndarray): Source corners (4×2)
- `corners2` (np.ndarray): Destination corners (4×2)

**Returns:**
- `np.ndarray`: 3×3 homography matrix

---

## Module: `split_dataset.py`

### Class: `DatasetSplitter`

Split homography pairs into train/val/test sets.

#### Constructor

```python
DatasetSplitter(
    train_ratio: float = 0.5,
    val_ratio: float = 0.25,
    test_ratio: float = 0.25,
    seed: int = 42
)
```

**Parameters:**
- `train_ratio` (float): Training set proportion
- `val_ratio` (float): Validation set proportion
- `test_ratio` (float): Test set proportion
- `seed` (int): Random seed for reproducibility

---

## Module: `utils.py`

### Functions

##### `compute_homography_orb(img1, img2, max_features=2000, match_ratio=0.75)`

Compute homography using ORB features.

**Parameters:**
- `img1` (np.ndarray): First image
- `img2` (np.ndarray): Second image
- `max_features` (int): Maximum ORB features
- `match_ratio` (float): Lowe's ratio test threshold

**Returns:**
- `tuple`: (H, num_inliers, num_total)
  - `H`: 3×3 homography matrix or None
  - `num_inliers`: Number of RANSAC inliers
  - `num_total`: Total matches before RANSAC

**Example:**
```python
from utils import compute_homography_orb

H, inliers, total = compute_homography_orb(img1, img2)
inlier_ratio = inliers / total if total > 0 else 0
print(f"ORB Inlier Ratio: {inlier_ratio:.2%}")
```

##### `compute_homography_sift(img1, img2, match_ratio=0.75)`

Compute homography using SIFT features.

**Returns:**
- `tuple`: (H, num_inliers, num_total)

##### `compute_corner_error(H_pred, H_gt, img_size=(256, 256))`

Compute Mean Average Corner Error (MACE).

**Parameters:**
- `H_pred` (np.ndarray): Predicted homography (3×3)
- `H_gt` (np.ndarray): Ground truth homography (3×3)
- `img_size` (tuple): Image dimensions (height, width)

**Returns:**
- `float`: MACE in pixels

**Example:**
```python
mace = compute_corner_error(H_predicted, H_ground_truth)
print(f"MACE: {mace:.2f} pixels")
```

##### `load_homography_pair(sample_dir: Path) -> dict`

Load a complete homography pair sample.

**Parameters:**
- `sample_dir` (Path): Path to sample directory

**Returns:**
- `dict`: Dictionary with keys:
  - `patch_A`: Original patch
  - `patch_B`: Warped patch
  - `homography_matrix`: 3×3 matrix
  - `4point_params`: 8-element vector

##### `compute_entropy(img: np.ndarray) -> float`

Compute Shannon entropy of image.

**Parameters:**
- `img` (np.ndarray): Grayscale image

**Returns:**
- `float`: Entropy value (bits)

##### `count_features(img: np.ndarray, method: str = 'orb') -> int`

Count detected features in image.

**Parameters:**
- `img` (np.ndarray): Grayscale image
- `method` (str): 'orb' or 'sift'

**Returns:**
- `int`: Number of detected keypoints

---

## Command Line Interface

### Preprocessing

```bash
python src/preprocess.py \
    --input_dir PATH \
    --output_dir PATH \
    [--config PATH]
```

### Homography Pair Generation

```bash
python src/generate_homography_pairs.py \
    --input_dir PATH \
    --output_dir PATH \
    [--num_pairs INT] \
    [--patch_size INT] \
    [--perturbation_range INT INT] \
    [--overlap_range FLOAT FLOAT]
```

### Dataset Splitting

```bash
python src/split_dataset.py \
    --input_dir PATH \
    --output_dir PATH \
    [--train_ratio FLOAT] \
    [--val_ratio FLOAT] \
    [--test_ratio FLOAT] \
    [--seed INT]
```

---

## Data Loading Example

### PyTorch Dataset

```python
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path

class HR_ThermalPV_Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.samples = sorted(list(self.data_dir.glob("sample_*")))
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_dir = self.samples[idx]
        
        # Load images
        patch_A = cv2.imread(str(sample_dir / "patch_A.tiff"), 
                            cv2.IMREAD_GRAYSCALE)
        patch_B = cv2.imread(str(sample_dir / "patch_B.tiff"), 
                            cv2.IMREAD_GRAYSCALE)
        
        # Load ground truth
        H = np.load(sample_dir / "homography_matrix.npy")
        params_4pt = np.load(sample_dir / "4point_params.npy")
        
        # Convert to tensors
        patch_A = torch.from_numpy(patch_A).float().unsqueeze(0) / 255.0
        patch_B = torch.from_numpy(patch_B).float().unsqueeze(0) / 255.0
        params_4pt = torch.from_numpy(params_4pt).float()
        
        return {
            'patch_A': patch_A,
            'patch_B': patch_B,
            '4point_params': params_4pt,
            'homography_matrix': H
        }
```

**Usage:**
```python
train_dataset = HR_ThermalPV_Dataset('data/splits/train')
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True
)
```

---

## Version Information

**Package Version**: 1.0.0  
**Python Requirements**: >=3.8  
**Key Dependencies**: opencv-python>=4.5.0, numpy>=1.21.0