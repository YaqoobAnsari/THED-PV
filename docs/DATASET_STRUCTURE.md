# HR-ThermalPV Dataset Structure

## Overview

The HR-ThermalPV dataset is organized hierarchically to preserve experimental metadata and enable easy filtering by capture conditions.

## Directory Hierarchy

```
HR-ThermalPV/
└── data/
    ├── raw/                    # Original 14-bit thermal images
    ├── preprocessed/           # Processed and quartered images
    ├── homography_pairs/       # Generated training pairs
    └── splits/                 # Train/val/test splits
```

## 1. Raw Data Structure

```
raw/
├── 2024-12-21/                # Day 1: Clean panels
├── 2024-12-22/                # Day 2: Soiled Level 1
├── 2024-12-23/                # Day 3: Soiled Level 2
├── 2024-12-24/                # Day 4: Soiled Level 3
└── 2024-12-25/                # Day 5: Soiled Level 4
    ├── 8am/                   # Morning capture
    ├── 10am/                  # Mid-morning capture
    ├── 12pm/                  # Noon capture
    └── 2pm/                   # Afternoon capture
        ├── 10cm/              # Height: 10cm from panel
        ├── 20cm/              # Height: 20cm from panel
        ├── 30cm/              # Height: 30cm from panel
        └── 40cm/              # Height: 40cm from panel
            ├── YYYYMMDD_HHMMSS_angle30.tiff
            └── YYYYMMDD_HHMMSS_angle60.tiff
```

### File Naming Convention (Raw)

**Format**: `YYYYMMDD_HHMMSS_angle{XX}.tiff`

- `YYYYMMDD`: Date of capture (e.g., 20241221)
- `HHMMSS`: Time of capture (e.g., 083045 = 08:30:45)
- `angleXX`: Camera inclination angle (30 or 60 degrees)

**Example**: `20241222_100523_angle30.tiff`
- Captured: December 22, 2024 at 10:05:23
- Camera angle: 30 degrees

### Image Specifications (Raw)

| Property | Value |
|----------|-------|
| Resolution | 640 × 512 pixels |
| Bit Depth | 14-bit |
| Format | TIFF (uncompressed) |
| Color Space | Grayscale (thermal) |
| File Size | ~640 KB per image |

## 2. Preprocessed Data Structure

The preprocessed data mirrors the raw structure but with quartered images:

```
preprocessed/
└── [Same hierarchy as raw]
    └── YYYYMMDD_HHMMSS_angle{XX}_patch{N}.tiff
```

### File Naming Convention (Preprocessed)

**Format**: `YYYYMMDD_HHMMSS_angle{XX}_patch{N}.tiff`

- Same as raw, plus:
- `patchN`: Patch index (0-3)
  - `patch0`: Top-left quarter
  - `patch1`: Top-right quarter
  - `patch2`: Bottom-left quarter
  - `patch3`: Bottom-right quarter

**Example**: `20241222_100523_angle30_patch2.tiff`

### Image Specifications (Preprocessed)

| Property | Value |
|----------|-------|
| Resolution | 320 × 256 pixels |
| Bit Depth | 8-bit (normalized) |
| Format | TIFF |
| Preprocessing | CLAHE + Bilateral + Glare/Shadow suppression |
| File Size | ~80 KB per patch |

### Preprocessing Pipeline Applied

1. **14-bit to 8-bit normalization**
2. **Glare suppression**: Adaptive thresholding (mean + 2σ)
3. **CLAHE**: Clip limit 0.018, 8×8 tiles
4. **Bilateral filtering**: d=9, σ_color=0.015, σ_space=8
5. **Shadow correction**: Morphological background estimation
6. **Sharpening**: Unsharp mask

## 3. Homography Pairs Structure

```
homography_pairs/
├── sample_000000/
├── sample_000001/
├── sample_000002/
└── ...
    ├── patch_A.tiff              # Original patch
    ├── patch_B.tiff              # Warped/translated patch
    ├── homography_matrix.npy     # 3×3 homography matrix
    ├── 4point_params.npy         # 8-element displacement vector
    └── metadata.json             # Generation parameters
```

### Metadata File Format

```json
{
  "sample_id": 12345,
  "patch1_topleft": [128, 96],
  "patch2_topleft": [140, 110],
  "corners1": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
  "corners2": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
  "attempts": 3
}
```

### Homography Pair Generation Parameters

| Parameter | Value |
|-----------|-------|
| Patch Size | 256 × 256 |
| Perturbation Range | 15-50 pixels |
| Overlap Ratio | 55-85% |
| Center Bias | 0.2 |
| Pairs per Image | 2 |

## 4. Split Dataset Structure

```
splits/
├── train/
│   ├── sample_000000/
│   ├── sample_000012/
│   └── ...
├── val/
│   ├── sample_000003/
│   ├── sample_000015/
│   └── ...
├── test/
│   ├── sample_000007/
│   ├── sample_000019/
│   └── ...
├── split_summary.json
├── train_samples.txt
├── val_samples.txt
└── test_samples.txt
```

### Split Ratios

| Split | Ratio | Approximate Count |
|-------|-------|-------------------|
| Train | 50%   | ~49,840 pairs |
| Val   | 25%   | ~24,920 pairs |
| Test  | 25%   | ~24,920 pairs |

### Split Strategy

- **Stratified by temporal groups**: Ensures each split contains samples from all days and times
- **Random seed**: 42 (for reproducibility)
- **No data leakage**: Samples from same source image are kept together

## Dataset Statistics

### Capture Conditions

| Dimension | Values |
|-----------|--------|
| Days | 5 (1 clean + 4 soiled) |
| Times | 4 (8am, 10am, 12pm, 2pm) |
| Heights | 4 (10cm, 20cm, 30cm, 40cm) |
| Angles | 2 (30°, 60°) |

**Total raw images**: 5 × 4 × 4 × 2 = 160 unique capture configurations
**Actual count**: 12,460 images (multiple captures per configuration)

### Storage Requirements

| Dataset Component | Size (Compressed) | Size (Uncompressed) |
|-------------------|-------------------|---------------------|
| Raw Data | ~8 GB | ~8 GB |
| Preprocessed | ~4 GB | ~4 GB |
| Homography Pairs | ~20 GB | ~20 GB |
| **Total** | **~32 GB** | **~32 GB** |

## Usage Examples

### Load Raw Image

```python
import cv2
img = cv2.imread('data/raw/2024-12-22/10am/20cm/20241222_100523_angle30.tiff', 
                 cv2.IMREAD_UNCHANGED)
# Shape: (512, 640), dtype: uint16 (14-bit)
```

### Load Preprocessed Patch

```python
img = cv2.imread('data/preprocessed/.../20241222_100523_angle30_patch0.tiff',
                 cv2.IMREAD_GRAYSCALE)
# Shape: (256, 320), dtype: uint8
```

### Load Homography Pair

```python
import numpy as np

sample_dir = 'data/homography_pairs/sample_000000/'
patch_A = cv2.imread(f'{sample_dir}/patch_A.tiff', cv2.IMREAD_GRAYSCALE)
patch_B = cv2.imread(f'{sample_dir}/patch_B.tiff', cv2.IMREAD_GRAYSCALE)
H = np.load(f'{sample_dir}/homography_matrix.npy')
params_4pt = np.load(f'{sample_dir}/4point_params.npy')
```

## Quality Control

All images have been validated for:
- ✅ Correct resolution
- ✅ No corruption
- ✅ Proper bit depth
- ✅ Complete metadata
- ✅ No black padding in homography pairs

See `scripts/verify_dataset.py` for validation code.