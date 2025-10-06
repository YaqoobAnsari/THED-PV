# HR-ThermalPV: High-Resolution Thermal Imaging Dataset for Photovoltaic Homography

[![DOI](https://zenodo.org/badge/DOI/XX.XXXX/zenodo.XXXXXXX.svg)](https://doi.org/XX.XXXX/zenodo.XXXXXXX)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

**HR-ThermalPV** is a high-resolution (640×512) thermal imaging dataset comprising **12,460 raw thermal images** of photovoltaic (PV) panels captured under diverse and controlled conditions. This dataset is specifically designed to advance deep learning models for thermal homography estimation in solar energy systems.

### Key Features
- ✅ **High Resolution**: 640×512 pixel thermal images (14-bit TIFF format)
- ✅ **Multi-Perspective Capture**: 4 heights (10, 20, 30, 40cm) × 2 angles (30°, 60°)
- ✅ **Temporal Variation**: 4 daily capture times (8 AM, 10 AM, 12 PM, 2 PM)
- ✅ **Environmental Diversity**: 5 cleanliness levels (clean → progressively soiled over 5 days)
- ✅ **Homography-Ready**: 99,680 preprocessed image pairs for deep learning
- ✅ **Artifact Suppression**: Glare, shadow, and noise removal pipeline included
- ✅ **Production Code**: Complete preprocessing, generation, and validation scripts

## Dataset Statistics

| Metric | Value |
|--------|-------|
| **Raw Images** | 12,460 |
| **Resolution** | 640×512 pixels |
| **Bit Depth** | 14-bit |
| **Preprocessed Patches** | 49,840 (320×256) |
| **Homography Pairs** | 99,680 |
| **Collection Period** | 5 days |
| **Capture Locations** | Qatar Environment & Energy Research Institute (QEERI) |

## Quick Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/HR-ThermalPV.git
cd HR-ThermalPV

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Usage

### Step 1: Download Dataset from Zenodo

```bash
# Download raw data (12,460 images, ~XX GB)
python scripts/download_data.py --type raw --output ./data/raw

# OR download preprocessed data (49,840 patches, ~XX GB)
python scripts/download_data.py --type preprocessed --output ./data/preprocessed
```

Full instructions: [data/download_instructions.md](data/download_instructions.md)

### Step 2: Preprocess Raw Images

```bash
python src/preprocess.py \
    --input_dir ./data/raw \
    --output_dir ./data/preprocessed \
    --config configs/preprocessing_config.yaml
```

**What this does:**
- CLAHE contrast enhancement (clip limit: 0.018)
- Bilateral filtering for noise reduction
- Adaptive glare suppression
- Shadow correction
- Image sharpening
- Quarters 640×512 images → four 320×256 patches

### Step 3: Generate Homography Pairs

```bash
python src/generate_homography_pairs.py \
    --input_dir ./data/preprocessed \
    --output_dir ./data/homography_pairs \
    --num_pairs 2 \
    --perturbation_range 15 50 \
    --overlap_range 0.55 0.85
```

**Generates:**
- Image pairs with controlled geometric transformations
- Ground truth homography matrices (3×3)
- 4-point parameterization for deep learning
- Metadata files with generation parameters

### Step 4: Split for Training

```bash
python src/split_dataset.py \
    --input_dir ./data/homography_pairs \
    --output_dir ./data/splits \
    --train_ratio 0.5 \
    --val_ratio 0.25 \
    --test_ratio 0.25 \
    --seed 42
```

## Dataset Structure

```
data/
├── raw/                          # Original 14-bit TIFF images
│   ├── 2024-12-21/               # Day 1 (Clean)
│   ├── 2024-12-22/               # Day 2 (Soiled 1)
│   ├── 2024-12-23/               # Day 3 (Soiled 2)
│   ├── 2024-12-24/               # Day 4 (Soiled 3)
│   └── 2024-12-25/               # Day 5 (Soiled 4)
│       └── {8am,10am,12pm,2pm}/
│           └── {10cm,20cm,30cm,40cm}/
│               ├── YYYYMMDD_HHMMSS_angle30.tiff
│               └── YYYYMMDD_HHMMSS_angle60.tiff
│
├── preprocessed/                 # 320×256 patches (8-bit)
│   └── [mirrors raw structure]
│       └── *_patch{0-3}.tiff
│
├── homography_pairs/             # Training-ready pairs
│   └── sample_XXXXX/
│       ├── patch_A.tiff
│       ├── patch_B.tiff
│       ├── homography_matrix.npy
│       └── metadata.txt
│
└── splits/                       # Train/val/test split
    ├── train/
    ├── val/
    └── test/
```

See [docs/DATASET_STRUCTURE.md](docs/DATASET_STRUCTURE.md) for complete details.

## Example Workflows

### Training HomographyNet

```python
from hr_thermalpv import load_dataset, HomographyDataset
import torch

# Load dataset
train_data = HomographyDataset('./data/splits/train')
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)

# Your training loop here
```

See [examples/03_training_example.ipynb](examples/03_training_example.ipynb)

### Feature-Based Homography (ORB/SIFT)

```python
from hr_thermalpv.utils import compute_homography_orb

img1 = cv2.imread('patch_A.tiff', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('patch_B.tiff', cv2.IMREAD_GRAYSCALE)

H, inliers = compute_homography_orb(img1, img2)
print(f"Inlier ratio: {inliers/total_matches:.2%}")
```

## Repository Structure

```
HR-ThermalPV/
├── src/                          # Core source code
│   ├── preprocess.py             # Preprocessing pipeline
│   ├── generate_homography_pairs.py
│   ├── split_dataset.py
│   ├── utils.py                  # Helper functions
│   └── validators.py             # Data validation
├── configs/                      # Configuration files
│   └── preprocessing_config.yaml
├── examples/                     # Jupyter notebooks
│   ├── 01_preprocessing_example.ipynb
│   ├── 02_homography_generation_example.ipynb
│   └── 03_training_example.ipynb
├── tests/                        # Unit tests
├── docs/                         # Documentation
└── scripts/                      # Utility scripts
    ├── download_data.py
    └── verify_dataset.py
```

## Validation Metrics

Our preprocessing and generation pipeline has been validated with:

| Metric | Raw Images | Preprocessed |
|--------|-----------|--------------|
| **PSNR (dB)** | - | 27.3 - 32.0 |
| **SSIM** | - | 0.83 - 0.94 |
| **ORB Keypoints** | 10 - 154 | 395 - 1545 |
| **SIFT Keypoints** | 14 - 154 | 150 - 823 |
| **Inlier Ratio (ORB)** | - | 0.92 - 0.98 |

See [Technical Validation](docs/TECHNICAL_VALIDATION.md)

## Citation

If you use HR-ThermalPV in your research, please cite:

```bibtex
@article{yaqoob2025hrthermalpv,
  title={HR-ThermalPV: A High-Resolution Multi-Perspective Thermal Imaging Dataset for Photovoltaic Homography Estimation},
  author={Yaqoob, Mohammed and Ansari, Mohammed Yusuf and Pillai, Dhanup Somasekharan and Flushing, Eduardo Feo},
  journal={Nature Scientific Data},
  year={2025},
  doi={XX.XXXX/XXXXX}
}
```

## License

This dataset is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE).

You are free to:
- Share and redistribute
- Adapt and build upon

Under the terms:
- Attribution required
- Indicate if changes were made

## Acknowledgements

This work was supported by the AICC grant AICC04-0715-210006 from the Qatar National Research Fund. Data collection was conducted at the Qatar Environment and Energy Research Institute (QEERI) Outdoor Test Facility.

## Contact & Support

- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/HR-ThermalPV/issues)
- **Email**: yansari@tamu.edu
- **Documentation**: [Full Docs](https://YOUR_USERNAME.github.io/HR-ThermalPV/)

## Changelog

**v1.0.0** (2025-XX-XX)
- Initial release
- 12,460 raw thermal images
- Preprocessing pipeline
- Homography pair generation
- Train/val/test splits

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.