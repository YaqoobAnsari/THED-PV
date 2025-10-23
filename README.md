# THED-PV: A High-Resolution Multi-Perspective Thermal Imaging Dataset for Photovoltaic Homography Estimation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17404246.svg)](https://doi.org/10.5281/zenodo.17404246)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

**THED-PV** (Thermal Homography Estimation Dataset for Photovoltaics) is a high-resolution (640×512) thermal imaging dataset comprising **12,460 raw thermal images** of photovoltaic (PV) panels captured under diverse and controlled conditions. This dataset is specifically designed to advance deep learning models for thermal homography estimation in solar energy systems and close-range robotic inspection scenarios.

### Key Features

- ✅ **High Resolution**: 640×512 pixel thermal images (14-bit TIFF format)
- ✅ **Multi-Perspective Capture**: 4 heights (10, 20, 30, 40cm) × 2 angles (30°, 60°)
- ✅ **Temporal Variation**: 4 daily capture times (8 AM, 10 AM, 12 PM, 2 PM)
- ✅ **Environmental Diversity**: 5 cleanliness levels (clean → progressively soiled over 5 days)
- ✅ **Homography-Ready**: Synthetic pair generation for 99,680+ training pairs
- ✅ **Artifact Suppression**: Complete preprocessing pipeline for glare, shadow, and noise removal
- ✅ **Rich Metadata**: Comprehensive environmental measurements (irradiance, temperature, humidity, wind, soiling)
- ✅ **Production Code**: Complete preprocessing, generation, and validation scripts

## Dataset Statistics

| Metric | Value |
|--------|-------|
| **Raw Images** | 12,460 |
| **Resolution** | 640×512 pixels |
| **Bit Depth** | 14-bit |
| **Preprocessed Patches** | 49,840 (320×256) |
| **Homography Pairs** | 99,680 (synthetic) |
| **Collection Period** | 5 days (December 20-24, 2024) |
| **Capture Location** | Qatar Environment & Energy Research Institute (QEERI-OTF) |
| **Total Raw Data Size** | ~27 GB |
| **Preprocessed Data Size (partial)** | ~4 GB |

## Quick Installation

```bash
# Clone repository
git clone https://github.com/YaqoobAnsari/THED-PV.git
cd THED-PV

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Dataset Access

### Download from Zenodo

The complete dataset is permanently archived on Zenodo:

**DOI**: [10.5281/zenodo.17404246](https://doi.org/10.5281/zenodo.17404246)

```bash
# Download raw data (12,460 images, ~27 GB)
python scripts/download_data.py --type raw --output ./data/raw

# Download preprocessed data (49,840 patches, ~5 GB)
python scripts/download_data.py --type preprocessed --output ./data/preprocessed

# Download environmental metadata
python scripts/download_data.py --type metadata --output ./data/metadata
```

Full download instructions: [data/download_instructions.md](data/download_instructions.md)

## Usage

### Step 1: Preprocess Raw Images

```bash
python src/preprocess.py \
    --input_dir ./data/raw \
    --output_dir ./data/preprocessed \
    --config configs/preprocessing_config.yaml
```

**What this does:**
- CLAHE contrast enhancement (clip limit: 0.018, tile size: 8×8)
- Bilateral filtering for noise reduction (σ_spatial=8, σ_color=0.015)
- Adaptive glare suppression (threshold: mean + 2σ)
- Shadow correction via localized histogram normalization
- Image sharpening
- Quarters 640×512 images → four 320×256 patches

### Step 2: Generate Homography Pairs

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
- Ground truth homography matrices (3×3) stored as NumPy arrays
- 4-point parameterization for deep learning
- Metadata files with generation parameters (perturbation, overlap, coordinates)

**Note**: Generation takes approximately 6-8 hours for the full dataset on a standard workstation.

### Step 3: Split Dataset for Training

```bash
python src/split_dataset.py \
    --input_dir ./data/homography_pairs \
    --output_dir ./data/splits \
    --train_ratio 0.5 \
    --val_ratio 0.25 \
    --test_ratio 0.25 \
    --seed 42
```

**Creates:**
- Training set: 49,840 pairs (50%)
- Validation set: 24,920 pairs (25%)
- Testing set: 24,920 pairs (25%)
- Stratified across cleanliness levels and capture times

## Dataset Structure

```
data/
├── raw/                          # Original 14-bit TIFF images (42 GB)
│   ├── 2024-12-20/               # Day 1 (Clean)
│   ├── 2024-12-21/               # Day 2 (Soiled 1)
│   ├── 2024-12-22/               # Day 3 (Soiled 2)
│   ├── 2024-12-23/               # Day 4 (Soiled 3)
│   └── 2024-12-24/               # Day 5 (Soiled 4)
│       └── {08h00_30deg,10h00_60deg,12h00_30deg,14h00_60deg}/
│           └── {10cm,20cm,30cm,40cm}/
│               └── thermal_YYYYMMDD_HHMMSS_frameXXX.tiff
│
├── preprocessed/                 # 320×256 patches (38 GB)
│   └── [mirrors raw structure]
│       └── *_patch{0-3}.tiff
│
├── homography_pairs/             # Training-ready pairs (generate on-demand)
│   └── sample_XXXXX/
│       ├── patch_A.tiff          # Original patch (256×256)
│       ├── patch_B.tiff          # Warped patch (256×256)
│       ├── H_matrix.npy          # Ground truth homography (3×3)
│       └── metadata.txt          # Generation parameters
│
├── environmental_metadata/       # CSV files with 1-min resolution
│   ├── environmental_day1.csv
│   ├── environmental_day2.csv
│   ├── environmental_day3.csv
│   ├── environmental_day4.csv
│   └── environmental_day5.csv
│
└── splits/                       # Train/val/test split
    ├── train/
    ├── val/
    ├── test/
    └── reference_split.csv       # Reproducibility reference
```

See [docs/DATASET_STRUCTURE.md](docs/DATASET_STRUCTURE.md) for complete details.

## Environmental Metadata

Each thermal image is synchronized with comprehensive environmental measurements recorded at one-minute resolution:

| Parameter | Description | Units |
|-----------|-------------|-------|
| **Plane-of-Array Irradiance** | Via calibrated pyranometer | W/m² |
| **POA Short-Circuit Current** | Reference crystalline silicon cell | A |
| **Global Horizontal Irradiance** | Total solar radiation | W/m² |
| **Direct Normal Irradiance** | Direct beam radiation | W/m² |
| **Surface Albedo** | Ground reflectance | - |
| **Wind Speed & Direction** | Meteorological station | m/s, degrees |
| **Soiling Ratio** | Dual DustIQ optical sensors | % |
| **Ambient Temperature** | Air temperature | °C |
| **Relative Humidity** | Moisture content | % |
| **UV-A & UV-B Radiation** | Total ultraviolet | W/m² |

**Environmental Conditions During Collection** (December 20-24, 2024):
- Peak irradiance: 652-876 W/m²
- Temperature range: 16.9-21.8°C
- Humidity: 27-62%
- Wind speed: 1.1-3.3 m/s
- Progressive soiling: 18.9% → 20.4%
- Location: Doha, Qatar (25°19'37.36" N, 51°25'58.44" E)

## Example Workflows

### Training HomographyNet

```python
from hr_thermalpv import load_dataset, HomographyDataset
import torch

# Load dataset
train_data = HomographyDataset('./data/splits/train')
train_loader = torch.utils.data.DataLoader(
    train_data, 
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Your training loop here
for epoch in range(num_epochs):
    for patch_a, patch_b, homography_gt in train_loader:
        # Train your model
        pass
```
 
### Feature-Based Homography (ORB/SIFT)

```python
from hr_thermalpv.utils import compute_homography_orb
import cv2

img1 = cv2.imread('patch_A.tiff', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('patch_B.tiff', cv2.IMREAD_GRAYSCALE)

H, inliers, total = compute_homography_orb(img1, img2)
print(f"Inlier ratio: {inliers/total:.2%}")
print(f"Homography matrix:\n{H}")
```

### Environmental Metadata Integration

```python
from hr_thermalpv.utils import match_metadata
import pandas as pd

# Load thermal image
image_path = './data/raw/2024-12-20/08h00_30deg/10cm/thermal_20241220_080023_frame001.tiff'

# Automatically match with environmental data
metadata = match_metadata(image_path, './data/environmental_metadata/')

print(f"Irradiance: {metadata['poa_irradiance']} W/m²")
print(f"Temperature: {metadata['ambient_temp']}°C")
print(f"Soiling: {metadata['soiling_ratio']}%")
```

## Repository Structure

```
THED-PV/
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
│   ├── DATASET_STRUCTURE.md
│   ├── TECHNICAL_VALIDATION.md
│   └── USAGE_NOTES.md
└── scripts/                      # Utility scripts
    ├── download_data.py
    ├── verify_dataset.py
    └── evaluate_features.py
```

## Technical Validation

Our preprocessing and generation pipeline has been comprehensively validated:

### Preprocessing Quality Metrics

| Metric | Raw Images | Preprocessed | Improvement |
|--------|-----------|--------------|-------------|
| **PSNR (dB)** | 27.3-31.5 | Artifact-free | Glare/shadow removed |
| **SSIM** | - | 0.83-0.94 | Structure preserved |
| **ORB Keypoints** | 10-154 | 395-1545 | **10-15×** increase |
| **SIFT Keypoints** | 14-154 | 150-823 | **5-10×** increase |
| **Entropy** | 16.75-17.31 | 17.25-17.76 | +2.5-3.0% |

### Geometric Preservation

| Validation Metric | Mean ± Std | Maximum |
|-------------------|-----------|---------|
| **Corner Displacement** | 0.09 ± 0.04 px | 0.23 px |
| **Homography Matrix Difference** | 0.0021 ± 0.0008 | 0.0047 |
| **Angular Preservation Error** | 0.31 ± 0.18° | 0.82° |

### Feature Matching Performance

| Day | ORB Inlier Ratio | SIFT Inlier Ratio |
|-----|------------------|-------------------|
| Day 1 (Clean) | 0.92 ± 0.03 | 0.55 ± 0.07 |
| Day 2 | 0.94 ± 0.02 | 0.61 ± 0.06 |
| Day 3 | 0.96 ± 0.02 | 0.70 ± 0.05 |
| Day 4 | 0.97 ± 0.01 | 0.78 ± 0.04 |
| Day 5 (Soiled) | 0.98 ± 0.01 | 0.83 ± 0.03 |

**Key Finding**: Feature matching performance improves progressively over the five-day period as panel soiling increases surface texture, demonstrating that THED-PV captures the full spectrum from feature-sparse (clean) to feature-rich (soiled) conditions.

### HomographyNet Benchmarking

| Day | Mean Average Corner Error (MACE) |
|-----|-----------------------------------|
| Day 1 | 16.65 px |
| Day 2 | 14.32 px |
| Day 3 | 12.15 px |
| Day 4 | 10.48 px |
| Day 5 | 9.60 px |

**Time-of-Day Analysis**:
- Best performance: 12 PM (7.71 px MACE) - optimal thermal contrast
- Worst performance: 10 AM (19.52 px MACE) - transitional glare conditions

See [docs/TECHNICAL_VALIDATION.md](docs/TECHNICAL_VALIDATION.md) for detailed validation methodology.

## Synthetic Pair Generation Details

The dataset generator creates homography pairs through a systematic process:

1. **Patch Selection**: Random 256×256 patches with Gaussian center bias (0.2)
2. **Geometric Perturbation**: Corner displacement of 15-50 pixels
3. **Spatial Translation**: Overlap ratio maintained at 55-85%
4. **Homography Computation**: Direct Linear Transformation for ground truth
5. **Validation**: Automatic rejection of boundary violations (~10-11%)

**Generation Statistics** (5-day dataset):

| Day | Initial Attempts | Rejected | Valid Pairs |
|-----|-----------------|----------|-------------|
| Day 1 | 26,468 | 2,629 (9.9%) | 23,839 |
| Day 2 | 26,512 | 2,619 (9.9%) | 23,893 |
| Day 3 | 26,488 | 2,620 (9.9%) | 23,868 |
| Day 4 | 26,509 | 2,661 (10.0%) | 23,848 |
| Day 5 | 26,498 | 2,633 (9.9%) | 23,865 |
| **Total** | **132,475** | **13,162 (9.9%)** | **119,313** |

## Citation

If you use THED-PV in your research, please cite:

```bibtex
@article{yaqoob2025thedpv,
  title={THED-PV: A High-Resolution Multi-Perspective Thermal Imaging Dataset for Photovoltaic Homography Estimation},
  author={Yaqoob, Mohammed and Ansari, Mohammed Yusuf and Pillai, Dhanup Somasekharan and Flushing, Eduardo Feo},
  journal={Nature Scientific Data},
  year={2025},
  doi={10.5281/zenodo.17404246},
  url={https://doi.org/10.5281/zenodo.17404246}
}
```

## License

This dataset is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

**You are free to:**
- ✅ Share and redistribute in any medium or format
- ✅ Adapt, remix, transform, and build upon the material
- ✅ Use for any purpose, even commercially

**Under the following terms:**
- 📝 **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made
- 🔓 **No additional restrictions**: You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits

## Acknowledgements

This work was supported by the AICC grant AICC04-0715-210006 from the Qatar National Research Fund. Data collection was conducted at the Qatar Environment and Energy Research Institute (QEERI) Outdoor Test Facility.

**Special thanks to:**
- QEERI-OTF team for facility access and technical support
- Carnegie Mellon University Qatar for computational resources
- Qatar National Research Fund for funding support

## Contact & Support

- **Primary Contact**: Mohammed Yaqoob - yansari@andrew.cmu.edu
- **Issues & Questions**: [GitHub Issues](https://github.com/YaqoobAnsari/THED-PV/issues)
- **Documentation**: [Full Documentation](https://YaqoobAnsari.github.io/THED-PV/)
- **Dataset Repository**: [Zenodo](https://doi.org/10.5281/zenodo.17404246)

## Limitations and Considerations

### Dataset Characteristics

1. **Single Installation Type**: Polycrystalline silicon panels in 2L landscape configuration; generalization to other panel types may require domain adaptation
2. **Seasonal Limitation**: Winter conditions only (December 2024); seasonal temperature variations not represented
3. **Geographic Specificity**: Desert climate (Doha, Qatar); may not generalize to humid or temperate climates
4. **Perturbation Range**: Synthetic pairs use 15-50 pixel perturbations suitable for close-range inspection; may not cover wide-baseline scenarios
5. **Collection Anomaly**: Day 5 shows reduced soiling (17.6% vs expected 21%), possibly due to morning dew

### Recommendations

- Validate models on sequential real captures when available
- Consider domain adaptation for different panel types or climates
- Use reference train/val/test split for benchmarking consistency
- Augment training with motion blur and photometric variations for robustness

See [docs/USAGE_NOTES.md](docs/USAGE_NOTES.md) for detailed guidance.

## Changelog

**v1.0.0** (2025-01-XX)
- Initial public release
- 12,460 raw thermal images (640×512, 14-bit)
- Complete preprocessing pipeline
- Synthetic homography pair generation
- Environmental metadata integration
- Reference train/val/test splits
- Comprehensive validation experiments

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Submitting bug reports
- Proposing new features
- Contributing code improvements
- Sharing research results

## Related Publications

Stay updated with research using THED-PV:
- [Publications Wiki](https://github.com/YaqoobAnsari/THED-PV/wiki/Publications)

---

**Keywords**: Thermal Imaging, Photovoltaic Systems, Homography Estimation, Deep Learning, Computer Vision, Solar Energy, Fault Detection, Dataset, Benchmark, Robotic Inspection

**Last Updated**: January 2025
