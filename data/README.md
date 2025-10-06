# HR-ThermalPV Data Directory

## Overview

This directory is where you should place or download the HR-ThermalPV dataset files. **The actual data is NOT included in this repository** due to size constraints (GitHub has a 100GB repository limit and individual file size limits).

## Directory Structure

After downloading the dataset, your `data/` directory should look like this:

```
data/
├── README.md                    # This file
├── download_instructions.md     # Download guide
├── raw/                         # Raw 14-bit thermal images (12,460 images)
│   ├── 2024-12-21/             # Day 1 - Clean panels
│   ├── 2024-12-22/             # Day 2 - Soiled Level 1
│   ├── 2024-12-23/             # Day 3 - Soiled Level 2
│   ├── 2024-12-24/             # Day 4 - Soiled Level 3
│   └── 2024-12-25/             # Day 5 - Soiled Level 4
├── preprocessed/                # Preprocessed 8-bit patches (49,840 patches)
├── homography_pairs/            # Generated training pairs (99,680 pairs)
└── splits/                      # Train/val/test splits
    ├── train/
    ├── val/
    └── test/
```

## Download Options

### Option 1: Download from Zenodo (Recommended)

The complete dataset is hosted on Zenodo at:
- **DOI**: [10.5281/zenodo.XXXXXX](https://doi.org/10.5281/zenodo.XXXXXX)

```bash
# Download raw data
python scripts/download_data.py --type raw --output ./data/raw

# Download preprocessed data
python scripts/download_data.py --type preprocessed --output ./data/preprocessed

# Download all
python scripts/download_data.py --type all --output ./data/
```

See [download_instructions.md](download_instructions.md) for detailed instructions.

### Option 2: Generate from Raw Data

If you only download the raw data, you can generate the preprocessed and homography pair datasets:

```bash
# Step 1: Preprocess raw images
python src/preprocess.py \
    --input_dir ./data/raw \
    --output_dir ./data/preprocessed \
    --config configs/preprocessing_config.yaml

# Step 2: Generate homography pairs
python src/generate_homography_pairs.py \
    --input_dir ./data/preprocessed \
    --output_dir ./data/homography_pairs \
    --num_pairs 2

# Step 3: Split dataset
python src/split_dataset.py \
    --input_dir ./data/homography_pairs \
    --output_dir ./data/splits \
    --train_ratio 0.5 --val_ratio 0.25 --test_ratio 0.25
```

## Storage Requirements

| Component | Compressed Size | Uncompressed Size |
|-----------|----------------|-------------------|
| Raw Data | ~8 GB | ~8 GB |
| Preprocessed | ~4 GB | ~4 GB |
| Homography Pairs | ~20 GB | ~20 GB |
| **Total** | **~32 GB** | **~32 GB** |

Ensure you have sufficient disk space before downloading.

## Data Formats

### Raw Images
- **Format**: 14-bit TIFF
- **Resolution**: 640 × 512 pixels
- **Naming**: `YYYYMMDD_HHMMSS_angle{30|60}.tiff`

### Preprocessed Images
- **Format**: 8-bit TIFF
- **Resolution**: 320 × 256 pixels (quartered)
- **Naming**: `YYYYMMDD_HHMMSS_angle{30|60}_patch{0-3}.tiff`

### Homography Pairs
Each sample directory contains:
- `patch_A.tiff` - Original patch (256×256)
- `patch_B.tiff` - Warped patch (256×256)
- `homography_matrix.npy` - 3×3 homography matrix
- `4point_params.npy` - 8-element displacement vector
- `metadata.json` - Generation parameters

## Verification

After downloading, verify data integrity:

```bash
python scripts/verify_dataset.py --data_dir ./data/raw
python scripts/verify_dataset.py --data_dir ./data/preprocessed
```

This checks:
- File counts match expected
- Image dimensions are correct
- No corrupted files
- Proper file naming conventions

## Important Notes

⚠️ **DO NOT commit data files to Git**

The `.gitignore` file is configured to exclude all data directories. Data should only be:
1. Downloaded from Zenodo
2. Generated locally using the provided scripts

⚠️ **Keep raw data as backup**

Preprocessed and homography pair datasets can be regenerated from raw data. Always keep the raw data as your source of truth.

## License

The HR-ThermalPV dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

When using this dataset, please cite:

```bibtex
@article{yaqoob2025hrthermalpv,
  title={HR-ThermalPV: A High-Resolution Multi-Perspective Thermal Imaging Dataset for Photovoltaic Homography Estimation},
  author={Yaqoob, Mohammed and Ansari, Mohammed Yusuf and Pillai, Dhanup Somasekharan and Flushing, Eduardo Feo},
  journal={Nature Scientific Data},
  year={2025},
  doi={XX.XXXX/XXXXX}
}
```

## Support

For issues with data download or processing:
- Check [TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md)
- Open an issue on [GitHub](https://github.com/YOUR_USERNAME/HR-ThermalPV/issues)
- Email: yansari@tamu.edu