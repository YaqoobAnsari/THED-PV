# HR-ThermalPV Dataset Download Instructions

## Option 1: Direct Download from Zenodo

### Raw Data (~XX GB)

```bash
# Using wget
wget https://zenodo.org/record/XXXXXX/files/hr-thermalpv-raw.zip

# Or using curl
curl -O https://zenodo.org/record/XXXXXX/files/hr-thermalpv-raw.zip

# Extract
unzip hr-thermalpv-raw.zip -d ./data/raw/
```

### Preprocessed Data (~XX GB)

```bash
wget https://zenodo.org/record/XXXXXX/files/hr-thermalpv-preprocessed.zip
unzip hr-thermalpv-preprocessed.zip -d ./data/preprocessed/
```

### Homography Pairs (~XX GB)

```bash
wget https://zenodo.org/record/XXXXXX/files/hr-thermalpv-homography-pairs.zip
unzip hr-thermalpv-homography-pairs.zip -d ./data/homography_pairs/
```

## Option 2: Using Python Script

```bash
# Install zenodo-get (if not already installed)
pip install zenodo-get

# Download raw data
python scripts/download_data.py --type raw --output ./data/raw

# Download preprocessed data
python scripts/download_data.py --type preprocessed --output ./data/preprocessed

# Download all
python scripts/download_data.py --type all --output ./data/
```

## Dataset Structure After Download

```
data/
├── raw/                          # 12,460 images (640×512, 14-bit)
│   ├── 2024-12-21/              # Day 1 - Clean panels
│   ├── 2024-12-22/              # Day 2 - Soiled 1
│   ├── 2024-12-23/              # Day 3 - Soiled 2
│   ├── 2024-12-24/              # Day 4 - Soiled 3
│   └── 2024-12-25/              # Day 5 - Soiled 4
│       └── {8am,10am,12pm,2pm}/
│           └── {10cm,20cm,30cm,40cm}/
│               ├── YYYYMMDD_HHMMSS_angle30.tiff
│               └── YYYYMMDD_HHMMSS_angle60.tiff
│
├── preprocessed/                 # 49,840 patches (320×256, 8-bit)
│   └── [same structure as raw, with _patch{0-3}.tiff suffix]
│
└── homography_pairs/             # 99,680 pairs
    └── sample_XXXXXX/
        ├── patch_A.tiff
        ├── patch_B.tiff
        ├── homography_matrix.npy
        ├── 4point_params.npy
        └── metadata.json
```

## Verify Download Integrity

```bash
# After download, verify checksums
python scripts/verify_dataset.py --data_dir ./data/raw
python scripts/verify_dataset.py --data_dir ./data/preprocessed
```

## Storage Requirements

| Dataset Version | Compressed | Uncompressed |
|----------------|-----------|--------------|
| Raw            | ~XX GB    | ~XX GB       |
| Preprocessed   | ~XX GB    | ~XX GB       |
| Homography Pairs | ~XX GB | ~XX GB       |
| **Total**      | **~XX GB** | **~XX GB** |

## Troubleshooting

### Download Interrupted

```bash
# Resume with wget
wget -c https://zenodo.org/record/XXXXXX/files/hr-thermalpv-raw.zip

# Or with curl
curl -C - -O https://zenodo.org/record/XXXXXX/files/hr-thermalpv-raw.zip
```

### Checksum Verification Failed

1. Re-download the specific file
2. Check available disk space
3. Verify network connection stability
4. Contact dataset maintainers if issue persists

### Extraction Issues

```bash
# For corrupted zip files, try 7zip
sudo apt-get install p7zip-full
7z x hr-thermalpv-raw.zip -o./data/raw/
```

## Citation

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

## License

This dataset is provided under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Support

For issues or questions:
- GitHub Issues: https://github.com/YOUR_USERNAME/HR-ThermalPV/issues
- Email: yansari@tamu.edu