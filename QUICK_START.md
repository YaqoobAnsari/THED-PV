# Quick Start Guide - THED-PV New Scripts

This guide helps you quickly integrate and test the three new scripts in your THED-PV repository.

## 📦 Files Provided

1. **match_metadata.py** (18 KB) - Metadata matching
2. **evaluate_features.py** (22 KB) - Feature evaluation
3. **visualize_pairs.py** (25 KB) - Visualization generation
4. **download_data.py** (9.8 KB) - Updated dataset downloader
5. **download_instructions.md** (11 KB) - Updated download guide
6. **IMPLEMENTATION_SUMMARY.md** (15 KB) - Complete documentation

---

## 🚀 Quick Integration Steps

### Step 1: Clone Your Repository (if not done)
```bash
git clone https://github.com/YaqoobAnsari/THED-PV.git
cd THED-PV
```

### Step 2: Copy New Scripts to Repository
```bash
# Copy the three main scripts to src/ directory
cp /path/to/match_metadata.py src/
cp /path/to/evaluate_features.py src/
cp /path/to/visualize_pairs.py src/

# Copy updated download files
cp /path/to/download_data.py scripts/
cp /path/to/download_instructions.md data/
```

### Step 3: Update requirements.txt
Add these dependencies if not already present:
```bash
echo "opencv-python>=4.5.0" >> requirements.txt
echo "pandas>=1.3.0" >> requirements.txt
echo "numpy>=1.21.0" >> requirements.txt
echo "scipy>=1.7.0" >> requirements.txt
echo "matplotlib>=3.4.0" >> requirements.txt
echo "requests>=2.28.0" >> requirements.txt
echo "tqdm>=4.64.0" >> requirements.txt
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🧪 Testing the Scripts

### Test 1: match_metadata.py

Create a test with sample data:
```bash
# If you have data already
python src/match_metadata.py \
  --image_dir ./data/raw/2024-12-20/08h00_30deg/10cm \
  --metadata_dir ./data/environmental_metadata \
  --summary \
  --output test_matched.csv

# Check output
head test_matched.csv
```

**Expected behavior:**
- Loads metadata CSV files
- Prints summary of available dates
- Matches images with timestamps
- Creates CSV with matched data

### Test 2: evaluate_features.py

Test with a sample image pair:
```bash
# If you have homography pairs
python src/evaluate_features.py \
  --image1 ./data/homography_pairs/sample_00001/patch_A.tiff \
  --image2 ./data/homography_pairs/sample_00001/patch_B.tiff \
  --detector ORB

# Or compare multiple detectors
python src/evaluate_features.py \
  --image1 ./data/homography_pairs/sample_00001/patch_A.tiff \
  --image2 ./data/homography_pairs/sample_00001/patch_B.tiff \
  --compare
```

**Expected behavior:**
- Detects keypoints in both images
- Matches features
- Estimates homography
- Prints evaluation metrics

### Test 3: visualize_pairs.py

Create visualizations:
```bash
# Single pair visualization
python src/visualize_pairs.py \
  --pair_dir ./data/homography_pairs/sample_00001 \
  --output_dir ./test_visualizations \
  --detector ORB

# Check output
ls -lh ./test_visualizations/
```

**Expected behavior:**
- Creates 5 visualization files:
  - `sample_00001_side_by_side.png`
  - `sample_00001_keypoints.png`
  - `sample_00001_matches.png`
  - `sample_00001_warp.png`
  - `sample_00001_grid.png`

---

## 🐛 Troubleshooting Quick Fixes

### Issue: "Module not found" error
```bash
# Reinstall dependencies
pip install opencv-python pandas numpy scipy matplotlib

# Or use conda
conda install -c conda-forge opencv pandas numpy scipy matplotlib
```

### Issue: "No metadata files found"
```bash
# Check directory structure
ls -la ./data/environmental_metadata/

# The script expects CSV files named:
# - environmental_day1.csv
# - environmental_day2.csv
# Or any CSV files with timestamp columns
```

### Issue: "Failed to load images"
```bash
# Verify image paths
ls -la ./data/homography_pairs/sample_00001/

# Should contain:
# - patch_A.tiff
# - patch_B.tiff
# - H_matrix.npy (optional)
# - metadata.txt (optional)
```

### Issue: OpenCV error with SIFT/SURF
```bash
# Install opencv-contrib instead
pip uninstall opencv-python
pip install opencv-contrib-python
```

---

## 📊 Quick Validation Checklist

Before committing, verify:

- [ ] All scripts have execute permissions: `chmod +x src/*.py`
- [ ] Scripts run without errors on sample data
- [ ] Help messages work: `python src/match_metadata.py --help`
- [ ] Output files are created in expected locations
- [ ] CSV outputs have correct columns
- [ ] Visualizations are generated properly
- [ ] No syntax errors: `python -m py_compile src/*.py`
- [ ] Requirements.txt is updated
- [ ] Documentation is clear

---

## 📝 Git Commit Template

Use this template for your commit:

```bash
git add src/match_metadata.py
git add src/evaluate_features.py
git add src/visualize_pairs.py
git add scripts/download_data.py
git add data/download_instructions.md

git commit -m "Add missing utility scripts for metadata matching, feature evaluation, and visualization

- Add match_metadata.py: Automatic temporal alignment of thermal images with environmental measurements
  * Supports multiple interpolation methods (nearest, linear, forward, backward)
  * Configurable time tolerance for matching
  * Batch processing for entire directories
  * CSV output with matched environmental data

- Add evaluate_features.py: Feature matching metrics for classical homography methods
  * Supports ORB, SIFT, AKAZE, BRISK, KAZE detectors
  * Calculates inlier ratios, keypoint counts, reprojection errors
  * Batch evaluation with statistical summaries
  * Detector comparison functionality

- Add visualize_pairs.py: Comprehensive diagnostic visualizations
  * Side-by-side image comparison
  * Keypoint detection overlay
  * Feature correspondence lines (inliers/outliers)
  * Homography warping with difference maps
  * Grid overlay for geometric distortion analysis

- Update download_data.py: Fix Zenodo record ID and API integration
  * Correct record: 17404247
  * Dynamic file fetching via Zenodo API
  * Progress tracking and checksum verification
  * Support for selective download by type

- Update download_instructions.md: Comprehensive download guide
  * Three download methods (script, browser, CLI tools)
  * Troubleshooting section
  * Batch processing examples
  * Updated file sizes and structure

All scripts:
- Python 3.8+ compatible
- Comprehensive CLI with --help
- Error handling and validation
- Batch processing support
- Well-documented with docstrings

Closes #XX (if there's an issue)"
```

---

## 🔄 Workflow Example

Here's a complete workflow using all scripts:

```bash
#!/bin/bash

# 1. Download dataset
python scripts/download_data.py \
  --output ./data \
  --type all

# 2. Preprocess (assuming existing script)
python src/preprocess.py \
  --input_dir ./data/raw \
  --output_dir ./data/preprocessed \
  --config configs/preprocessing_config.yaml

# 3. Match with metadata
python src/match_metadata.py \
  --image_dir ./data/preprocessed \
  --metadata_dir ./data/environmental_metadata \
  --recursive \
  --output ./data/matched_metadata.csv

# 4. Generate homography pairs (assuming existing script)
python src/generate_homography_pairs.py \
  --input_dir ./data/preprocessed \
  --output_dir ./data/homography_pairs \
  --num_pairs 2

# 5. Split dataset (assuming existing script)
python src/split_dataset.py \
  --input_dir ./data/homography_pairs \
  --output_dir ./data/splits \
  --train_ratio 0.5 \
  --val_ratio 0.25 \
  --test_ratio 0.25

# 6. Evaluate classical methods
echo "Evaluating ORB..."
python src/evaluate_features.py \
  --dataset_dir ./data/splits/test \
  --detector ORB \
  --output ./results/orb_results.csv

echo "Evaluating SIFT..."
python src/evaluate_features.py \
  --dataset_dir ./data/splits/test \
  --detector SIFT \
  --output ./results/sift_results.csv

# 7. Create visualizations
python src/visualize_pairs.py \
  --dataset_dir ./data/splits/test \
  --output_dir ./visualizations \
  --max_pairs 50 \
  --detector ORB

echo "Pipeline complete!"
```

---

## 📖 Next Steps

1. **Review the IMPLEMENTATION_SUMMARY.md** for detailed documentation
2. **Test each script individually** with your data
3. **Create example Jupyter notebooks** demonstrating usage
4. **Update main README.md** to mention new scripts
5. **Add unit tests** in the tests/ directory
6. **Run through complete pipeline** to ensure integration

---

## 💡 Tips for Success

### For match_metadata.py:
- Ensure environmental CSV files have clear timestamp columns
- Check that image filenames follow consistent naming conventions
- Use `--summary` flag first to verify metadata is loaded correctly

### For evaluate_features.py:
- Start with ORB (fastest) for quick testing
- Use SIFT for publication-quality benchmarks
- Try `--compare` mode to see which detector works best for your data

### For visualize_pairs.py:
- Use `--max_pairs 5` first to test quickly
- Increase `--dpi 300` for publication figures
- Check that pair directories have all required files

### General:
- Always use `--help` to see all available options
- Start with small datasets for testing
- Check log output for warnings or errors
- Verify output files are created correctly

---

## ✅ Pre-Push Checklist

Before pushing to GitHub:

```bash
# 1. Syntax check
python -m py_compile src/match_metadata.py
python -m py_compile src/evaluate_features.py
python -m py_compile src/visualize_pairs.py

# 2. Run help to verify CLI works
python src/match_metadata.py --help
python src/evaluate_features.py --help
python src/visualize_pairs.py --help

# 3. Quick functional test (if data available)
# [Run appropriate test commands from above]

# 4. Check file permissions
chmod +x src/match_metadata.py
chmod +x src/evaluate_features.py
chmod +x src/visualize_pairs.py

# 5. Verify git status
git status

# 6. Review changes
git diff src/

# 7. Commit and push
git commit -m "Add missing utility scripts"
git push origin main
```

---

## 📞 Getting Help

If you encounter issues:

1. **Check the IMPLEMENTATION_SUMMARY.md** for detailed docs
2. **Review error messages carefully** - they often indicate the issue
3. **Verify dependencies** are installed: `pip list | grep opencv`
4. **Test with minimal example** before full dataset
5. **Open GitHub issue** with:
   - Error message
   - Command used
   - Python version (`python --version`)
   - Operating system

---

**Created**: January 2025  
**Status**: Ready for integration ✅  
**Python**: 3.8+