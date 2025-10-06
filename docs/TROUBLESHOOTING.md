# HR-ThermalPV Troubleshooting Guide

## Installation Issues

### Problem: `pip install -e .` fails

**Symptom:**
```
ERROR: Could not find a version that satisfies the requirement opencv-python
```

**Solution:**
```bash
# Update pip
python -m pip install --upgrade pip

# Install OpenCV separately
pip install opencv-python opencv-contrib-python

# Try again
pip install -e .
```

### Problem: Import errors after installation

**Symptom:**
```python
ModuleNotFoundError: No module named 'preprocess'
```

**Solution:**
```bash
# Ensure you're in the repository root
cd HR-ThermalPV

# Install in development mode
pip install -e .

# Verify installation
python -c "import sys; print(sys.path)"
```

---

## Preprocessing Issues

### Problem: "Failed to read image" error

**Symptom:**
```
ValueError: Failed to read image: /path/to/image.tiff
```

**Possible Causes & Solutions:**

1. **File doesn't exist:**
   ```bash
   ls -lh /path/to/image.tiff  # Check if file exists
   ```

2. **Corrupted TIFF file:**
   ```bash
   # Verify with imagemagick
   identify /path/to/image.tiff
   ```

3. **Permission issues:**
   ```bash
   chmod +r /path/to/image.tiff
   ```

4. **Wrong OpenCV version:**
   ```bash
   pip install --upgrade opencv-python==4.8.0.76
   ```

### Problem: Preprocessing runs out of memory

**Symptom:**
```
MemoryError: Unable to allocate array
```

**Solution:**

Process in batches:
```python
# Modify preprocess.py to process in smaller batches
# Or reduce image resolution temporarily for testing

# Check available memory
import psutil
print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")
```

**Minimum Requirements:**
- RAM: 8 GB
- Disk Space: 50 GB free

### Problem: CLAHE produces artifacts

**Symptom:**
Blocky patterns or over-enhanced regions in preprocessed images.

**Solution:**

Adjust CLAHE parameters in `configs/preprocessing_config.yaml`:

```yaml
clahe:
  clip_limit: 0.01    # Lower value (was 0.018)
  tile_grid_size: [16, 16]  # Larger tiles (was [8, 8])
```

### Problem: Glare not properly suppressed

**Symptom:**
Bright spots remain after preprocessing.

**Solution:**

```yaml
glare:
  threshold_factor: 1.5  # More aggressive (was 2.0)
```

---

## Homography Generation Issues

### Problem: "Failed to generate valid pair after 100 attempts"

**Symptom:**
Many images fail to produce valid homography pairs.

**Possible Causes:**

1. **Image too small:**
   - Ensure preprocessed images are 320×256
   - Patch size must fit in image

2. **Extreme perturbations:**
   ```bash
   # Reduce perturbation range
   python src/generate_homography_pairs.py \
       --perturbation_range 10 30  # Reduced from 15 50
   ```

3. **Too strict overlap requirements:**
   ```bash
   --overlap_range 0.4 0.9  # Wider range
   ```

### Problem: Black padding in generated patches

**Symptom:**
`patch_B.tiff` contains black regions.

**Solution:**

This is automatically filtered by the generator. If you see this:

1. **Check generator validation:**
   ```python
   # In generate_homography_pairs.py
   # validate_pair() should catch this
   ```

2. **Reduce perturbation:**
   Lower perturbation keeps patches within bounds more easily.

### Problem: Very low inlier ratios

**Symptom:**
When testing with ORB/SIFT, inlier ratio < 0.5

**Possible Causes:**

1. **Poor image quality:**
   - Check preprocessing output
   - Verify features are detected: `python -c "from utils import count_features; ..."

`

2. **Too much geometric distortion:**
   - Reduce perturbation range
   - Increase overlap ratio

---

## Dataset Splitting Issues

### Problem: Unequal split ratios

**Symptom:**
Actual splits don't match requested ratios.

**Solution:**

This is normal due to stratification. Check `split_summary.json`:

```json
{
  "actual_ratios": {
    "train": 0.501,  // Close to 0.5
    "val": 0.248,    // Close to 0.25
    "test": 0.251    // Close to 0.25
  }
}
```

Small deviations (< 2%) are expected.

### Problem: Files not copied to split directories

**Symptom:**
Split directories are empty or incomplete.

**Solution:**

```bash
# Check disk space
df -h

# Check permissions
ls -la data/splits/

# Re-run with verbose logging
python src/split_dataset.py --input_dir ... --output_dir ... -v
```

---

## Data Download Issues

### Problem: Zenodo download fails

**Symptom:**
```
ConnectionError: HTTPSConnectionPool
```

**Solution:**

1. **Resume download:**
   ```bash
   wget -c https://zenodo.org/record/XXXXXX/files/hr-thermalpv-raw.zip
   ```

2. **Use alternative download method:**
   ```bash
   curl -L -C - -O https://zenodo.org/...
   ```

3. **Check internet connection:**
   ```bash
   ping zenodo.org
   ```

### Problem: Checksum verification fails

**Symptom:**
```
ERROR: Checksum mismatch for file X
```

**Solution:**

```bash
# Re-download the specific file
rm corrupted_file.zip
wget https://zenodo.org/.../corrupted_file.zip

# Manual verification
md5sum file.zip
# Compare with Zenodo's published checksums
```

---

## Feature Extraction Issues

### Problem: No features detected

**Symptom:**
```python
count_features(img, 'orb')  # Returns 0
```

**Solution:**

1. **Check image contrast:**
   ```python
   import cv2
   import numpy as np
   
   img = cv2.imread('image.tiff', cv2.IMREAD_GRAYSCALE)
   print(f"Min: {img.min()}, Max: {img.max()}, Std: {img.std()}")
   
   # If std < 10, image has very low contrast
   ```

2. **Preprocess image:**
   - Ensure CLAHE was applied
   - Check if image is blank (all zeros)

3. **Adjust detector parameters:**
   ```python
   orb = cv2.ORB_create(
       nfeatures=5000,  # Increase from 2000
       scaleFactor=1.1,
       nlevels=12
   )
   ```

### Problem: ORB/SIFT inlier ratio very low

**Symptom:**
Inlier ratio < 0.3 on valid pairs.

**Solution:**

1. **Check match ratio threshold:**
   ```python
   H, inliers, total = compute_homography_orb(
       img1, img2, 
       match_ratio=0.8  # Less strict (was 0.75)
   )
   ```

2. **Verify images are similar:**
   ```python
   # Compute SSIM
   from skimage.metrics import structural_similarity
   ssim = structural_similarity(img1, img2)
   print(f"SSIM: {ssim:.3f}")  # Should be > 0.3
   ```

---

## Performance Issues

### Problem: Preprocessing is very slow

**Symptom:**
< 1 image per second on modern hardware.

**Solution:**

1. **Enable parallel processing:**
   ```yaml
   # In preprocessing_config.yaml
   processing:
     parallel: true
     num_workers: 4
   ```

2. **Profile bottlenecks:**
   ```python
   import cProfile
   cProfile.run('preprocessor.process_file(...)')
   ```

3. **Use SSD instead of HDD:**
   I/O is often the bottleneck.

### Problem: Homography generation is slow

**Symptom:**
< 10 pairs per minute.

**Solution:**

```python
# Reduce max_attempts
generator = HomographyPairGenerator(max_attempts=50)  # Was 100

# Or reduce quality checks
# (Not recommended for training data)
```

---

## File Format Issues

### Problem: "Not a valid TIFF file"

**Symptom:**
```
cv2.error: Unable to read TIFF file
```

**Solution:**

```bash
# Check file type
file image.tiff

# Convert if needed
convert image.png image.tiff  # Using ImageMagick

# Or with Python
from PIL import Image
img = Image.open('image.png')
img.save('image.tiff', 'TIFF')
```

### Problem: NumPy array dtype mismatch

**Symptom:**
```
TypeError: Cannot cast array data from dtype('uint16') to dtype('uint8')
```

**Solution:**

```python
# Proper conversion
img_8bit = (img_16bit / 256).astype(np.uint8)  # For 14-bit

# Or use normalize_to_8bit()
from preprocess import ThermalPreprocessor
preprocessor = ThermalPreprocessor()
img_8bit = preprocessor.normalize_to_8bit(img_16bit)