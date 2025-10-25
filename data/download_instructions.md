# THED-PV Dataset Download Instructions

This document provides comprehensive instructions for downloading the THED-PV dataset from Zenodo.

## Dataset Information

- **Zenodo Record ID**: `17404247`
- **DOI**: `10.5281/zenodo.17404247`
- **Direct Link**: [https://zenodo.org/records/17404247](https://zenodo.org/records/17404247)
- **Total Size**: ~26.6 GB
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)

## Dataset Contents

The THED-PV dataset consists of the following components:

| Component | Description | Approximate Size |
|-----------|-------------|------------------|
| **Raw Thermal Images** | 12,460 high-resolution thermal images (640×512, 14-bit TIFF) | ~22-24 GB |
| **Preprocessed Patches** | 49,840 preprocessed patches (320×256) | ~2-3 GB |
| **Environmental Metadata** | CSV files with environmental measurements | ~50-100 MB |
| **Code.zip** | Complete preprocessing and generation scripts | ~50-100 MB |

**Note**: File sizes are approximate and may vary slightly from the exact values on Zenodo.

---

## Download Methods

There are three primary methods to download the dataset:

1. **Automated Script (Recommended)** - Using our provided Python script
2. **Direct Browser Download** - Manual download from Zenodo website
3. **Command-line Tools** - Using `zenodo_get` or `wget`

---

## Method 1: Automated Script (Recommended)

Our provided `download_data.py` script handles downloading, progress tracking, and checksum verification automatically.

### Prerequisites

```bash
# Install required packages
pip install requests tqdm
```

### Usage

#### Download All Files (~26.6 GB)

```bash
python scripts/download_data.py --output ./data --type all
```

#### Download Specific Components

**Raw thermal images only:**
```bash
python scripts/download_data.py --output ./data/raw --type raw
```

**Preprocessed patches only:**
```bash
python scripts/download_data.py --output ./data/preprocessed --type preprocessed
```

**Environmental metadata and code:**
```bash
python scripts/download_data.py --output ./data/metadata --type metadata
```

**Code/scripts only:**
```bash
python scripts/download_data.py --output ./data --type code
```

### Script Features

- ✅ **Progress tracking** with detailed progress bars
- ✅ **Automatic checksum verification** (MD5)
- ✅ **Resume capability** - skips already downloaded files
- ✅ **Error handling** with automatic retry suggestions
- ✅ **Size validation** before download

### Command-line Options

```bash
python scripts/download_data.py --help
```

Options:
- `--output DIR` : Output directory (required)
- `--type TYPE` : File type to download (all/raw/preprocessed/metadata/code)
- `--force` : Force re-download even if files exist

---

## Method 2: Direct Browser Download

### Step-by-Step Instructions

1. **Visit the Zenodo record page:**
   - URL: [https://zenodo.org/records/17404247](https://zenodo.org/records/17404247)

2. **Review available files:**
   - Scroll to the "Files" section
   - Review file names and sizes

3. **Download files:**
   - Click on individual files to download them
   - **OR** Click "Download all" to download the entire dataset as a single archive

4. **Verify downloads:**
   - Check file sizes match those listed on Zenodo
   - Optionally verify MD5 checksums (provided on Zenodo page)

### Browser Download Tips

- **Large files**: For files larger than 1 GB, use a download manager that supports resume capability (e.g., Free Download Manager, JDownloader)
- **Slow connection**: Download individual files separately rather than using "Download all"
- **Multiple attempts**: If download fails, refresh the page and retry

---

## Method 3: Command-line Tools

### Option A: Using `zenodo_get`

`zenodo_get` is a dedicated tool for downloading Zenodo records.

#### Installation

```bash
# Using pip
pip install zenodo-get

# Using conda/mamba
conda install -c conda-forge zenodo_get
```

#### Usage

```bash
# Download entire record
zenodo_get 17404247 -o ./data

# Download with MD5 verification
zenodo_get 17404247 -o ./data -m

# Download specific files using glob patterns
zenodo_get 17404247 -o ./data -g "*.zip"
zenodo_get 17404247 -o ./data -g "*raw*,*metadata*"
```

#### zenodo_get Options

- `-o DIR` : Output directory
- `-m` : Generate and verify MD5 checksums
- `-g PATTERN` : Download only files matching glob pattern
- `-w FILE` : Generate URL list instead of downloading
- `-e` : Continue on error
- `-n` : Do not resume, force new download

### Option B: Using `wget`

First, get the direct download URLs from the Zenodo API:

```bash
# Get file URLs
curl -s "https://zenodo.org/api/records/17404247" | \
  python -c "import sys, json; \
  [print(f['links']['self']) for f in json.load(sys.stdin)['files']]" \
  > urls.txt

# Download all files
wget -i urls.txt -P ./data

# Download with progress bar and continue on failure
wget -c -i urls.txt -P ./data --progress=bar:force
```

### Option C: Using `curl`

```bash
# Download single file
curl -O https://zenodo.org/records/17404247/files/Code.zip

# Download with progress bar
curl -# -O https://zenodo.org/records/17404247/files/Code.zip

# Resume interrupted download
curl -C - -O https://zenodo.org/records/17404247/files/[filename]
```

---

## Verification

### Verifying File Integrity

After downloading, verify your files using MD5 checksums:

#### Using our script (automatic):
The `download_data.py` script automatically verifies checksums.

#### Manual verification:

**On Linux/macOS:**
```bash
# Generate checksums
md5sum data/* > checksums.txt

# Compare with Zenodo (manually check against web page)
cat checksums.txt
```

**On Windows (PowerShell):**
```powershell
Get-FileHash -Algorithm MD5 data\* | Format-Table
```

#### Using zenodo_get:
```bash
zenodo_get 17404247 -o ./data -m
# This generates md5sums.txt automatically
```

### Expected File Structure

After downloading, your directory should look like this:

```
data/
├── raw_thermal_images.zip         # or extracted folders
├── preprocessed_patches.zip       # or extracted folders
├── environmental_metadata.zip     # or extracted folders
└── Code.zip                       # Scripts and code
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. **Download Interrupted**

**Solution:** Most tools support resume capability
```bash
# wget
wget -c [URL]

# curl
curl -C - -O [URL]

# Our script automatically resumes
python scripts/download_data.py --output ./data --type all
```

#### 2. **Checksum Mismatch**

**Solution:** Re-download the affected file
```bash
# Delete corrupted file
rm data/[corrupted_file]

# Re-download
python scripts/download_data.py --output ./data --type all
```

#### 3. **Insufficient Disk Space**

**Solution:** Download components separately
```bash
# Download only what you need
python scripts/download_data.py --output ./data --type raw
# Process and delete before downloading next component
python scripts/download_data.py --output ./data --type preprocessed
```

#### 4. **Slow Download Speed**

**Solutions:**
- Use command-line tools instead of browser
- Try different times of day (less network congestion)
- Use a download manager with parallel connections
- Download from a different network/location

#### 5. **Network Timeout**

**Solution:** Increase timeout in scripts or use tools with automatic retry
```bash
# Using wget with retries
wget --timeout=30 --tries=5 --retry-connrefused [URL]
```

#### 6. **Permission Denied Error**

**Solution:** Check directory permissions
```bash
# Create directory with proper permissions
mkdir -p ./data
chmod 755 ./data

# Run with appropriate permissions
python scripts/download_data.py --output ./data --type all
```

---

## Storage Requirements

Before downloading, ensure you have adequate storage:

| Component | Required Space |
|-----------|----------------|
| Raw images | ~24 GB |
| Preprocessed patches | ~3 GB |
| Metadata + Code | ~0.2 GB |
| **Total** | **~27 GB** |
| **Recommended** | **40+ GB** (for processing) |

**Note:** Add extra space for:
- Generated homography pairs during processing
- Temporary extraction files (if downloading zipped archives)
- Training/validation splits

---

## Post-Download Steps

After successfully downloading the dataset:

1. **Extract archives** (if downloaded as ZIP files):
   ```bash
   unzip Code.zip -d ./data/
   # Extract other archives as needed
   ```

2. **Verify structure** matches the expected layout:
   ```bash
   tree -L 2 ./data/
   ```

3. **Review documentation** in `Code.zip`:
   - README files
   - Configuration files
   - Example scripts

4. **Run preprocessing** (if you downloaded raw images):
   ```bash
   python src/preprocess.py \
     --input_dir ./data/raw \
     --output_dir ./data/preprocessed \
     --config configs/preprocessing_config.yaml
   ```

---

## Batch Processing for Large Downloads

For downloading on remote servers or in batch mode:

### Using Screen (Linux/macOS)

```bash
# Start screen session
screen -S thedpv_download

# Run download
python scripts/download_data.py --output ./data --type all

# Detach: Ctrl+A, then D
# Reattach: screen -r thedpv_download
```

### Using tmux (Linux/macOS)

```bash
# Start tmux session
tmux new -s thedpv_download

# Run download
python scripts/download_data.py --output ./data --type all

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t thedpv_download
```

### Using nohup (Linux/macOS)

```bash
# Run in background
nohup python scripts/download_data.py --output ./data --type all > download.log 2>&1 &

# Check progress
tail -f download.log
```

---

## Alternative Download Sources

If you experience issues with the primary Zenodo link, the dataset is also available via:

- **Direct Zenodo API**: `https://zenodo.org/api/records/17404247`
- **DOI resolver**: `https://doi.org/10.5281/zenodo.17404247`

---

## Getting Help

If you encounter issues not covered in this guide:

1. **Check existing issues**: [GitHub Issues](https://github.com/YaqoobAnsari/THED-PV/issues)
2. **Create a new issue**: Provide details about:
   - Operating system
   - Download method used
   - Error messages
   - Network conditions
3. **Contact authors**: yansari@andrew.cmu.edu

---

## Citation

If you use this dataset, please cite:

```bibtex
@article{yaqoob2025thedpv,
  title={THED-PV: A High-Resolution Multi-Perspective Thermal Imaging Dataset for Photovoltaic Homography Estimation},
  author={Yaqoob, Mohammed and Ansari, Mohammed Yusuf and Pillai, Dhanup Somasekharan and Flushing, Eduardo Feo},
  journal={Nature Scientific Data},
  year={2025},
  doi={10.5281/zenodo.17404247},
  url={https://doi.org/10.5281/zenodo.17404247}
}
```

---

## License

The THED-PV dataset is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

You are free to:
- ✅ Share and redistribute in any medium or format
- ✅ Adapt, remix, transform, and build upon the material
- ✅ Use for any purpose, even commercially

Under the condition that you provide appropriate attribution.

---

**Last Updated**: January 2025  
**Dataset Version**: v1.0.0  
**Zenodo Record**: 17404247