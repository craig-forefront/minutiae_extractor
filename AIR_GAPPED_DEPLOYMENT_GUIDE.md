# Air-Gapped Network Deployment Guide

This guide provides step-by-step instructions for preparing and deploying the Minutiae Extractor project in an air-gapped (offline) network environment.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Phase 1: Preparation on Internet-Connected Machine](#phase-1-preparation-on-internet-connected-machine)
4. [Phase 2: Transfer to Air-Gapped Network](#phase-2-transfer-to-air-gapped-network)
5. [Phase 3: Installation in Air-Gapped Environment](#phase-3-installation-in-air-gapped-environment)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The Minutiae Extractor requires:
- Python packages and dependencies (~500MB)
- Pre-trained neural network models (~1.5GB)
- System dependencies (optional, for GPU support)

**Total estimated size: ~2GB**

---

## Prerequisites

### Internet-Connected Machine Requirements
- Python >= 3.11
- Git
- Internet connection
- Sufficient disk space (~5GB recommended)
- Same OS/architecture as target air-gapped system (Linux x86_64, ARM64, etc.)

### Air-Gapped Machine Requirements
- Python >= 3.11 (pre-installed or bundled)
- Same OS and architecture as preparation machine
- Sufficient disk space (~3GB)

---

## Phase 1: Preparation on Internet-Connected Machine

### Step 1.1: Clone the Repository

```bash
# Clone the project
git clone https://github.com/tourmii/minutiae_extractor
cd minutiae_extractor

# Optional: checkout specific version for reproducibility
# git checkout <commit-hash>
```

### Step 1.2: Download Pre-Trained Models

```bash
# Install huggingface_hub temporarily (if not already installed)
pip install huggingface-hub

# Download models (will download ~1.5GB to models/ directory)
python scripts/install_model.py
```

**Models downloaded:**
- `models/CoarseNet.h5` (78 MB) - Initial minutiae detection
- `models/FineNet.h5` (624 MB) - Refined minutiae extraction
- `models/ClassifyNet_6_classes.h5` (591 MB) - Minutiae classification
- `models/CoreNet.weights` (245 MB) - Core region detection

**Total model size: ~1.5 GB**

### Step 1.3: Install uv Package Manager

```bash
# Install uv for Python package management
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (if needed)
source $HOME/.cargo/env
```

### Step 1.4: Download All Python Dependencies

We'll create a complete offline package bundle.

#### Option A: Using uv (Recommended)

```bash
# Create a clean environment and download all dependencies
uv sync

# Export the complete lockfile (already exists as uv.lock)
# This ensures reproducible installations

# Download wheels for offline installation
mkdir -p offline_packages
uv pip compile pyproject.toml --all-extras -o requirements-compiled.txt
uv pip download -r requirements-compiled.txt -d offline_packages/
```

#### Option B: Using pip

```bash
# Generate requirements.txt from pyproject.toml
pip install build
python -m build --sdist

# Download all packages with dependencies
mkdir -p offline_packages
pip download -r pyproject.toml --dest offline_packages/ \
    --platform manylinux2014_x86_64 \
    --python-version 311 \
    --only-binary=:all:

# Also download source distributions as fallback
pip download -r pyproject.toml --dest offline_packages_src/ \
    --no-binary=:all:
```

**Expected packages (from pyproject.toml):**
- tensorflow[and-cuda]>=2.20.0 (largest, ~500MB with CUDA)
- opencv-python==4.11.0.86
- numpy==2.3.3
- pandas==2.2.3
- matplotlib==3.10.6
- scikit-image>=0.25.2
- scikit-learn==1.6.1
- scipy==1.15.2
- huggingface-hub>=0.35.3
- And all transitive dependencies

### Step 1.5: Download uv Binary for Air-Gapped System

```bash
# Download standalone uv binary
mkdir -p offline_tools

# For Linux x86_64
wget -O offline_tools/uv https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu
chmod +x offline_tools/uv

# For Linux ARM64
# wget -O offline_tools/uv https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-unknown-linux-gnu
# chmod +x offline_tools/uv

# For macOS x86_64
# wget -O offline_tools/uv https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-apple-darwin
# chmod +x offline_tools/uv

# For macOS ARM64
# wget -O offline_tools/uv https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-apple-darwin
# chmod +x offline_tools/uv
```

### Step 1.6: (Optional) Download CUDA/cuDNN for GPU Support

If you plan to use GPU acceleration in the air-gapped environment:

```bash
# Download CUDA toolkit (if needed)
# Visit https://developer.nvidia.com/cuda-downloads
# Download the appropriate installer for your system

# TensorFlow 2.20.0 requires:
# - CUDA 12.x
# - cuDNN 8.9 or 9.x

# Note: As of the demo.ipynb, RTX 5090 (compute capability 12.0)
# is not yet supported, so CPU mode is used
```

### Step 1.7: Create Transfer Package

```bash
# Create a complete offline bundle
mkdir -p minutiae_extractor_airgap
cd minutiae_extractor_airgap

# Copy repository (excluding .git to save space)
rsync -av --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
    ../minutiae_extractor/ ./minutiae_extractor/

# Copy offline packages
cp -r ../offline_packages ./
cp -r ../offline_tools ./

# Create deployment script
cat > install_airgap.sh << 'EOF'
#!/bin/bash
set -e

echo "=== Minutiae Extractor Air-Gapped Installation ==="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "ERROR: Python >= 3.11 required, found $PYTHON_VERSION"
    exit 1
fi

echo "✓ Python $PYTHON_VERSION detected"

# Install uv
if [ ! -f "offline_tools/uv" ]; then
    echo "ERROR: uv binary not found in offline_tools/"
    exit 1
fi

mkdir -p ~/.local/bin
cp offline_tools/uv ~/.local/bin/
chmod +x ~/.local/bin/uv
export PATH="$HOME/.local/bin:$PATH"

echo "✓ uv package manager installed"

# Navigate to project directory
cd minutiae_extractor

# Create virtual environment and install packages
echo "Creating virtual environment and installing packages..."
~/.local/bin/uv venv

# Install packages from offline cache
~/.local/bin/uv pip install --no-index --find-links ../offline_packages -e .

echo ""
echo "=== Installation Complete ==="
echo ""
echo "To activate the environment:"
echo "  cd minutiae_extractor"
echo "  source .venv/bin/activate"
echo ""
echo "To test the installation:"
echo "  python -c 'from fingerflow.extractor import Extractor; print(\"Success!\")'"
echo ""
EOF

chmod +x install_airgap.sh

# Create README for the package
cat > README_AIRGAP.md << 'EOF'
# Minutiae Extractor - Air-Gapped Installation Package

This package contains everything needed to install the Minutiae Extractor
in an air-gapped (offline) environment.

## Contents

- `minutiae_extractor/` - Complete source code with pre-downloaded models
- `offline_packages/` - All Python package wheels (~500MB)
- `offline_tools/` - uv package manager binary
- `install_airgap.sh` - Automated installation script
- `README_AIRGAP.md` - This file

## System Requirements

- Linux, macOS, or Windows with WSL2
- Python >= 3.11
- 3GB free disk space
- (Optional) NVIDIA GPU with CUDA support

## Quick Start

1. Extract this package on the air-gapped system
2. Run the installation script:
   ```bash
   ./install_airgap.sh
   ```
3. Activate the environment and test:
   ```bash
   cd minutiae_extractor
   source .venv/bin/activate
   python -c "from fingerflow.extractor import Extractor; print('Success!')"
   ```

## Manual Installation

If the automated script fails, follow these steps:

1. Install uv:
   ```bash
   mkdir -p ~/.local/bin
   cp offline_tools/uv ~/.local/bin/
   chmod +x ~/.local/bin/uv
   export PATH="$HOME/.local/bin:$PATH"
   ```

2. Create virtual environment:
   ```bash
   cd minutiae_extractor
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install packages from offline cache:
   ```bash
   uv pip install --no-index --find-links ../offline_packages -e .
   ```

## Verification

Run the included demo notebook or test script:

```bash
source .venv/bin/activate
cd minutiae_extractor

# Quick test
python -c "
import cv2
from fingerflow.extractor import Extractor
print('Loading models...')
extractor = Extractor('models/CoarseNet.h5', 'models/FineNet.h5',
                     'models/ClassifyNet_6_classes.h5', 'models/CoreNet.weights')
print('✓ Models loaded successfully!')
"
```

## Models Included

- CoarseNet.h5 (78 MB) - Initial minutiae detection
- FineNet.h5 (624 MB) - Refined minutiae extraction
- ClassifyNet_6_classes.h5 (591 MB) - Minutiae classification
- CoreNet.weights (245 MB) - Core region detection

Total: ~1.5 GB

## Troubleshooting

### Python Version Issues
- Ensure Python 3.11 or higher is installed
- Check: `python3 --version`

### Package Installation Fails
- Verify all packages are in `offline_packages/` directory
- Try installing packages one by one to identify the problematic package
- Check for architecture mismatches (x86_64 vs ARM64)

### Model Loading Errors
- Verify all model files exist in `minutiae_extractor/models/`
- Check file integrity: models should total ~1.5 GB
- Ensure TensorFlow is properly installed

### GPU Not Working
- CPU mode is the default fallback
- For GPU support, ensure CUDA 12.x and cuDNN 8.9+ are installed
- Note: Some newer GPUs may not be supported by TensorFlow 2.20.0

## Support

For issues, refer to the main project documentation in
`minutiae_extractor/README.md`
EOF

cd ..

# Create compressed archive
echo "Creating compressed archive..."
tar czf minutiae_extractor_airgap.tar.gz minutiae_extractor_airgap/

echo ""
echo "=== Package Creation Complete ==="
echo ""
echo "Transfer file: minutiae_extractor_airgap.tar.gz"
echo "Size: $(du -h minutiae_extractor_airgap.tar.gz | cut -f1)"
echo ""
echo "This file contains:"
echo "  - Complete source code"
echo "  - Pre-downloaded models (1.5GB)"
echo "  - All Python packages (~500MB)"
echo "  - Installation tools"
echo ""
echo "Next steps:"
echo "1. Transfer minutiae_extractor_airgap.tar.gz to air-gapped system"
echo "2. Extract: tar xzf minutiae_extractor_airgap.tar.gz"
echo "3. Run: cd minutiae_extractor_airgap && ./install_airgap.sh"
```

---

## Phase 2: Transfer to Air-Gapped Network

### Method A: USB Drive

```bash
# On internet-connected machine
cp minutiae_extractor_airgap.tar.gz /media/usb_drive/

# On air-gapped machine
cp /media/usb_drive/minutiae_extractor_airgap.tar.gz ~/
```

### Method B: Network File Share

```bash
# Copy to network share accessible from both environments
cp minutiae_extractor_airgap.tar.gz /mnt/transfer_share/
```

### Method C: Physical Media

Burn `minutiae_extractor_airgap.tar.gz` to DVD/CD if size permits, or use external hard drive.

---

## Phase 3: Installation in Air-Gapped Environment

### Step 3.1: Extract Package

```bash
# On air-gapped machine
cd ~
tar xzf minutiae_extractor_airgap.tar.gz
cd minutiae_extractor_airgap
```

### Step 3.2: Run Installation

```bash
# Automated installation
./install_airgap.sh
```

The script will:
1. Verify Python version (>= 3.11)
2. Install uv package manager
3. Create virtual environment
4. Install all Python packages from offline cache
5. Set up the project

### Step 3.3: Activate Environment

```bash
cd minutiae_extractor
source .venv/bin/activate  # On Linux/macOS
# Or on Windows: .venv\Scripts\activate
```

---

## Verification

### Quick Verification

```bash
# Test import
python -c "from fingerflow.extractor import Extractor; print('Import successful!')"

# Verify models exist
python << EOF
import os
models = ['models/CoarseNet.h5', 'models/FineNet.h5',
          'models/ClassifyNet_6_classes.h5', 'models/CoreNet.weights']
for model in models:
    if os.path.exists(model):
        size = os.path.getsize(model) / (1024*1024)
        print(f"✓ {model} ({size:.1f} MB)")
    else:
        print(f"✗ {model} - MISSING!")
EOF
```

### Full Functional Test

```bash
# Run a test extraction
python << EOF
import cv2
import numpy as np
from fingerflow.extractor import Extractor

print("Loading models...")
extractor = Extractor(
    'models/CoarseNet.h5',
    'models/FineNet.h5',
    'models/ClassifyNet_6_classes.h5',
    'models/CoreNet.weights'
)
print("✓ Models loaded successfully")

# Test with a sample image (if available)
try:
    image = cv2.imread('sample/sample_fingerprint.tif')
    if image is not None:
        print("Running extraction on sample image...")
        result = extractor.extract_minutiae(image)
        print(f"✓ Extracted {len(result['minutiae'])} minutiae points")
    else:
        print("Sample image not found, but model loading successful")
except Exception as e:
    print(f"Warning: {e}")
    print("Note: This is expected if no sample image is available")
EOF
```

### Run Demo Notebook (Optional)

```bash
# Install Jupyter if needed (should already be installed)
# Start Jupyter
jupyter notebook demo.ipynb
```

---

## Troubleshooting

### Issue: Python Version Too Old

**Solution:**
- Install Python 3.11+ from source in air-gapped environment
- Or prepare a Python 3.11 installation bundle on internet-connected machine

### Issue: Package Installation Fails

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement...
```

**Solutions:**
1. Verify architecture matches: `uname -m` (should be same on both machines)
2. Check Python version matches: `python3 --version`
3. Ensure all packages downloaded: `ls -lh offline_packages/ | wc -l`
4. Try installing problematic package individually:
   ```bash
   uv pip install --no-index --find-links ../offline_packages <package-name>
   ```

### Issue: Model Files Corrupt

**Symptoms:**
```
Error loading model: Unable to open file...
```

**Solutions:**
1. Verify file sizes match expected values
2. Check MD5 checksums (create on internet machine, verify on air-gapped):
   ```bash
   # On internet machine
   md5sum models/*.h5 models/*.weights > model_checksums.txt

   # On air-gapped machine
   md5sum -c model_checksums.txt
   ```

### Issue: TensorFlow GPU Not Working

**Symptoms:**
```
Could not load dynamic library 'libcudart.so.12'
```

**Solutions:**
1. Fall back to CPU mode (default behavior):
   ```python
   import os
   os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
   ```
2. Install CUDA toolkit offline (prepare installation files on internet machine)
3. Check GPU compatibility with TensorFlow 2.20.0

### Issue: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'fingerflow'
```

**Solutions:**
1. Verify virtual environment is activated
2. Install in editable mode:
   ```bash
   uv pip install -e .
   ```
3. Check PYTHONPATH includes project directory

---

## Additional Notes

### Updating Models

If you need to update models later:

1. On internet-connected machine:
   ```bash
   python scripts/install_model.py
   tar czf models_only.tar.gz models/
   ```

2. Transfer `models_only.tar.gz` to air-gapped machine

3. On air-gapped machine:
   ```bash
   cd minutiae_extractor
   tar xzf ../models_only.tar.gz
   ```

### Updating Python Packages

To add new packages or update existing ones:

1. Update `pyproject.toml` on internet-connected machine
2. Download new packages:
   ```bash
   uv pip download -r pyproject.toml -d offline_packages/
   ```
3. Create new transfer package following Phase 1 steps

### Size Optimization

To reduce transfer size:

**Skip test/demo files:**
```bash
rsync -av --exclude='.git' --exclude='.venv' --exclude='test/' \
    --exclude='sample/' ../minutiae_extractor/ ./minutiae_extractor/
```

**Use compression:**
```bash
# XZ compression (better ratio, slower)
tar cJf minutiae_extractor_airgap.tar.xz minutiae_extractor_airgap/

# Zip format (wider compatibility)
zip -r minutiae_extractor_airgap.zip minutiae_extractor_airgap/
```

---

## Security Considerations

1. **Integrity Verification:**
   - Generate checksums on internet machine
   - Verify checksums on air-gapped machine
   - Use SHA-256 for cryptographic integrity:
     ```bash
     sha256sum minutiae_extractor_airgap.tar.gz > checksums.sha256
     ```

2. **Malware Scanning:**
   - Scan package on internet machine before transfer
   - Follow your organization's security policies

3. **Package Validation:**
   - Only download from trusted sources (PyPI, official repositories)
   - Verify package signatures when available

4. **Access Control:**
   - Restrict who can create transfer packages
   - Log all transfers to air-gapped environment
   - Follow principle of least privilege

---

## Performance Considerations

### CPU vs GPU

- **CPU Mode (Default):** Works on all systems, slower inference (~2-5 seconds per image)
- **GPU Mode:** Requires CUDA setup, faster inference (~0.1-0.5 seconds per image)

### Memory Requirements

- **Minimum:** 4GB RAM
- **Recommended:** 8GB+ RAM
- **Peak Usage:** ~2-3GB during model loading and inference

### Disk Space

- **Installation:** ~2GB
- **Runtime cache:** ~500MB
- **Output storage:** Varies based on usage

---

## Success Checklist

- [ ] Python 3.11+ installed on air-gapped machine
- [ ] All models downloaded (1.5GB total)
- [ ] All Python packages downloaded (~500MB)
- [ ] uv binary included in transfer package
- [ ] Transfer package created and checksummed
- [ ] Package successfully transferred to air-gapped environment
- [ ] Installation script executed without errors
- [ ] Virtual environment activated
- [ ] Import test successful
- [ ] Model loading test successful
- [ ] Sample extraction test successful (if sample image available)

---

## Quick Reference Commands

```bash
# Phase 1: Preparation (Internet-connected)
git clone https://github.com/tourmii/minutiae_extractor
cd minutiae_extractor
python scripts/install_model.py
uv sync
mkdir offline_packages && uv pip download -r pyproject.toml -d offline_packages/
# Follow packaging steps in guide

# Phase 2: Transfer
# (Copy minutiae_extractor_airgap.tar.gz via USB/network)

# Phase 3: Installation (Air-gapped)
tar xzf minutiae_extractor_airgap.tar.gz
cd minutiae_extractor_airgap
./install_airgap.sh
cd minutiae_extractor
source .venv/bin/activate

# Verification
python -c "from fingerflow.extractor import Extractor; print('Success!')"
```

---

## Support and Contact

For questions or issues not covered in this guide:

1. Check the main [README.md](README.md)
2. Review [demo.ipynb](demo.ipynb) for usage examples
3. Consult the project repository: https://github.com/tourmii/minutiae_extractor
4. Refer to the original papers: https://arxiv.org/pdf/1712.09401

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

---

**Document Version:** 1.0
**Last Updated:** 2026-01-14
**Compatible with:** Minutiae Extractor v0.1.0, TensorFlow 2.20.0, Python 3.11+
