#!/bin/bash
# Video Intelligence System - Installation Script
# Run: bash install.sh

set -e

echo "============================================="
echo "  Video Intelligence System - Installation"
echo "============================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python version
echo ""
echo "[1/6] Checking Python..."
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "ERROR: Python not found. Install Python 3.11+"
    exit 1
fi

PY_VERSION=$($PYTHON --version 2>&1 | grep -oP '\d+\.\d+')
echo "  Found Python $PY_VERSION"

# Create virtual environment
echo ""
echo "[2/6] Creating virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON -m venv venv
    echo "  Created venv/"
else
    echo "  venv/ already exists"
fi

# Activate venv
source venv/bin/activate
pip install --upgrade pip -q

# Install PyTorch with CUDA
echo ""
echo "[3/6] Installing PyTorch with CUDA support..."
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  PyTorch with CUDA already installed"
else
    echo "  Installing PyTorch (this may take a few minutes)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
fi

# Install dependencies
echo ""
echo "[4/6] Installing Python dependencies..."
pip install -r requirements.txt -q
echo "  Dependencies installed"

# Check FFmpeg
echo ""
echo "[5/6] Checking FFmpeg..."
if command -v ffmpeg &>/dev/null; then
    FFMPEG_V=$(ffmpeg -version 2>&1 | head -1)
    echo "  $FFMPEG_V"
else
    echo "  WARNING: FFmpeg not found. Install with: sudo apt install ffmpeg"
fi

# Verify GPU
echo ""
echo "[6/6] Verifying GPU..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
    print(f'  CUDA: {torch.version.cuda}')
else:
    print('  WARNING: No GPU detected. Processing will use CPU (slower).')
" 2>/dev/null || echo "  WARNING: Could not verify GPU"

# Create directory structure
echo ""
echo "Creating directory structure..."
CONFIG_INPUT=$(python -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['paths']['input'])" 2>/dev/null || echo "/mnt/video_hub/00_ENTRADA")
CONFIG_OUTPUT=$(python -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['paths']['output'])" 2>/dev/null || echo "/mnt/video_hub/01_PROCESADOS")
CONFIG_ANALYSIS=$(python -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['paths']['analysis'])" 2>/dev/null || echo "/mnt/video_hub/_ANALYSIS")

for DIR in "$CONFIG_INPUT" "$CONFIG_OUTPUT" "$CONFIG_ANALYSIS" "${CONFIG_ANALYSIS}/database" "${CONFIG_ANALYSIS}/thumbnails/videos" "${CONFIG_ANALYSIS}/thumbnails/faces" "${CONFIG_ANALYSIS}/reports" "${CONFIG_ANALYSIS}/exports" "${CONFIG_ANALYSIS}/logs"; do
    if mkdir -p "$DIR" 2>/dev/null; then
        echo "  Created: $DIR"
    else
        echo "  Skipped (no permission): $DIR"
    fi
done

echo ""
echo "============================================="
echo "  Installation Complete!"
echo "============================================="
echo ""
echo "  Usage:"
echo "    source venv/bin/activate"
echo "    python main.py"
echo ""
echo "  Edit config.yaml to customize settings."
echo "============================================="
