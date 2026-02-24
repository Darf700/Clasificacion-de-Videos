#!/bin/bash
# ============================================================================
# Video Intelligence System - Installation Script
# ============================================================================
# Installs all dependencies for Ubuntu 24.04 with NVIDIA GPU support.
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
# ============================================================================

set -e

echo "============================================"
echo " Video Intelligence System - Installer"
echo "============================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- System dependencies ---
echo -e "${YELLOW}[1/5] Installing system dependencies...${NC}"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    ffmpeg \
    python3-pip \
    python3-venv \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    > /dev/null 2>&1

echo -e "${GREEN}  System dependencies installed.${NC}"

# --- Python virtual environment ---
echo -e "${YELLOW}[2/5] Creating Python virtual environment...${NC}"
VENV_DIR="${SCRIPT_DIR}/venv"

if [ -d "$VENV_DIR" ]; then
    echo "  Virtual environment already exists at ${VENV_DIR}"
else
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}  Virtual environment created at ${VENV_DIR}${NC}"
fi

source "${VENV_DIR}/bin/activate"

# --- Upgrade pip ---
echo -e "${YELLOW}[3/5] Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel -q

# --- Install PyTorch with CUDA ---
echo -e "${YELLOW}[4/5] Installing PyTorch with CUDA support...${NC}"
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 -q

# --- Install remaining dependencies ---
echo -e "${YELLOW}[5/5] Installing Python dependencies...${NC}"
pip install -r "${SCRIPT_DIR}/requirements.txt" \
    --ignore-installed torch torchvision torchaudio -q

# Install OpenAI CLIP from GitHub
pip install git+https://github.com/openai/CLIP.git -q

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN} Installation Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

# --- Verify GPU ---
echo "Verifying GPU setup..."
python3 -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'  GPU: {gpu} ({vram:.1f} GB VRAM)')
    print('  CUDA: OK')
else:
    print('  WARNING: CUDA not available. Will use CPU (slower).')
"

# --- Verify FFmpeg ---
echo ""
echo "Verifying FFmpeg..."
if command -v ffmpeg &> /dev/null; then
    ffmpeg_version=$(ffmpeg -version | head -1)
    echo "  ${ffmpeg_version}"
else
    echo -e "${RED}  WARNING: FFmpeg not found!${NC}"
fi

echo ""
echo "To start processing:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  python main.py"
echo ""
echo "For a dry run (no file moves):"
echo "  python main.py --dry-run"
echo ""
