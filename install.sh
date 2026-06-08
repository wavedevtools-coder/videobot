#!/bin/bash

# VideoBot Server Installation Script
# Automates dependency installation, model downloads, and service setup

set -e

echo "🚀 VideoBot Server Installation"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${YELLOW}Warning: Not running as root. Some installations may fail.${NC}"
fi

# 1. System Dependencies
echo -e "\n${GREEN}[1/6] Installing system dependencies...${NC}"
apt-get update -y
apt-get install -y \
    python3-pip \
    python3-venv \
    ffmpeg \
    git \
    curl \
    wget \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    zstd

# 2. Python Environment
echo -e "\n${GREEN}[2/6] Setting up Python environment...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

rm -rf venv
python3 -m venv venv

# Bootstrap pip if it is missing (common in some minimal Debian/Ubuntu environments)
if [ ! -f "venv/bin/pip" ]; then
    echo "Warning: pip not found in virtual environment. Bootstrapping with get-pip.py..."
    curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    ./venv/bin/python3 get-pip.py --quiet
    rm -q get-pip.py 2>/dev/null || rm -f get-pip.py
fi

. venv/bin/activate

# 3. Python Dependencies
echo -e "\n${GREEN}[3/6] Installing Python dependencies...${NC}"
./venv/bin/pip install --upgrade pip -q
./venv/bin/pip install -r requirements.txt -q
./venv/bin/pip install tiktoken sentencepiece -q

# 4. Ollama Installation (Optional but recommended)
echo -e "\n${GREEN}[4/6] Setting up Ollama...${NC}"
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh > /dev/null 2>&1
    echo "Ollama installed successfully"
else
    echo "Ollama already installed"
fi

# Start Ollama service (only if not already running)
echo "Checking Ollama service..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Ollama already running"
else
    echo "Starting Ollama service..."
    ollama serve > /var/log/ollama.log 2>&1 &
    # Wait until Ollama is actually ready (up to 30s)
    for i in $(seq 1 30); do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "Ollama ready after ${i}s"
            break
        fi
        sleep 1
    done
fi

# Pull required model
echo "Pulling qwen3:32b model (~20GB)..."
ollama pull qwen3:32b
echo -e "${GREEN}✓ Ollama ready${NC}"

# 5. Download Models (Pre-warm cache)
echo -e "\n${GREEN}[5/6] Pre-downloading AI models...${NC}"
echo "This may take 10-20 minutes depending on your connection..."

./venv/bin/python3 << 'EOF'
import os
import torch
os.environ['HF_HOME'] = '/tmp/huggingface'

hf_token = os.environ.get('HF_TOKEN', '')
token_kwargs = {'token': hf_token} if hf_token else {}

# 1. Image Generation (FLUX.1-dev — gated, needs HF_TOKEN)
try:
    from diffusers import AutoPipelineForText2Image
    print("Downloading FLUX.1-dev...")
    AutoPipelineForText2Image.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        **token_kwargs,
    )
    print("✓ FLUX.1-dev downloaded")
except Exception as e:
    print(f"Note: FLUX.1-dev will download on first run (needs HF_TOKEN): {e}")

# 2. Video Generation (Wan 2.1 I2V)
try:
    from diffusers import WanImageToVideoPipeline
    print("Downloading Wan2.1-I2V-14B-480P...")
    WanImageToVideoPipeline.from_pretrained(
        "Wan-AI/Wan2.1-I2V-14B-480P",
        torch_dtype=torch.bfloat16,
    )
    print("✓ Wan2.1-I2V downloaded")
except Exception as e:
    print(f"Note: Wan I2V will download on first run: {e}")

# 3. Audio Generation (stabilityai/stable-audio-open-1.0)
try:
    from diffusers import StableAudioPipeline
    print("Downloading Stable Audio Open...")
    StableAudioPipeline.from_pretrained(
        "stabilityai/stable-audio-open-1.0",
        torch_dtype=torch.float16,
        **token_kwargs,
    )
    print("✓ Stable Audio Open downloaded")
except Exception as e:
    print(f"Note: Stable Audio Open will download on first run: {e}")

try:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except:
    pass
EOF

# 6. Create necessary directories
echo -e "\n${GREEN}[6/6] Setting up directories...${NC}"
mkdir -p outputs/videos outputs/images outputs/audio logs
chmod -R 755 outputs logs

echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}✅ Installation Complete!${NC}"
echo -e "${GREEN}================================${NC}"

echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Activate environment: . venv/bin/activate"
echo "2. Run pipeline: python main.py"
echo "3. Or use the service: sudo systemctl start videobot"
echo ""
echo -e "${YELLOW}To run immediately:${NC}"
echo ". venv/bin/activate && python main.py"
