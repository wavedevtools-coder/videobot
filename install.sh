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
apt-get update -qq
apt-get install -y -qq \
    python3-pip \
    python3-venv \
    ffmpeg \
    git \
    curl \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    > /dev/null 2>&1 || echo "Note: Some system packages may already be installed"

# 2. Python Environment
echo -e "\n${GREEN}[2/6] Setting up Python environment...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

python3 -m venv venv
source venv/bin/activate

# 3. Python Dependencies
echo -e "\n${GREEN}[3/6] Installing Python dependencies...${NC}"
./venv/bin/pip install --upgrade pip -q
./venv/bin/pip install -r requirements.txt -q

# 4. Ollama Installation (Optional but recommended)
echo -e "\n${GREEN}[4/6] Setting up Ollama...${NC}"
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh > /dev/null 2>&1
    echo "Ollama installed successfully"
else
    echo "Ollama already installed"
fi

# Start Ollama service
echo "Starting Ollama service..."
ollama serve > /var/log/ollama.log 2>&1 &
sleep 5

# Pull required model
echo "Pulling llama3.2 model..."
ollama pull llama3.2 > /dev/null 2>&1
echo -e "${GREEN}✓ Ollama ready${NC}"

# 5. Download Models (Pre-warm cache)
echo -e "\n${GREEN}[5/6] Pre-downloading AI models...${NC}"
echo "This may take 10-20 minutes depending on your connection..."

./venv/bin/python3 << 'EOF'
import os
os.environ['HF_HOME'] = '/tmp/huggingface'

try:
    from diffusers import FluxPipeline
    print("Downloading FLUX.1-schnell...")
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype="auto",
        use_safetensors=True
    )
    print("✓ FLUX.1-schnell downloaded")
except Exception as e:
    print(f"Note: Model will download on first run: {e}")

try:
    import torch
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
echo "1. Activate environment: source venv/bin/activate"
echo "2. Run pipeline: python main.py"
echo "3. Or use the service: sudo systemctl start videobot"
echo ""
echo -e "${YELLOW}To run immediately:${NC}"
echo "source venv/bin/activate && python main.py"
