# VideoBot Server Deployment Guide

## Quick Start (One Command)

```bash
# Clone and run installation
git clone https://github.com/wavedevtools-coder/videobot.git
cd videobot
chmod +x install.sh
sudo ./install.sh
```

## Manual Installation Steps

### 1. Clone Repository
```bash
git clone https://github.com/wavedevtools-coder/videobot.git
cd videobot
```

### 2. Install System Dependencies
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg git curl wget build-essential libgl1 libglib2.0-0 zstd
```

### 3. Setup Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install Ollama (for AI script generation)
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve &

# Pull required model (Qwen3 32B for high-quality storytelling)
ollama pull qwen3:32b
```

### 5. Run the Pipeline
```bash
source venv/bin/activate
python main.py
```

## Production Deployment (Systemd Service)

### 1. Copy files to production location
```bash
sudo mkdir -p /opt/videobot
sudo cp -r * /opt/videobot/
cd /opt/videobot
```

### 2. Run installation script
```bash
sudo ./install.sh
```

### 3. Setup systemd service
```bash
# Copy service file
sudo cp videobot.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start service
sudo systemctl enable videobot
sudo systemctl start videobot

# Check status
sudo systemctl status videobot

# View logs
sudo journalctl -u videobot -f
```

### 4. Schedule with Cron (Alternative to systemd)
```bash
# Edit crontab
crontab -e

# Add this line to run every 30 minutes
*/30 * * * * cd /opt/videobot && source venv/bin/activate && python main.py >> logs/cron.log 2>&1
```

## Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# Hugging Face Token (optional, for gated models)
HF_TOKEN=your_token_here

# Ollama Host
OLLAMA_HOST=127.0.0.1:11434

# Cache directory
HF_HOME=/tmp/huggingface

# GPU Settings (if available)
CUDA_VISIBLE_DEVICES=0
```

### Config File (`config.yaml`)
Edit `config.yaml` to customize:
- Video duration
- Image resolution
- Model settings
- Output formats

## Monitoring

### Check Logs
```bash
# Application logs
tail -f logs/app.log

# Systemd service logs
sudo journalctl -u videobot -f

# Ollama logs
tail -f /var/log/ollama.log
```

### Health Check
```bash
# Check if pipeline is running
ps aux | grep main.py

# Check Ollama status
ollama list

# Check GPU usage (if available)
nvidia-smi
```

## Troubleshooting

### Common Issues

**1. Out of Memory**
```bash
# Reduce batch size or image resolution in config.yaml
# Or add swap space
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**2. Ollama Connection Failed**
```bash
# Restart Ollama
sudo systemctl restart ollama

# Check if running
ollama list
```

**3. Disk Space Full**
```bash
# Clean up cache
rm -rf /tmp/huggingface/hub/*

# Remove old videos
find outputs/videos -mtime +7 -delete
```

**4. Slow Generation / Out of Memory**
- The new pipeline (Wan I2V + FLUX.1 Dev) requires a high-end GPU (min ~24GB VRAM like RTX 3090/4090/A100)
- Consider adding a GPU or renting cloud instances
- Use FLUX.1-schnell instead of FLUX.1-dev for faster image generation (edit `config.yaml`)
- Wan 2.1 14B is highly recommended for stability over Wan 2.2

## Performance Optimization

### GPU Acceleration
```bash
# Install NVIDIA drivers (if not installed)
sudo apt-get install -y nvidia-driver-535

# Install CUDA toolkit
sudo apt-get install -y nvidia-cuda-toolkit

# Verify GPU
nvidia-smi
```

### Multi-GPU Setup
```bash
# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Run pipeline
python main.py
```

## Updates

```bash
# Pull latest changes
cd /opt/videobot
git pull

# Restart service
sudo systemctl restart videobot

# Or run manually
source venv/bin/activate
python main.py
```

## Support

For issues and feature requests:
- GitHub Issues: https://github.com/wavedevtools-coder/videobot/issues
- Documentation: https://github.com/wavedevtools-coder/videobot/wiki
