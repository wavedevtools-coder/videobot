# Tiny Dino AI Shorts Factory v2.0.1

Automated AI-powered short video generation pipeline running entirely on local hardware. Creates 30-second animated "Tiny Dino" stories optimized for YouTube Shorts, Instagram Reels, and TikTok.

## Features

| Feature | Status |
|---------|--------|
| Story Formula Enforcement (6-scene arc) | Implemented |
| Character Consistency (Tiny Dino profile) | Implemented |
| Anti-Repetition (1000-story history) | Implemented |
| Story Quality Scoring (5 dimensions) | Implemented |
| Budget Control (INR tracking) | Implemented |
| Upload Queue with Scheduling | Implemented |
| Outro Versioning (cache invalidation) | Implemented |
| Batch Generation with Crash Safety | Implemented |
| RTX 3090 24GB Optimized | Yes |

## Architecture

```
Generate Story (Qwen3 14B via Ollama)
    |
Story Quality Score (>70/100 required)
    |
Validate Story (anti-repeat check)
    |
Accept / Regenerate (max 3 retries)
    |
Build Scene Prompts (character-enriched)
    |
Generate Images (FLUX.1 Dev via Diffusers)
    |
Generate Videos (LTX-Video-2-3)
    |
Generate Audio (Stable Audio Open)
    |
Assemble (FFmpeg + outro + branding)
    |
Quality Check (duration, resolution, brightness)
    |
Upload Queue (scheduled YouTube upload)
```

## System Requirements

### Hardware
- **GPU**: NVIDIA RTX 3090 24GB (or equivalent VRAM)
- **RAM**: 32GB+ recommended
- **Storage**: 50GB+ free (models + generated videos)
- **OS**: Windows 10/11, Linux, or macOS (with modifications)

### Software
- Python 3.10+
- FFmpeg 5.0+
- CUDA 12.1+
- Ollama (for Qwen3 14B)

### AI Models (Local)
| Model | Purpose | VRAM | Backend |
|-------|---------|------|---------|
| Qwen3 14B | Story generation | ~10GB | Ollama |
| FLUX.1 Dev | Image generation | ~12GB | Diffusers |
| LTX-Video-2-3 | Video generation | ~16GB | Diffusers |
| Stable Audio Open | Audio/SFX | ~4GB | Transformers |

*Models load sequentially, not simultaneously. Peak VRAM usage stays within 24GB.*

## Installation

### 1. Clone and Setup
```bash
# Clone repository
git clone <repo-url>
cd tiny-dino-factory

# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install Ollama + Qwen3
```bash
# Install Ollama from https://ollama.com

# Pull Qwen3 14B
ollama pull qwen3:14b

# Test
ollama run qwen3:14b "Say hello"
```

### 3. Install FFmpeg
```bash
# Windows (Chocolatey)
choco install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### 4. Download Models

Models download automatically on first run via HuggingFace Diffusers/Transformers.

For offline use, pre-download:
```python
# Download FLUX.1 Dev
from diffusers import FluxPipeline
FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")

# Download LTX-Video
from diffusers import LTXPipeline
LTXPipeline.from_pretrained("Lightricks/LTX-Video-2-3")

# Download Stable Audio
from transformers import AutoProcessor, StableAudioSpectralDiffusionPipeline
AutoProcessor.from_pretrained("stabilityai/stable-audio-open-1.0")
```

### 5. Configure

Edit `config.yaml`:
```yaml
project_root: "D:/dino"  # Change to your path

# Budget
budget:
  monthly_budget_inr: 1000  # Your budget

# Upload (optional)
upload:
  enabled: false  # Set true when ready
```

### 6. YouTube Setup (Optional)

For automated uploads:
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create project > Enable YouTube Data API v3
3. Create OAuth 2.0 credentials (Desktop app)
4. Download `client_secret.json` to project root
5. Set: `set YOUTUBE_CLIENT_SECRETS=client_secret.json`

## Usage

### Check System Readiness
```bash
python main.py --check
```

### Generate 1 Video
```bash
python main.py
```

### Generate Multiple Videos
```bash
python main.py --count 5
```

### Generate and Score Story Only (Test)
```bash
python main.py --story-only
```

### Check Status & Budget
```bash
python main.py --status
```

### Process Upload Queue
```bash
python main.py --upload
```

## Project Structure

```
tiny-dino-factory/
├── main.py                    # Entry point
├── config.yaml                # Main configuration
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── modules/                   # Core pipeline modules
│   ├── __init__.py
│   ├── config_loader.py       # YAML configuration
│   ├── character_manager.py   # Tiny Dino profile
│   ├── story_scorer.py        # Quality scoring (5 dimensions)
│   ├── story_generator.py     # LLM story generation
│   ├── scene_generator.py     # Prompt engineering
│   ├── image_generator.py     # FLUX.1 Dev (Diffusers)
│   ├── video_generator.py     # LTX-Video-2-3
│   ├── audio_generator.py     # Stable Audio Open
│   ├── assembly_engine.py     # FFmpeg assembly
│   ├── upload_manager.py      # YouTube upload queue
│   ├── cost_tracker.py        # Budget tracking
│   ├── quality_checker.py     # Final validation
│   └── pipeline.py            # Orchestration engine
│
├── assets/                    # Static assets
│   ├── character_profile.json # Tiny Dino definition
│   ├── outro.mp4             # Cached outro (auto-gen)
│   └── outro.json            # Outro version metadata
│
├── data/                      # Runtime data
│   ├── story_history.json     # Anti-repeat database
│   ├── generation_stats.json  # Cost & usage stats
│   └── upload_queue.json      # Pending uploads
│
├── output/                    # Generated content
│   ├── videos/               # Final videos
│   ├── temp/                 # Temporary workspace
│   └── uploaded/             # Uploaded videos log
│
└── logs/                      # Log files
    └── factory.log
```

## Configuration Reference

### Video Settings
```yaml
video:
  width: 720           # 9:16 vertical
  height: 1280
  fps: 30
  scene_duration: 5    # seconds per scene
  outro_duration: 4
```

### Quality Thresholds
```yaml
quality:
  min_story_score: 70  # Reject stories below this
  max_regenerations: 3
```

### Budget Control
```yaml
budget:
  monthly_budget_inr: 1000
  stop_when_exceeded: true
  warning_threshold_percent: 80
```

### Generation
```yaml
generation:
  videos_per_run: 1
  max_retry: 3
  save_after_each_video: true  # Crash safety
```

### Upload Schedule
```yaml
upload:
  enabled: true
  videos_per_day: 1
  upload_times:
    - "18:00"         # 6 PM daily
    # - "10:00"       # Future: multiple uploads
```

## Story Quality Scoring

Stories are scored on 5 dimensions (0-100 each):

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Hook Strength | 25% | Opening scene grabs attention |
| Comedy Potential | 25% | Physical comedy, slapstick beats |
| Visual Clarity | 20% | AI can render the scene |
| Uniqueness | 15% | Not repetitive vs history |
| Ending Quality | 15% | Satisfying resolution |

**Minimum passing score: 70/100**

## Cost Estimates (per video, 6 scenes)

| Stage | GPU Minutes | Cost (INR) |
|-------|-------------|------------|
| Story (LLM) | ~0.1 | ~0.01 |
| Images (FLUX) | ~3.0 | ~1.20 |
| Video (LTX) | ~12.0 | ~14.40 |
| Audio (Stable Audio) | ~1.8 | ~0.36 |
| Assembly (FFmpeg) | ~0.05 | ~0.01 |
| **Total** | **~17** | **~16** |

*At ₹1000/month budget: ~60-62 videos per month*

## Character: Tiny Dino

Stored in `assets/character_profile.json`:
- **Species**: Baby Tyrannosaurus Rex
- **Colors**: Bright green body, light yellow belly
- **Eyes**: Large expressive with sparkle highlights
- **Personality**: Curious, playful, clumsy, optimistic
- **Communication**: Squeaks, roars, body language only
- **Humor Style**: Physical comedy, slapstick, exaggerated reactions

Every scene prompt automatically enriches with Tiny Dino's appearance and current mood.

## Troubleshooting

### CUDA Out of Memory
```yaml
# In config.yaml - enable memory optimizations
models:
  image:
    enable_cpu_offload: true
  video:
    enable_vae_slicing: true
```

### Ollama Connection Failed
```bash
# Ensure Ollama is running
ollama serve

# Test connection
curl http://localhost:11434/api/tags
```

### FFmpeg Not Found
```bash
# Add to PATH or set full path
set PATH=%PATH%;C:\Program Files\ffmpeg\bin
```

### Model Download Slow
```bash
# Set HuggingFace cache to SSD
set HF_HOME=D:\huggingface_cache

# Or use mirror
set HF_ENDPOINT=https://hf-mirror.com
```

## License

MIT License - See LICENSE file

## Credits

- **Tiny Dino** concept and character design
- **FLUX.1 Dev** by Black Forest Labs
- **LTX-Video** by Lightricks
- **Stable Audio Open** by Stability AI
- **Qwen3** by Alibaba Cloud
- Built for RTX 3090 24GB local deployment
