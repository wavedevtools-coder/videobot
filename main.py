#!/usr/bin/env python3
"""
Tiny Dino AI Shorts Factory v2.0.1
Main entry point for the automated video generation pipeline.

Usage:
    python main.py                    # Generate 1 video (default)
    python main.py --count 5          # Generate 5 videos
    python main.py --story-only       # Generate and score story only
    python main.py --check            # Check system readiness
    python main.py --status           # Show status and budget
    python main.py --upload           # Process upload queue

Environment:
    Set DINOFACTORY_CONFIG to override config.yaml path.
"""

import argparse
import logging
import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.config_loader import Config
from modules.pipeline import GenerationPipeline
from modules.cost_tracker import CostTracker
from modules.character_manager import CharacterManager


def setup_logging(config: Config):
    """Configure logging with file and console handlers."""
    log_cfg = config.section('logging')
    level = getattr(logging, log_cfg.get('level', 'INFO').upper(), logging.INFO)
    log_format = log_cfg.get('format', '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
    log_file = log_cfg.get('file', 'logs/factory.log')

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # File handler with rotation
    try:
        from logging.handlers import RotatingFileHandler
        max_bytes = log_cfg.get('max_size_mb', 50) * 1024 * 1024
        backup = log_cfg.get('backup_count', 5)
        fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup)
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(fh)
    except Exception as e:
        print(f"Warning: Could not set up file logging: {e}")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(ch)

    return root_logger


def check_system(config: Config) -> bool:
    """Check system readiness for generation."""
    import subprocess

    logger = logging.getLogger('main')
    logger.info("=" * 60)
    logger.info("SYSTEM READINESS CHECK")
    logger.info("=" * 60)

    checks = []

    # 1. Check FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=10)
        checks.append(("FFmpeg", result.returncode == 0, "Install: choco install ffmpeg"))
    except FileNotFoundError:
        checks.append(("FFmpeg", False, "NOT FOUND - Install: choco install ffmpeg"))

    # 2. Check Ollama - server-side: make it optional with fallback
    try:
        import requests
        base = config.models.get('llm', {}).get('base_url', 'http://localhost:11434')
        r = requests.get(f"{base}/api/tags", timeout=5)
        checks.append(("Ollama", r.status_code == 200, f"Running @ {base}"))
    except Exception as e:
        # Server mode: Ollama is recommended but not critical if alternative LLM is configured
        has_alternative = config.models.get('llm_fallback') is not None
        checks.append(("Ollama", False, f"NOT RUNNING - Start: ollama serve ({e})"))

    # 3. Check Python dependencies
    deps = {
        'torch': 'PyTorch',
        'diffusers': 'Diffusers (FLUX, LTX)',
        'transformers': 'Transformers (Stable Audio)',
        'requests': 'Requests (Ollama API)',
        'PIL': 'Pillow (Image processing)',
        'cv2': 'OpenCV (Video processing)',
        'yaml': 'PyYAML (Config)',
    }

    for module, name in deps.items():
        try:
            __import__(module)
            checks.append((name, True, "OK"))
        except ImportError:
            # Special handling for transformers - it might be installed but not importable due to version issues
            if module == 'transformers':
                # Try a more specific check
                try:
                    import transformers
                    checks.append((name, True, "OK"))
                except ImportError:
                    checks.append((name, False, f"pip install {module}"))
            else:
                checks.append((name, False, f"pip install {module}"))

    # 4. Check CUDA
    try:
        import torch
        cuda = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if cuda else "N/A"
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3 if cuda else 0
        checks.append(("CUDA GPU", cuda, f"{device} ({mem:.1f} GB)" if cuda else "No GPU found"))
    except Exception as e:
        checks.append(("CUDA GPU", False, str(e)))

    # 5. Check disk space - server-side: warn but don't fail
    root = config.get('project_root') or '.'
    if os.path.exists(root):
        stat = os.statvfs(root) if hasattr(os, 'statvfs') else None
        if stat:
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            checks.append(("Disk Space", free_gb > 2, f"{free_gb:.1f} GB free (min 2GB for server)"))
        else:
            checks.append(("Disk Space", True, "Cannot check on Windows"))
    else:
        checks.append(("Disk Space", False, f"Project root not found: {root}"))

    # 6. Check character profile
    char_mgr = CharacterManager(config)
    try:
        valid = char_mgr.validate_profile()
        checks.append(("Character Profile", valid, "Valid" if valid else "Invalid"))
    except Exception as e:
        checks.append(("Character Profile", False, str(e)))

    # Print results - server mode: only critical failures block execution
    all_pass = True
    critical_failures = []
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        icon = "+" if passed else "-"
        logger.info(f"  [{icon}] {name:20s} | {status:4s} | {detail}")
        
        # Server-side: Only FFmpeg is a hard blocker
        # Ollama, GPU, disk space are warnings for server operation
        if not passed:
            if name == "FFmpeg":
                critical_failures.append(name)
                all_pass = False
            elif name == "Ollama":
                logger.info("  Note: Ollama not running - ensure LLM service is available or configure fallback")
            elif name == "Transformers (Stable Audio)":
                logger.info("  Note: Transformers optional - audio generation will be skipped")
            elif name == "CUDA GPU":
                logger.info("  Note: Running on CPU - generation will be slower")
            elif name == "Disk Space":
                logger.info("  Note: Low disk space - monitor during generation")
            elif name == "Character Profile":
                critical_failures.append(name)
                all_pass = False

    logger.info("=" * 60)
    logger.info(f"Overall: {'READY' if all_pass else 'NOT READY - Fix issues above'}")
    logger.info("=" * 60)

    return all_pass


def show_status(config: Config):
    """Show current pipeline status and budget."""
    tracker = CostTracker(config)
    print(tracker.get_stats_report())


def generate_story_only(config: Config):
    """Generate and score a story without video."""
    from modules.story_generator import StoryGenerator
    from modules.scene_generator import ScenePromptBuilder

    logger = logging.getLogger('main')
    logger.info("Story-only mode: Generating story...")

    story_gen = StoryGenerator(config)
    story = story_gen.generate()

    score = story.get('_score', {})
    print("\n" + "=" * 60)
    print(f"TITLE: {story.get('title', 'Untitled')}")
    print(f"THEME: {story.get('theme', 'N/A')}")
    print("=" * 60)
    print(f"SCORE: {score.get('total', 0)}/100")
    print(f"  Hook:    {score.get('hook', 0)}/100")
    print(f"  Comedy:  {score.get('comedy', 0)}/100")
    print(f"  Visual:  {score.get('visual', 0)}/100")
    print(f"  Unique:  {score.get('unique', 0)}/100")
    print(f"  Ending:  {score.get('ending', 0)}/100")
    print("-" * 60)
    print("SCENES:")
    for scene in story.get('scenes', []):
        print(f"\n  Scene {scene.get('scene_number')} [{scene.get('type', '?')}] "
              f"| {scene.get('mood', '?')}")
        print(f"  {scene.get('description', '')[:120]}...")
    print("=" * 60)

    # Show prompts
    print("\nGENERATED PROMPTS:")
    builder = ScenePromptBuilder(config)
    prompts = builder.build_all_prompts(story)
    for p in prompts:
        print(f"\n  Scene {p.get('scene_number')}:")
        print(f"  Prompt: {p['prompt'][:150]}...")

    return story


def process_upload_queue(config: Config):
    """Process pending uploads."""
    from modules.upload_manager import YouTubeUploader

    logger = logging.getLogger('main')
    logger.info("Processing upload queue...")

    uploader = YouTubeUploader(config)

    # Authenticate
    secrets = os.environ.get('YOUTUBE_CLIENT_SECRETS', 'client_secrets.json')
    if not os.path.exists(secrets):
        logger.error(f"YouTube client secrets not found: {secrets}")
        logger.info("Download from Google Cloud Console:")
        logger.info("  https://console.cloud.google.com/apis/credentials")
        return False

    if not uploader.authenticate(secrets):
        return False

    uploader.process_queue()
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Tiny Dino AI Shorts Factory v2.0.1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Generate 1 video
  python main.py -c 5               # Generate 5 videos
  python main.py --story-only       # Test story generation
  python main.py --check            # Check system readiness
  python main.py --status           # Show budget and stats
  python main.py --upload           # Process upload queue
        """
    )

    parser.add_argument('-c', '--count', type=int, default=1,
                        help='Number of videos to generate (default: 1)')
    parser.add_argument('--story-only', action='store_true',
                        help='Generate story only, no video')
    parser.add_argument('--check', action='store_true',
                        help='Check system readiness')
    parser.add_argument('--status', action='store_true',
                        help='Show current status and budget')
    parser.add_argument('--upload', action='store_true',
                        help='Process upload queue')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml override')

    args = parser.parse_args()

    # Load configuration
    config_path = args.config or os.environ.get('DINOFACTORY_CONFIG')
    config = Config(config_path)

    # Setup logging
    logger = setup_logging(config)
    logger.info("=" * 60)
    logger.info("TINY DINO AI SHORTS FACTORY v2.0.1")
    logger.info("=" * 60)

    # Execute command
    if args.check:
        success = check_system(config)
        sys.exit(0 if success else 1)

    elif args.status:
        show_status(config)
        return

    elif args.upload:
        success = process_upload_queue(config)
        sys.exit(0 if success else 1)

    elif args.story_only:
        generate_story_only(config)
        return

    else:
        # Check system first
        if not check_system(config):
            logger.error("System not ready. Run with --check for details.")
            sys.exit(1)

        # Generate videos
        pipeline = GenerationPipeline(config)
        results = pipeline.run_batch(args.count)

        # Print summary
        success_count = sum(1 for r in results if r['success'])
        print(f"\n{'='*60}")
        print(f"COMPLETE: {success_count}/{len(results)} videos generated successfully")
        for r in results:
            status = "OK" if r['success'] else "FAIL"
            path = r['path'] or "N/A"
            print(f"  [{status}] Video {r['index']}: {path}")
        print(f"{'='*60}")

        # Show final budget
        show_status(config)

        sys.exit(0 if success_count > 0 else 1)


if __name__ == '__main__':
    main()
