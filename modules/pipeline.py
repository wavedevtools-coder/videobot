# modules/pipeline.py
"""Main orchestration engine for Tiny Dino AI Shorts Factory."""

import logging
import os
import sys
import time
import shutil
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from .config_loader import Config
from .character_manager import CharacterManager
from .story_scorer import StoryScorer
from .story_generator import StoryGenerator
from .scene_generator import ScenePromptBuilder
from .image_generator import FLUXImageGenerator
from .video_generator import LTXVideoGenerator
from .audio_generator import StableAudioGenerator
from .assembly_engine import AssemblyEngine
from .upload_manager import YouTubeUploader
from .cost_tracker import CostTracker
from .quality_checker import QualityChecker

logger = logging.getLogger('pipeline')


class PipelineStage:
    """Represents a pipeline stage with retry logic."""

    def __init__(self, name: str, max_retries: int = 3):
        self.name = name
        self.max_retries = max_retries
        self.attempts = 0
        self.success = False
        self.error = None
        self.result = None
        self.duration = 0.0


class GenerationPipeline:
    """End-to-end pipeline orchestrating video generation."""

    STAGES = [
        'story_generation',
        'scene_prompting',
        'image_generation',
        'video_generation',
        'audio_generation',
        'assembly',
        'quality_check',
        'upload',
    ]

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.char_manager = CharacterManager(self.config)
        self.scorer = StoryScorer(self.config.quality.get('min_story_score', 70))
        self.story_gen = StoryGenerator(self.config)
        self.scene_builder = ScenePromptBuilder(self.config)
        self.image_gen = FLUXImageGenerator(self.config)
        self.video_gen = LTXVideoGenerator(self.config)
        self.audio_gen = StableAudioGenerator(self.config)
        self.assembler = AssemblyEngine(self.config)
        self.uploader = YouTubeUploader(self.config)
        self.cost_tracker = CostTracker(self.config)
        self.quality = QualityChecker(self.config)

        self.project_root = self.config.get('project_root') or '.'
        self.output_dir = os.path.join(self.project_root, 'output', 'videos')
        self.temp_dir = os.path.join(self.project_root, 'output', 'temp')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        self._stage_results: Dict[str, PipelineStage] = {}
        self._current_video = 0
        self._episode = self._get_next_episode()

    def _get_next_episode(self) -> int:
        """Get next episode number from history."""
        history_file = os.path.join(self.project_root, 'data', 'generation_stats.json')
        if os.path.exists(history_file):
            import json
            with open(history_file, 'r') as f:
                stats = json.load(f)
            return stats.get('videos_generated', 0) + 1
        return 1

    def _create_temp_workspace(self) -> str:
        """Create temporary workspace for current video."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        workspace = os.path.join(self.temp_dir, f"video_{timestamp}")
        os.makedirs(workspace, exist_ok=True)
        os.makedirs(os.path.join(workspace, 'images'), exist_ok=True)
        os.makedirs(os.path.join(workspace, 'videos'), exist_ok=True)
        os.makedirs(os.path.join(workspace, 'audio'), exist_ok=True)
        return workspace

    def _cleanup_workspace(self, workspace: str):
        """Clean up temporary workspace."""
        if self.config.generation.get('temp_cleanup', True):
            try:
                shutil.rmtree(workspace, ignore_errors=True)
                logger.debug(f"Cleaned workspace: {workspace}")
            except Exception as e:
                logger.warning(f"Workspace cleanup failed: {e}")

    def _save_progress(self, stage: str, data: Dict[str, Any]):
        """Save pipeline progress for crash recovery."""
        progress_file = os.path.join(self.temp_dir, 'pipeline_progress.json')
        progress = {
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'episode': self._episode,
            'data': data,
        }
        import json
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def run_stage(self, name: str, func, *args, **kwargs) -> PipelineStage:
        """Execute a pipeline stage with retries."""
        stage = PipelineStage(name, self.config.generation.get('max_retry', 3))
        logger.info(f"Starting stage: {name}")

        for attempt in range(1, stage.max_retries + 1):
            stage.attempts = attempt
            start = time.time()

            try:
                result = func(*args, **kwargs)
                stage.result = result
                stage.success = True
                stage.duration = time.time() - start
                logger.info(f"Stage {name} complete in {stage.duration:.1f}s")
                return stage

            except Exception as e:
                stage.error = str(e)
                stage.duration = time.time() - start
                self.cost_tracker.record_retry(name)
                logger.warning(f"Stage {name} attempt {attempt} failed: {e}")

                if attempt < stage.max_retries:
                    wait = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait}s...")
                    time.sleep(wait)

        logger.error(f"Stage {name} failed after {stage.max_retries} attempts")
        self.cost_tracker.record_failure(name, stage.error)
        return stage

    def generate_single_video(self, save_workspace: bool = False) -> Optional[str]:
        """Generate one complete video through all stages."""
        video_id = f"ep{self._episode:03d}"
        logger.info(f"{'='*60}")
        logger.info(f"GENERATING VIDEO: {video_id}")
        logger.info(f"{'='*60}")

        workspace = self._create_temp_workspace()
        story = None
        output_video = None

        try:
            # --- Stage 1: Story Generation ---
            budget_status = self.cost_tracker.check_budget()
            if not budget_status['can_generate']:
                logger.error(f"Cannot generate: {budget_status['message']}")
                return None

            stage = self.run_stage('story_generation', self.story_gen.generate)
            if not stage.success:
                return None
            story = stage.result
            self._save_progress('story', {'title': story.get('title', '')})

            # Log story score
            score = story.get('_score', {})
            logger.info(
                f"Story '{story.get('title')}' | "
                f"Score: {score.get('total', 0)}/100"
            )

            # Estimate cost
            num_scenes = len(story.get('scenes', []))
            estimate = self.cost_tracker.get_estimate(num_scenes)
            if not self.cost_tracker.can_afford(estimate['total_cost_inr']):
                logger.error(f"Cannot afford: need ₹{estimate['total_cost_inr']}, have ₹{budget_status['remaining_inr']}")
                return None

            logger.info(f"Cost estimate: ₹{estimate['total_cost_inr']:.2f}")

            # --- Stage 2: Scene Prompting ---
            stage = self.run_stage(
                'scene_prompting',
                self.scene_builder.build_all_prompts,
                story
            )
            if not stage.success:
                return None
            prompts = stage.result
            self._save_progress('prompts', {'count': len(prompts)})

            # --- Stage 3: Image Generation ---
            img_dir = os.path.join(workspace, 'images')
            stage = self.run_stage(
                'image_generation',
                self.image_gen.generate_scenes,
                prompts, img_dir,
                base_seed=42 + self._episode
            )
            if not stage.success:
                return None
            image_paths = stage.result
            self.cost_tracker.record_gpu_time('image_generation', estimate['image_gpu_min'])
            self._save_progress('images', {'count': len(image_paths)})

            # --- Stage 4: Video Generation ---
            vid_dir = os.path.join(workspace, 'videos')
            stage = self.run_stage(
                'video_generation',
                self.video_gen.generate_from_scenes,
                image_paths, prompts, vid_dir,
                base_seed=42 + self._episode
            )
            if not stage.success:
                return None
            scene_videos = stage.result
            self.cost_tracker.record_gpu_time('video_generation', estimate['video_gpu_min'])
            self._save_progress('videos', {'count': len(scene_videos)})

            # --- Stage 5: Audio Generation ---
            aud_dir = os.path.join(workspace, 'audio')
            stage = self.run_stage(
                'audio_generation',
                self.audio_gen.generate_sfx_mix,
                story.get('scenes', []), aud_dir,
                base_seed=42 + self._episode
            )
            if not stage.success:
                return None
            sfx_paths = stage.result
            self.cost_tracker.record_gpu_time('audio_generation', estimate['audio_gpu_min'])
            self._save_progress('audio', {'count': len(sfx_paths)})

            # Generate background music
            total_duration = sum(s.get('duration_seconds', 5) for s in story.get('scenes', []))
            bgm_path = os.path.join(aud_dir, 'background_music.wav')
            try:
                self.audio_gen.generate_background_music(
                    story_duration=total_duration,
                    output_path=bgm_path,
                    seed=42 + self._episode,
                )
            except Exception as e:
                logger.warning(f"Background music generation failed: {e}")
                bgm_path = None

            # --- Stage 6: Assembly ---
            output_video = os.path.join(self.output_dir, f"tiny_dino_{video_id}.mp4")
            stage = self.run_stage(
                'assembly',
                self.assembler.assemble,
                scene_videos=scene_videos,
                audio_path=bgm_path,
                output_path=output_video,
                title=story.get('title', 'Tiny Dino'),
                episode=self._episode,
            )
            if not stage.success:
                return None
            self.cost_tracker.record_gpu_time('assembly', 0.05)
            self._save_progress('assembled', {'path': output_video})

            # --- Stage 7: Quality Check ---
            stage = self.run_stage('quality_check', self.quality.check, output_video)
            if stage.success and stage.result and stage.result.passed:
                logger.info("Quality check PASSED")
            else:
                logger.warning("Quality check warnings present")
                if stage.result and stage.result.feedback:
                    for fb in stage.result.feedback:
                        logger.warning(f"  - {fb}")
                # Continue anyway if basic checks pass

            # Record successful generation
            self.cost_tracker.record_video_complete(
                output_video,
                story.get('title', 'Unknown')
            )

            logger.info(f"Video generated: {output_video}")
            self._episode += 1

            # --- Stage 8: Upload (if enabled) ---
            upload_cfg = self.config.upload
            if upload_cfg.get('enabled', False):
                metadata = self.uploader.generate_metadata(story, self._episode - 1)
                self.uploader.schedule_upload(output_video, metadata)
                logger.info("Video scheduled for upload")

            # Save after each video (batch safety)
            if self.config.generation.get('save_after_each_video', True):
                logger.info("Progress saved (save_after_each_video enabled)")

            return output_video

        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")
            self.cost_tracker.record_failure('pipeline', str(e))
            return None

        finally:
            if not save_workspace:
                self._cleanup_workspace(workspace)

    def run_batch(self, count: Optional[int] = None):
        """Run batch generation."""
        count = count or self.config.generation.get('videos_per_run', 1)
        logger.info(f"Starting batch: {count} video(s)")

        results = []
        for i in range(count):
            logger.info(f"\n--- Video {i+1}/{count} ---")

            # Check budget before each video
            budget = self.cost_tracker.check_budget()
            if not budget['can_generate']:
                logger.error(f"Stopping batch: {budget['message']}")
                break

            video_path = self.generate_single_video()
            results.append({
                'index': i + 1,
                'path': video_path,
                'success': video_path is not None,
            })

            if video_path is None and i < count - 1:
                logger.info("Waiting 10s before next attempt...")
                time.sleep(10)

        # Summary
        success_count = sum(1 for r in results if r['success'])
        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH COMPLETE: {success_count}/{count} videos generated")
        logger.info(f"{'='*60}")

        return results

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        budget = self.cost_tracker.check_budget()
        return {
            'episode': self._episode,
            'budget': budget,
            'stats': self.cost_tracker._stats,
        }
