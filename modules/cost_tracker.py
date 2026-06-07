# modules/cost_tracker.py
"""Cost tracking and budget control for Tiny Dino Factory."""

import logging
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path

from .config_loader import Config

logger = logging.getLogger('cost_tracker')


class CostTracker:
    """Tracks GPU usage and costs against monthly budget."""

    STATS_FILE = "data/generation_stats.json"

    # Cost estimates per minute of GPU time (₹INR)
    COST_RATES = {
        'image_generation': 0.4,   # FLUX.1 Dev
        'video_generation': 1.2,   # LTX-2.3
        'audio_generation': 0.2,   # Stable Audio Open
        'llm_generation': 0.1,     # Qwen3 via Ollama (minimal)
        'assembly': 0.05,          # FFmpeg (CPU mostly)
    }

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.stats_file = os.path.join(
            self.config.get('project_root') or '.', self.STATS_FILE
        )
        self.budget_config = self.config.budget
        self.monthly_budget = self.budget_config.get('monthly_budget_inr', 1000)
        self.stop_when_exceeded = self.budget_config.get('stop_when_exceeded', True)
        self.warning_threshold = self.budget_config.get('warning_threshold_percent', 80)

        self._stats = self._load_stats()
        self._session_start = time.time()
        self._session_gpu_minutes = 0.0
        self._current_video_cost = 0.0

    def _load_stats(self) -> Dict[str, Any]:
        """Load generation statistics."""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                # Reset monthly stats if new month
                last_month = stats.get('last_reset_month', '')
                current_month = datetime.now().strftime('%Y-%m')
                if last_month != current_month:
                    stats['monthly_gpu_minutes'] = 0
                    stats['monthly_cost_inr'] = 0
                    stats['monthly_videos'] = 0
                    stats['last_reset_month'] = current_month
                return stats
            except json.JSONDecodeError:
                pass

        return {
            'videos_generated': 0,
            'gpu_minutes': 0,
            'estimated_cost_inr': 0,
            'failed_generations': 0,
            'retried_scenes': 0,
            'monthly_gpu_minutes': 0,
            'monthly_cost_inr': 0,
            'monthly_videos': 0,
            'last_reset_month': datetime.now().strftime('%Y-%m'),
            'generation_history': [],
            'first_run': datetime.now().isoformat(),
        }

    def _save_stats(self):
        """Save statistics to file."""
        os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(self._stats, f, indent=2)

    def record_gpu_time(self, task_type: str, minutes: float):
        """Record GPU usage for a task."""
        rate = self.COST_RATES.get(task_type, 0.1)
        cost = minutes * rate

        self._session_gpu_minutes += minutes
        self._current_video_cost += cost

        self._stats['gpu_minutes'] += minutes
        self._stats['monthly_gpu_minutes'] += minutes
        self._stats['estimated_cost_inr'] += cost
        self._stats['monthly_cost_inr'] += cost

        logger.debug(f"GPU time: {task_type} = {minutes:.2f}min, cost: ₹{cost:.2f}")

    def record_video_complete(self, video_path: str, story_title: str = ""):
        """Record successful video generation."""
        self._stats['videos_generated'] += 1
        self._stats['monthly_videos'] += 1

        entry = {
            'timestamp': datetime.now().isoformat(),
            'title': story_title,
            'file': os.path.basename(video_path),
            'cost_inr': round(self._current_video_cost, 2),
            'gpu_minutes': round(self._session_gpu_minutes, 2),
        }
        self._stats['generation_history'].append(entry)

        # Keep only last 100 entries
        self._stats['generation_history'] = self._stats['generation_history'][-100:]

        self._save_stats()

        logger.info(
            f"Video complete: '{story_title}' | "
            f"Cost: ₹{self._current_video_cost:.2f} | "
            f"Session GPU: {self._session_gpu_minutes:.1f}min"
        )

        # Reset session tracking
        self._current_video_cost = 0.0
        self._session_gpu_minutes = 0.0
        self._session_start = time.time()

    def record_failure(self, stage: str, error: str):
        """Record a generation failure."""
        self._stats['failed_generations'] += 1
        self._save_stats()
        logger.warning(f"Generation failure at {stage}: {error}")

    def record_retry(self, stage: str):
        """Record a retry attempt."""
        self._stats['retried_scenes'] += 1
        self._save_stats()
        logger.info(f"Retry at stage: {stage}")

    def check_budget(self) -> Dict[str, Any]:
        """Check current budget status."""
        monthly_cost = self._stats['monthly_cost_inr']
        budget_pct = (monthly_cost / self.monthly_budget * 100) if self.monthly_budget > 0 else 0

        status = {
            'monthly_budget_inr': self.monthly_budget,
            'spent_inr': round(monthly_cost, 2),
            'remaining_inr': round(self.monthly_budget - monthly_cost, 2),
            'percentage_used': round(budget_pct, 1),
            'can_generate': True,
            'warning': False,
            'message': '',
        }

        if budget_pct >= 100:
            status['can_generate'] = not self.stop_when_exceeded
            status['message'] = f"BUDGET EXHAUSTED: ₹{monthly_cost:.2f} / ₹{self.monthly_budget:.2f}"
            logger.error(status['message'])
        elif budget_pct >= self.warning_threshold:
            status['warning'] = True
            status['message'] = (
                f"Budget warning: {budget_pct:.1f}% used "
                f"(₹{monthly_cost:.2f} / ₹{self.monthly_budget:.2f})"
            )
            logger.warning(status['message'])
        else:
            status['message'] = (
                f"Budget OK: {budget_pct:.1f}% used "
                f"(₹{monthly_cost:.2f} / ₹{self.monthly_budget:.2f})"
            )

        return status

    def can_afford(self, estimated_cost_inr: float) -> bool:
        """Check if we can afford a generation."""
        status = self.check_budget()
        if not status['can_generate']:
            return False
        return (status['remaining_inr'] - estimated_cost_inr) >= 0

    def get_estimate(self, num_scenes: int) -> Dict[str, float]:
        """Estimate cost for a video with given scene count."""
        # Typical timing per scene
        img_time = 0.5  # minutes
        vid_time = 2.0  # minutes
        aud_time = 0.3  # minutes

        total_img = img_time * num_scenes
        total_vid = vid_time * num_scenes
        total_aud = aud_time * num_scenes

        cost_img = total_img * self.COST_RATES['image_generation']
        cost_vid = total_vid * self.COST_RATES['video_generation']
        cost_aud = total_aud * self.COST_RATES['audio_generation']
        cost_llm = 0.1  # story generation
        cost_asm = 0.05  # assembly

        total_cost = cost_img + cost_vid + cost_aud + cost_llm + cost_asm
        total_gpu = total_img + total_vid + total_aud

        return {
            'image_gpu_min': round(total_img, 1),
            'video_gpu_min': round(total_vid, 1),
            'audio_gpu_min': round(total_aud, 1),
            'total_gpu_min': round(total_gpu, 1),
            'image_cost_inr': round(cost_img, 2),
            'video_cost_inr': round(cost_vid, 2),
            'audio_cost_inr': round(cost_aud, 2),
            'llm_cost_inr': round(cost_llm, 2),
            'assembly_cost_inr': round(cost_asm, 2),
            'total_cost_inr': round(total_cost, 2),
        }

    def get_stats_report(self) -> str:
        """Generate formatted statistics report."""
        s = self._stats
        budget = self.check_budget()

        lines = [
            "=" * 60,
            "TINY DINO FACTORY - COST REPORT",
            "=" * 60,
            f"Monthly Budget:       ₹{self.monthly_budget:.2f}",
            f"Monthly Spent:        ₹{budget['spent_inr']:.2f} ({budget['percentage_used']}%)",
            f"Monthly Remaining:    ₹{budget['remaining_inr']:.2f}",
            "-" * 60,
            f"Total Videos:         {s['videos_generated']}",
            f"Monthly Videos:       {s['monthly_videos']}",
            f"Total GPU Minutes:    {s['gpu_minutes']:.1f}",
            f"Monthly GPU Minutes:  {s['monthly_gpu_minutes']:.1f}",
            f"Total Cost:           ₹{s['estimated_cost_inr']:.2f}",
            f"Failures:             {s['failed_generations']}",
            f"Retries:              {s['retried_scenes']}",
            "=" * 60,
        ]

        if s['generation_history']:
            lines.append("\nRecent Generations:")
            for entry in s['generation_history'][-5:]:
                lines.append(
                    f"  • {entry['title'][:40]:40s} | "
                    f"₹{entry['cost_inr']:.2f} | {entry['timestamp'][:10]}"
                )

        return "\n".join(lines)
