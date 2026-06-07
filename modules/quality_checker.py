# modules/quality_checker.py
"""Final quality validation before upload."""

import logging
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from .config_loader import Config

logger = logging.getLogger('quality_checker')


@dataclass
class QualityReport:
    """Quality check results."""
    file_exists: bool = False
    valid_size: bool = False
    valid_duration: bool = False
    valid_resolution: bool = False
    valid_fps: bool = False
    has_audio: bool = False
    no_black_frames: bool = False
    brightness_ok: bool = False
    passed: bool = False
    duration_seconds: float = 0.0
    resolution: str = ""
    fps: float = 0.0
    file_size_mb: float = 0.0
    mean_brightness: float = 0.0
    black_frame_count: int = 0
    feedback: List[str] = field(default_factory=list)


class QualityChecker:
    """Validates assembled video before upload."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.video_cfg = self.config.video
        self.min_duration = 15  # minimum 15 seconds
        self.max_duration = 90  # maximum 90 seconds for Shorts
        self.min_brightness = 15  # not too dark
        self.max_brightness = 250  # not washed out
        self.black_threshold = 10  # pixel value below this = black
        self.max_black_frames = 5  # allow up to 5 black frames

    def check(self, video_path: str) -> QualityReport:
        """Run full quality check on assembled video."""
        report = QualityReport()

        # 1. File exists and has size
        report.file_exists = os.path.exists(video_path)
        if not report.file_exists:
            report.feedback.append("Video file does not exist")
            return report

        file_size = os.path.getsize(video_path)
        report.file_size_mb = file_size / (1024 * 1024)
        report.valid_size = 0.1 < report.file_size_mb < 500  # 100KB - 500MB
        if not report.valid_size:
            report.feedback.append(f"Suspicious file size: {report.file_size_mb:.1f} MB")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            report.feedback.append("Cannot open video file")
            return report

        try:
            # 2. Check duration and FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            report.fps = fps
            report.duration_seconds = duration
            report.fps = fps

            report.valid_fps = 20 <= fps <= 60
            report.valid_duration = self.min_duration <= duration <= self.max_duration

            if not report.valid_duration:
                report.feedback.append(
                    f"Duration {duration:.1f}s outside range [{self.min_duration}-{self.max_duration}]s"
                )

            # 3. Check resolution
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            report.resolution = f"{w}x{h}"
            report.valid_resolution = w >= 360 and h >= 640  # minimum Shorts resolution

            if not report.valid_resolution:
                report.feedback.append(f"Resolution too low: {w}x{h}")

            # 4. Check for black frames and brightness
            frame_idx = 0
            brightness_values = []
            black_frames = 0
            sample_interval = max(1, frame_count // 50)  # sample ~50 frames

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_interval == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    mean_brightness = np.mean(gray)
                    brightness_values.append(mean_brightness)

                    if mean_brightness < self.black_threshold:
                        black_frames += 1

                frame_idx += 1

            if brightness_values:
                report.mean_brightness = np.mean(brightness_values)
                report.black_frame_count = black_frames
                report.no_black_frames = black_frames <= self.max_black_frames
                report.brightness_ok = self.min_brightness <= report.mean_brightness <= self.max_brightness

                if not report.no_black_frames:
                    report.feedback.append(f"Too many black frames: {black_frames}")
                if not report.brightness_ok:
                    report.feedback.append(
                        f"Brightness {report.mean_brightness:.1f} outside range"
                    )

            # 5. Check audio presence (file size heuristic)
            report.has_audio = report.file_size_mb > 1.0  # rough check

        finally:
            cap.release()

        # Overall pass/fail
        report.passed = all([
            report.file_exists,
            report.valid_size,
            report.valid_duration,
            report.valid_resolution,
            report.valid_fps,
            report.no_black_frames,
            report.brightness_ok,
        ])

        return report

    def quick_check(self, video_path: str) -> bool:
        """Quick 3-point check."""
        if not os.path.exists(video_path):
            return False
        if os.path.getsize(video_path) < 100 * 1024:  # 100KB
            return False

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count > 10

    def format_report(self, report: QualityReport) -> str:
        """Format quality report as readable string."""
        lines = [
            "=" * 50,
            "VIDEO QUALITY REPORT",
            "=" * 50,
            f"File Exists:      {'PASS' if report.file_exists else 'FAIL'}",
            f"File Size:        {'PASS' if report.valid_size else 'FAIL'} ({report.file_size_mb:.1f} MB)",
            f"Duration:         {'PASS' if report.valid_duration else 'FAIL'} ({report.duration_seconds:.1f}s)",
            f"Resolution:       {'PASS' if report.valid_resolution else 'FAIL'} ({report.resolution})",
            f"FPS:              {'PASS' if report.valid_fps else 'FAIL'} ({report.fps:.1f})",
            f"Audio Present:    {'PASS' if report.has_audio else 'WARN'}",
            f"Black Frames:     {'PASS' if report.no_black_frames else 'FAIL'} ({report.black_frame_count})",
            f"Brightness:       {'PASS' if report.brightness_ok else 'FAIL'} ({report.mean_brightness:.1f})",
            "-" * 50,
            f"OVERALL:          {'PASS' if report.passed else 'FAIL'}",
            "=" * 50,
        ]

        if report.feedback:
            lines.append("\nIssues Found:")
            for issue in report.feedback:
                lines.append(f"  - {issue}")

        return "\n".join(lines)
