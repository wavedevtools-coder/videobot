# modules/assembly_engine.py
"""Video assembly using FFmpeg - combines scenes, audio, outro, and branding.

Changes v2.1.0:
  - Watermark: added bottom-right overlay from assets/watermark.png
  - Dino character reference: dinocharacter.png passed to image prompts via config
  - Outro: auto-generate from outro_ref.png if outro.mp4 is missing, then merge
"""

import logging
import os
import re
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .config_loader import Config
from .character_manager import CharacterManager

logger = logging.getLogger('assembly_engine')


def _ffmpeg_path(path: str) -> str:
    """Normalize a filesystem path for FFmpeg/ffprobe.

    On Windows: forward slashes and backslashes both work for FFmpeg.
    On Linux:   Windows drive-letter paths (D:/foo) are mapped to /mnt/d/foo
                (standard WSL mount convention). Regular Linux paths are
                resolved to absolute paths.
    """
    import re
    if not path:
        return path
    normalized = path.replace('\\', '/')
    drive_match = re.match(r'^([A-Za-z]):/(.+)$', normalized)
    if drive_match:
        drive, rest = drive_match.group(1), drive_match.group(2)
        if os.name == 'nt':
            # Windows: return with forward slashes (FFmpeg handles both)
            return f'{drive.upper()}:/{rest}'
        else:
            # Linux/WSL: map D:/foo → /mnt/d/foo
            return f'/mnt/{drive.lower()}/{rest}'
    # Regular path: resolve to absolute
    return str(Path(path).resolve())


class AssemblyEngine:
    """Assembles final video from scenes using FFmpeg."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.video_cfg = self.config.video
        self.assets = self.config.section('assets')
        self.char_manager = CharacterManager(self.config)
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        """Verify FFmpeg is available."""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                version = result.stdout.split('\n')[0]
                logger.info(f"FFmpeg found: {version}")
            else:
                raise RuntimeError("FFmpeg not working properly")
        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg not found! Install FFmpeg:\n"
                "  Windows: choco install ffmpeg\n"
                "  Linux: sudo apt install ffmpeg\n"
                "  Mac: brew install ffmpeg"
            )

    def _get_video_info(self, path: str) -> Dict[str, Any]:
        """Get video file information using ffprobe."""
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_format', '-show_streams',
                 '-of', 'json', _ffmpeg_path(path)],
                capture_output=True, text=True, timeout=30
            )
            return json.loads(result.stdout)
        except Exception as e:
            logger.error(f"ffprobe failed for {path}: {e}")
            return {}

    # ------------------------------------------------------------------
    # WATERMARK HELPER
    # ------------------------------------------------------------------
    def _build_watermark_filter(
        self,
        watermark_path: str,
        margin: int = 20,
        scale: str = "120:-1",
    ) -> str:
        """
        Return the overlay filter string for bottom-right watermark.
        The watermark is scaled to 120px wide (keeping aspect ratio)
        and placed margin px from the right and bottom edges.
        """
        return (
            f"[wm_in]scale={scale}[wm_scaled];"
            f"[base][wm_scaled]overlay="
            f"x=W-w-{margin}:y=H-h-{margin}:format=auto[out]"
        )

    # ------------------------------------------------------------------
    # OUTRO HELPERS
    # ------------------------------------------------------------------
    def _generate_outro_from_ref(
        self,
        outro_video: str,
        outro_ref: str,
        duration: int = 4,
    ) -> Optional[str]:
        """
        Build a branded outro MP4 from outro_ref.png using FFmpeg.
        Adds fade-in / fade-out and a 'Subscribe!' text overlay.
        """
        w = self.video_cfg.get('width', 720)
        h = self.video_cfg.get('height', 1280)
        fps = self.video_cfg.get('fps', 30)

        logger.info(f"Generating outro from reference image: {outro_ref}")

        # Scale + pad the reference image to fill 9:16 frame
        # NOTE: use long-form fade params (start_time/duration) — some FFmpeg
        # builds reject the short forms st= and d= on certain filter chains.
        # Strip emoji from drawtext — headless Linux containers often lack emoji fonts.
        vf = (
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"drawtext=text='Subscribe for more Tiny Dino':"
            f"fontcolor=white:fontsize=44:"
            f"x=(w-text_w)/2:y=h*0.82:"
            f"box=1:boxcolor=black@0.45:boxborderw=12,"
            f"fade=t=in:start_time=0:duration=0.5,"
            f"fade=t=out:start_time={duration - 0.6}:duration=0.5"
        )

        outro_dir = os.path.dirname(_ffmpeg_path(outro_video))
        os.makedirs(outro_dir, exist_ok=True)

        cmd = [
            'ffmpeg', '-y',
            '-loop', '1',
            '-i', _ffmpeg_path(outro_ref),
            '-vf', vf,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-t', str(duration),
            '-r', str(fps),
            '-pix_fmt', 'yuv420p',
            _ffmpeg_path(outro_video),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                logger.error(f"Outro generation failed: {result.stderr[-500:]}")
                return None
            logger.info(f"Outro generated: {outro_video}")
            return outro_video
        except Exception as e:
            logger.error(f"Outro generation error: {e}")
            return None

    def _get_or_generate_outro(self) -> Optional[str]:
        """
        Return path to outro MP4.
        Priority:
          1. Cached outro.mp4 (metadata still valid)  →  reuse
          2. outro_ref.png exists                      →  generate from it
          3. Fallback FFmpeg gradient outro            →  legacy generator
        """
        outro_video    = self.assets.get('outro_video',    'assets/outro.mp4')
        outro_metadata = self.assets.get('outro_metadata', 'assets/outro.json')
        outro_ref      = self.assets.get('outro_ref',      'assets/outro_ref.png')

        current_meta = self.char_manager.get_outro_metadata()
        current_meta['outro_version'] = '1.1'   # bump when logic changes

        # ── 1. try cached ──────────────────────────────────────────────
        if os.path.exists(outro_metadata) and os.path.exists(outro_video):
            try:
                with open(outro_metadata, 'r') as f:
                    cached_meta = json.load(f)
                if cached_meta == current_meta:
                    logger.info("Using cached outro (metadata matches)")
                    return outro_video
                logger.info("Outro metadata changed, regenerating...")
            except (json.JSONDecodeError, KeyError):
                logger.warning("Invalid outro metadata, regenerating...")

        # ── 2. generate from reference image ──────────────────────────
        if os.path.exists(outro_ref):
            duration = self.video_cfg.get('outro_duration', 4)
            result = self._generate_outro_from_ref(outro_video, outro_ref, duration)
            if result:
                self._save_outro_metadata(outro_metadata, current_meta)
                return result

        # ── 3. legacy gradient fallback ───────────────────────────────
        return self._generate_outro(outro_video, outro_metadata, current_meta)

    def _save_outro_metadata(self, path: str, meta: Dict) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------
    # MAIN FFMPEG COMMAND BUILDER
    # ------------------------------------------------------------------
    def _build_ffmpeg_command(
        self,
        scene_videos: List[str],
        audio_path: Optional[str],
        outro_path: Optional[str],
        output_path: str,
        title: str = "",
        episode: int = 0,
    ) -> List[str]:
        """Build FFmpeg command for final assembly with watermark."""
        w   = self.video_cfg.get('width',  720)
        h   = self.video_cfg.get('height', 1280)
        fps = self.video_cfg.get('fps',    30)
        crf = self.video_cfg.get('crf',    23)
        preset = self.video_cfg.get('preset', 'medium')

        watermark_path = self.assets.get('watermark', 'assets/watermark.png')
        use_watermark  = os.path.exists(watermark_path)

        inputs = []
        filter_parts = []

        # ── inputs ────────────────────────────────────────────────────
        for sv in scene_videos:
            inputs.extend(['-i', _ffmpeg_path(sv)])

        if audio_path and os.path.exists(audio_path):
            inputs.extend(['-i', _ffmpeg_path(audio_path)])
            audio_idx = len(scene_videos)
        else:
            audio_idx = None

        if outro_path and os.path.exists(outro_path):
            inputs.extend(['-i', _ffmpeg_path(outro_path)])
            outro_idx = len(scene_videos) + (1 if audio_idx is not None else 0)
        else:
            outro_idx = None

        # Watermark gets its own input index
        if use_watermark:
            inputs.extend(['-i', _ffmpeg_path(watermark_path)])
            wm_idx = len(scene_videos) \
                   + (1 if audio_idx is not None else 0) \
                   + (1 if outro_idx is not None else 0)
        else:
            wm_idx = None

        # ── video filter chain ────────────────────────────────────────
        # Step 1: concatenate scene clips
        if len(scene_videos) > 1:
            concat_in = ''.join([f'[{i}:v:0]' for i in range(len(scene_videos))])
            filter_parts.append(
                f'{concat_in}concat=n={len(scene_videos)}:v=1:a=0[mainv]'
            )
            last_video = 'mainv'
        else:
            # Single scene: pull into filtergraph via copy so -map [vout] resolves.
            # Using a raw stream specifier like '0:v:0' with brackets in -map
            # makes FFmpeg look for a filter output label — it doesn't exist.
            filter_parts.append('[0:v:0]copy[vout]')
            last_video = 'vout'

        # Step 2: append outro
        if outro_idx is not None:
            filter_parts.append(
                f'[{last_video}][{outro_idx}:v:0]concat=n=2:v=1:a=0[withoutro]'
            )
            last_video = 'withoutro'

        # Step 3: title overlay (optional)
        # NOTE: apostrophes in filter_complex text= values break FFmpeg's parser.
        # We strip them and split drawtext + fade into separate filter steps.
        if title:
            # Remove characters that break FFmpeg filter_complex string parsing
            title_safe = title.replace("'", "").replace('"', '').replace('\\', '')
            filter_parts.append(
                f'[{last_video}]drawtext='
                f'text={title_safe}:'
                f'fontcolor=white:fontsize=48:'
                f'x=(w-text_w)/2:y=(h-text_h)/2:'
                f'box=1:boxcolor=black@0.5:boxborderw=10[title_drawn]'
            )
            filter_parts.append(
                f'[title_drawn]fade=t=out:start_time=1.5:duration=0.5[titled]'
            )
            last_video = 'titled'

        # Step 4: watermark — bottom-right corner
        if wm_idx is not None:
            margin = 20
            filter_parts.append(
                f'[{wm_idx}:v]scale=120:-1[wm_scaled]'
            )
            filter_parts.append(
                f'[{last_video}][wm_scaled]overlay='
                f'x=W-w-{margin}:y=H-h-{margin}:format=auto[watermarked]'
            )
            last_video = 'watermarked'

        # ── audio chain ───────────────────────────────────────────────
        # Real audio: loop it to cover the full video length
        if audio_idx is not None:
            filter_parts.append(
                f'[{audio_idx}:a:0]aloop=loop=-1:size=2e+09[audio]'
            )
            last_audio = 'audio'
            silence_input = None
        else:
            # No audio file — inject a silent anullsrc as a lavfi input.
            # aevalsrc in filter_complex only generated 1 s of silence which
            # caused -shortest to cut the whole video to 1 second.
            silence_input = ['-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo']
            # anullsrc is the last input; add its index after scene/outro/wm inputs
            silence_idx = len(scene_videos) \
                        + (1 if audio_idx is not None else 0) \
                        + (1 if outro_idx is not None else 0) \
                        + (1 if wm_idx is not None else 0)
            filter_parts.append(
                f'[{silence_idx}:a]aresample=44100[silence]'
            )
            last_audio = 'silence'

        # ── assemble command ──────────────────────────────────────────
        cmd = ['ffmpeg', '-y']
        cmd.extend(inputs)
        # Insert the lavfi silence input BEFORE filter_complex (after real inputs)
        if silence_input:
            cmd.extend(silence_input)

        if filter_parts:
            cmd.extend(['-filter_complex', ';'.join(filter_parts)])

        cmd.extend(['-map', f'[{last_video}]'])
        cmd.extend(['-map', f'[{last_audio}]'])

        cmd.extend([
            '-c:v', 'libx264',
            '-preset', preset,
            '-crf', str(crf),
            '-r', str(fps),
            '-s', f'{w}x{h}',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-ar', '44100',
            '-shortest',
            '-movflags', '+faststart',
            _ffmpeg_path(output_path),
        ])

        return cmd

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def assemble(
        self,
        scene_videos: List[str],
        audio_path: Optional[str] = None,
        output_path: str = "",
        title: str = "",
        episode: int = 0,
        include_outro: bool = True,
    ) -> str:
        """Assemble final video: scenes + outro + watermark."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        valid_scenes = [s for s in scene_videos if os.path.exists(s)]
        if not valid_scenes:
            raise ValueError("No valid scene videos found")

        outro_path = None
        if include_outro:
            outro_path = self._get_or_generate_outro()
            if outro_path:
                logger.info(f"Outro will be merged: {outro_path}")
            else:
                logger.warning("Outro unavailable — skipping outro merge")

        cmd = self._build_ffmpeg_command(
            scene_videos=valid_scenes,
            audio_path=audio_path,
            outro_path=outro_path,
            output_path=output_path,
            title=title,
            episode=episode,
        )

        logger.info(f"Assembling video: {output_path}")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0:
                logger.error(f"FFmpeg stderr: {result.stderr[-1000:]}")
                raise RuntimeError(f"FFmpeg failed with code {result.returncode}")

            if not os.path.exists(output_path):
                raise RuntimeError("Output file not created")

            size_mb = os.path.getsize(output_path) / 1024 / 1024
            logger.info(f"Video assembled: {output_path} ({size_mb:.1f} MB)")
            return output_path

        except subprocess.TimeoutExpired:
            raise RuntimeError("FFmpeg assembly timed out after 600s")

    # ------------------------------------------------------------------
    # LEGACY OUTRO FALLBACK (gradient only, no ref image)
    # ------------------------------------------------------------------
    def _generate_outro(
        self,
        outro_video: str,
        outro_metadata: str,
        metadata: Dict[str, str],
    ) -> Optional[str]:
        """Generate simple gradient branded outro (fallback when no ref image)."""
        logger.info("Generating fallback gradient outro...")
        w = self.video_cfg.get('width', 720)
        h = self.video_cfg.get('height', 1280)
        duration = self.video_cfg.get('outro_duration', 4)
        fps = self.video_cfg.get('fps', 30)

        vf = (
            f"drawtext=text='Thanks for watching!':"
            f"fontcolor=white:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2-100:"
            f"box=1:boxcolor=black@0.3:boxborderw=10,"
            f"drawtext=text='Subscribe for more Tiny Dino!':"
            f"fontcolor=0xFFFACD:fontsize=36:x=(w-text_w)/2:y=(h-text_h)/2+50:"
            f"box=1:boxcolor=black@0.3:boxborderw=8,"
            f"fade=t=in:start_time=0:duration=0.5,"
            f"fade=t=out:start_time={duration-0.5}:duration=0.5"
        )

        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi', '-i', f'color=c=0x39FF14:s={w}x{h}:d={duration}',
            '-vf', vf,
            '-c:v', 'libx264', '-preset', 'fast',
            '-r', str(fps), '-pix_fmt', 'yuv420p',
            '-t', str(duration),
            _ffmpeg_path(outro_video),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                logger.error(f"Fallback outro failed: {result.stderr[-300:]}")
                return None
            self._save_outro_metadata(outro_metadata, metadata)
            logger.info(f"Fallback outro generated: {outro_video}")
            return outro_video
        except Exception as e:
            logger.error(f"Fallback outro error: {e}")
            return None

    # ------------------------------------------------------------------
    # SUBTITLES
    # ------------------------------------------------------------------
    def add_subtitles(
        self,
        video_path: str,
        scenes: List[Dict[str, Any]],
        output_path: str,
    ) -> str:
        """Add scene narration text overlays as burned-in subtitles."""
        subtitle_path = video_path.rsplit('.', 1)[0] + '.srt'
        with open(subtitle_path, 'w', encoding='utf-8') as f:
            current_time = 0.0
            for i, scene in enumerate(scenes):
                start    = current_time
                duration = scene.get('duration_seconds', 5)
                end      = start + duration
                desc     = scene.get('description', '')
                short    = desc[:80] + '...' if len(desc) > 80 else desc
                f.write(f"{i+1}\n")
                f.write(f"{self._format_time(start)} --> {self._format_time(end)}\n")
                f.write(f"{short}\n\n")
                current_time = end

        sub_path = _ffmpeg_path(subtitle_path).replace('\\', '/').replace(':', '\\:')
        cmd = [
            'ffmpeg', '-y',
            '-i', _ffmpeg_path(video_path),
            '-vf', f"subtitles='{sub_path}'",
            '-c:v', 'libx264', '-crf', '23', '-preset', 'medium',
            '-c:a', 'copy',
            _ffmpeg_path(output_path),
        ]
        subprocess.run(cmd, capture_output=True, timeout=120)
        return output_path

    @staticmethod
    def _format_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace('.', ',')

    def verify_output(self, path: str, expected_duration: float) -> bool:
        """Verify assembled video meets minimum quality requirements."""
        if not os.path.exists(path):
            return False
        info = self._get_video_info(path)
        if not info:
            return False
        duration = float(info.get('format', {}).get('duration', 0))
        if duration < expected_duration * 0.8:
            logger.error(f"Video too short: {duration}s < {expected_duration}s")
            return False
        for stream in info.get('streams', []):
            if stream.get('codec_type') == 'video':
                w = stream.get('width', 0)
                h = stream.get('height', 0)
                if w < 100 or h < 100:
                    logger.error(f"Invalid resolution: {w}x{h}")
                    return False
        size = os.path.getsize(path)
        if size < 100 * 1024:
            logger.error(f"Video file too small: {size} bytes")
            return False
        logger.info(f"Video verified: {duration:.1f}s, {w}x{h}, {size/1024/1024:.1f}MB")
        return True
