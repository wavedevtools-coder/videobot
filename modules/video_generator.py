# modules/video_generator.py
"""Video generation using Wan 2.1 / 2.2 Image-to-Video.

Changes v3.0.0:
  - Replaced LTX-Video with Wan I2V for significantly better motion quality.
  - Supports both Wan2.1-I2V-14B-480P and Wan2.2-I2V models.
  - Generates at Wan-native resolution (480×832 portrait), upscales to target.
  - Frame count snapped to multiples of 4 (Wan requirement).
  - Native FPS: 16 (Wan default), upsampled to target FPS by FFmpeg in assembly.
  - Explicit VRAM cleanup between generations.
"""

import logging
import os
import torch
from typing import Optional, List, Dict, Any
from pathlib import Path

from .config_loader import Config

logger = logging.getLogger('video_generator')

DEFAULT_VIDEO_NEGATIVE = (
    "Bright light, overexposed, static, blurry details, subtitles, "
    "style, works, paintings, images, static, overall gray, worst quality, "
    "low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, "
    "three legs, many people in the background, walking backwards, "
    "morphing, distortion, flickering, inconsistent motion"
)


def _wan_num_frames(duration_seconds: float, fps: int) -> int:
    """Snap frame count to Wan requirement: multiples of 4, plus 1.

    Wan models expect frame counts of 4k+1 (e.g., 17, 21, 25, 33, 41, 49, 81).
    We clamp to a reasonable range to avoid OOM.
    """
    target = max(5, int(duration_seconds * fps))
    # Round up to nearest 4k+1
    k = max(1, (target - 1 + 3) // 4)
    frames = k * 4 + 1
    # Clamp to safe maximum (81 frames is ~5s at 16fps, good for shorts)
    return min(frames, 81)


class WanVideoGenerator:
    """Wan 2.1/2.2 Image-to-Video generation.

    Replaces LTXVideoGenerator with the same public API so the pipeline
    orchestrator requires only a class name swap.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.vid_config = self.config.models.get('video', {})
        self.model_id = self.vid_config.get('model', 'Wan-AI/Wan2.1-I2V-14B-480P')

        dtype_name = self.vid_config.get('dtype', 'bfloat16')
        self.dtype = {
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp16": torch.float16,  "float16":  torch.float16,
            "fp32": torch.float32,  "float32":  torch.float32,
        }.get(dtype_name, torch.bfloat16)

        self.num_steps = self.vid_config.get('num_inference_steps', 30)
        self.guidance_scale = self.vid_config.get('guidance_scale', 5.0)
        self.enable_vae_slicing = self.vid_config.get('enable_vae_slicing', True)
        self.target_fps = self.vid_config.get('target_fps', 16)

        # Wan-native generation resolution (portrait 9:16)
        self.gen_w = self.vid_config.get('max_area_width', 480)
        self.gen_h = self.vid_config.get('max_area_height', 832)

        self._pipe = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Pipeline loader
    # ------------------------------------------------------------------
    def _load_pipeline(self):
        """Lazy-load Wan I2V pipeline with memory optimizations."""
        if self._loaded:
            return

        try:
            from diffusers import WanImageToVideoPipeline

            logger.info(f"Loading Wan I2V model: {self.model_id}")

            self._pipe = WanImageToVideoPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
            )
            self._pipe.enable_model_cpu_offload()

            if self.enable_vae_slicing and hasattr(self._pipe, 'vae'):
                try:
                    self._pipe.vae.enable_slicing()
                    logger.info("Enabled VAE slicing for memory savings")
                except Exception:
                    pass

            if hasattr(self._pipe, 'vae') and hasattr(self._pipe.vae, 'enable_tiling'):
                try:
                    self._pipe.vae.enable_tiling()
                    logger.info("Enabled VAE tiling for memory savings")
                except Exception:
                    pass

            self._loaded = True
            logger.info(
                f"Wan I2V pipeline loaded: {type(self._pipe).__name__} | "
                f"model={self.model_id} | steps={self.num_steps} | "
                f"gen_res={self.gen_w}x{self.gen_h}"
            )

        except ImportError:
            raise RuntimeError(
                "diffusers >= 0.32.0 required for Wan I2V. Install with:\n"
                "  pip install diffusers>=0.32.0 transformers accelerate"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Wan I2V pipeline: {e}")

    # ------------------------------------------------------------------
    # Core generate()
    # ------------------------------------------------------------------
    def generate(
        self,
        image_path: str,
        prompt: str,
        output_path: str,
        num_frames: int = 41,
        fps: Optional[int] = None,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
    ) -> str:
        """Generate video from image using Wan I2V.

        Args:
            image_path: Path to the input scene image.
            prompt: Motion/scene description prompt.
            output_path: Where to save the output MP4.
            num_frames: Number of frames (will be snapped to 4k+1).
            fps: Output FPS (defaults to Wan native 16fps).
            seed: Optional random seed for reproducibility.
            negative_prompt: Optional negative prompt.

        Returns:
            Path to the saved video file.
        """
        self._load_pipeline()

        fps = fps or self.target_fps
        neg = negative_prompt or DEFAULT_VIDEO_NEGATIVE

        generator = None
        if seed is not None:
            generator = torch.Generator("cpu").manual_seed(seed)

        logger.info(
            f"Generating video from: {image_path} "
            f"({num_frames} frames @ {fps}fps, {self.gen_w}x{self.gen_h})"
        )
        torch.cuda.empty_cache()

        try:
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            orig_w, orig_h = image.size

            # Resize to Wan-native generation resolution
            if image.size != (self.gen_w, self.gen_h):
                logger.info(
                    f"Resizing image {orig_w}x{orig_h} → "
                    f"{self.gen_w}x{self.gen_h} for Wan generation"
                )
                image = image.resize((self.gen_w, self.gen_h), Image.LANCZOS)

            pipe_kwargs = dict(
                image=image,
                prompt=prompt,
                negative_prompt=neg,
                num_inference_steps=self.num_steps,
                guidance_scale=self.guidance_scale,
                num_frames=num_frames,
                width=self.gen_w,
                height=self.gen_h,
                generator=generator,
            )

            result = self._pipe(**pipe_kwargs)
            video = result.frames[0]

            # Upscale frames back to original/target resolution if needed
            target_w = self.config.video.get('width', 720)
            target_h = self.config.video.get('height', 1280)
            if (self.gen_w, self.gen_h) != (target_w, target_h):
                logger.info(
                    f"Upscaling generated frames "
                    f"{self.gen_w}x{self.gen_h} → {target_w}x{target_h}"
                )
                video = [
                    frame.resize((target_w, target_h), Image.LANCZOS)
                    if hasattr(frame, 'resize')
                    else frame
                    for frame in video
                ]

            self._save_video(video, output_path, fps)
            logger.info(f"Video saved: {output_path} ({len(video)} frames)")

            # Proactive VRAM cleanup after each generation
            torch.cuda.empty_cache()

            return output_path

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA OOM during Wan I2V generation!")
            torch.cuda.empty_cache()
            raise

    # ------------------------------------------------------------------
    # Video saving
    # ------------------------------------------------------------------
    def _save_video(self, frames: list, output_path: str, fps: int):
        """Save frames as MP4 using imageio with yuv420p for broad compatibility."""
        import numpy as np
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        try:
            import imageio
            writer = imageio.get_writer(
                output_path,
                fps=fps,
                codec='libx264',
                quality=8,
                pixelformat='yuv420p',
                macro_block_size=1,
            )
            for frame in frames:
                if not isinstance(frame, np.ndarray):
                    frame = np.array(frame)
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                writer.append_data(frame)
            writer.close()
        except ImportError:
            logger.warning("imageio not available, trying cv2 fallback")
            self._save_video_cv2(frames, output_path, fps)

    def _save_video_cv2(self, frames: list, output_path: str, fps: int):
        """Fallback video saving with OpenCV."""
        import cv2
        import numpy as np

        if not frames:
            raise ValueError("No frames to save")

        first = frames[0]
        if hasattr(first, 'shape'):
            h, w = first.shape[:2]
        elif hasattr(first, 'size'):
            w, h = first.size
        else:
            h, w = 1280, 720

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for frame in frames:
            if not isinstance(frame, np.ndarray):
                arr = np.array(frame)
            elif hasattr(frame, 'numpy'):
                arr = frame.numpy()
            else:
                arr = frame
            # Convert RGB to BGR for OpenCV
            if arr.ndim == 3 and arr.shape[-1] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            writer.write(arr)

        writer.release()

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------
    def generate_from_scenes(
        self,
        image_paths: List[str],
        prompts: List[Dict[str, str]],
        output_dir: str,
        base_seed: Optional[int] = None,
    ) -> List[str]:
        """Generate videos for all scenes."""
        os.makedirs(output_dir, exist_ok=True)
        video_paths = []

        for i, (img_path, prompt_data) in enumerate(zip(image_paths, prompts)):
            scene_num = prompt_data.get('scene_number', i + 1)
            output_path = os.path.join(output_dir, f"scene_{scene_num:02d}.mp4")

            if os.path.exists(output_path):
                logger.info(f"Video scene {scene_num} exists, skipping")
                video_paths.append(output_path)
                continue

            seed = (base_seed + scene_num * 100) if base_seed else None

            try:
                duration = prompt_data.get('duration', 5)
                fps = self.target_fps
                num_frames = _wan_num_frames(duration, fps)

                video_prompt = prompt_data.get('video_prompt') or prompt_data['prompt']
                video_neg = prompt_data.get('video_negative_prompt')

                path = self.generate(
                    image_path=img_path,
                    prompt=video_prompt,
                    output_path=output_path,
                    num_frames=num_frames,
                    fps=fps,
                    seed=seed,
                    negative_prompt=video_neg,
                )
                video_paths.append(path)
            except Exception as e:
                logger.error(f"Failed video scene {scene_num}: {e}")
                raise

        return video_paths

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    @staticmethod
    def verify_video(path: str) -> bool:
        """Verify generated video is valid."""
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return False
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return frame_count > 5  # At least 5 frames
        except Exception as e:
            logger.error(f"Video verification failed for {path}: {e}")
            return False
