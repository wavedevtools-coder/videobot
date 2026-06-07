# modules/video_generator.py
"""Video generation using LTX-Video-2-3."""

import logging
import os
import torch
from typing import Optional, List, Dict, Any
from pathlib import Path

from .config_loader import Config

logger = logging.getLogger('video_generator')

# LTX requires num_frames = 8n + 1 (e.g. 121, 145, 153)
DEFAULT_VIDEO_NEGATIVE = (
    "fade to black, darkening, black frames, static freeze, flickering, "
    "glitch, blur, inconsistent motion, morphing, distortion, jittery, low quality"
)


def _ltx_num_frames(duration_seconds: float, fps: int) -> int:
    """Snap frame count to LTX requirement: 8n + 1."""
    target = max(9, int(duration_seconds * fps))
    n = max(1, (target - 1 + 7) // 8)
    return n * 8 + 1


class LTXVideoGenerator:
    """LTX-2.3 image-to-video generation with RTX 5090 optimizations."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.vid_config = self.config.models.get('video', {})
        self.model_id = self.vid_config.get('model', 'Lightricks/LTX-Video-2-3')
        self.dtype = getattr(torch, self.vid_config.get('dtype', 'bfloat16'))
        self.num_steps = self.vid_config.get('num_inference_steps', 30)
        self.enable_vae_slicing = self.vid_config.get('enable_vae_slicing', True)

        self._pipe = None
        self._loaded = False

    def _load_pipeline(self):
        """Lazy-load LTX pipeline with memory optimizations."""
        if self._loaded:
            return

        try:
            from diffusers import LTXImageToVideoPipeline

            logger.info(f"Loading LTX-Video-2-3 model: {self.model_id}")

            self._pipe = LTXImageToVideoPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
            )
            self._pipe.enable_model_cpu_offload()

            if self.enable_vae_slicing:
                self._pipe.vae.enable_slicing()
                logger.info("Enabled VAE slicing for memory savings")

            # Additional memory optimization (call on VAE sub-component)
            if hasattr(self._pipe.vae, 'enable_tiling'):
                self._pipe.vae.enable_tiling()
                logger.info("Enabled VAE tiling for memory savings")

            self._loaded = True
            logger.info("LTX-2.3 pipeline loaded successfully")

        except ImportError:
            raise RuntimeError(
                "diffusers not installed. Install with: "
                "pip install diffusers transformers accelerate"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load LTX pipeline: {e}")

    def generate(
        self,
        image_path: str,
        prompt: str,
        output_path: str,
        num_frames: int = 121,
        fps: int = 24,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
    ) -> str:
        """Generate video from image using LTX."""
        self._load_pipeline()

        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)

        neg = negative_prompt or DEFAULT_VIDEO_NEGATIVE

        logger.info(f"Generating video from: {image_path} ({num_frames} frames @ {fps}fps)")
        torch.cuda.empty_cache()

        try:
            from PIL import Image

            image = Image.open(image_path).convert("RGB")

            orig_w, orig_h = image.size

            gen_w, gen_h = 576, 1024
            if image.size != (gen_w, gen_h):
                logger.info(f"Resizing image {orig_w}x{orig_h} → {gen_w}x{gen_h} for LTX generation")
                image = image.resize((gen_w, gen_h), Image.LANCZOS)

            pipe_kwargs = dict(
                image=image,
                prompt=prompt,
                negative_prompt=neg,
                num_inference_steps=self.num_steps,
                num_frames=num_frames,
                width=gen_w,
                height=gen_h,
                generator=generator,
            )

            result = self._pipe(**pipe_kwargs)

            video = result.frames[0]

            if (gen_w, gen_h) != (orig_w, orig_h):
                logger.info(f"Upscaling generated frames back to {orig_w}x{orig_h}")
                video = [frame.resize((orig_w, orig_h), Image.LANCZOS) for frame in video]

            self._save_video(video, output_path, fps)
            logger.info(f"Video saved: {output_path} ({len(video)} frames)")
            return output_path

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA OOM during video generation!")
            torch.cuda.empty_cache()
            raise

    def _save_video(self, frames: list, output_path: str, fps: int):
        """Save frames as MP4 using imageio with yuv420p for broad player compatibility."""
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

        h, w = frames[0].shape[:2] if hasattr(frames[0], 'shape') else (1280, 720)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for frame in frames:
            # Convert PIL Image to numpy array if needed
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
                fps = self.config.video.get('fps', 30)
                num_frames = _ltx_num_frames(duration, fps)

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
