# modules/video_generator.py
"""Video generation using LTX-Video-2-3."""

import logging
import os
import torch
from typing import Optional, List, Dict, Any
from pathlib import Path

from .config_loader import Config

logger = logging.getLogger('video_generator')


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
            self._pipe = self._pipe.to("cuda")

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
        num_frames: int = 121,  # ~5 seconds at 24fps
        fps: int = 24,
        seed: Optional[int] = None,
    ) -> str:
        """Generate video from image using LTX."""
        self._load_pipeline()

        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)

        logger.info(f"Generating video from: {image_path}")

        try:
            from PIL import Image

            image = Image.open(image_path).convert("RGB")

            width, height = image.size  # PIL gives (w, h)
            result = self._pipe(
                image=image,
                prompt=prompt,
                num_inference_steps=self.num_steps,
                num_frames=num_frames,
                width=width,
                height=height,
                generator=generator,
            )

            video = result.frames[0]  # List of PIL Images

            # Save as MP4
            self._save_video(video, output_path, fps)
            logger.info(f"Video saved: {output_path}")
            return output_path

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA OOM during video generation!")
            torch.cuda.empty_cache()
            raise

    def _save_video(self, frames: list, output_path: str, fps: int):
        """Save frames as MP4 using imageio."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            import imageio
            writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
            for frame in frames:
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
            if hasattr(frame, 'numpy'):
                arr = frame.numpy()
            else:
                arr = frame
            # Convert RGB to BGR for OpenCV
            if arr.shape[-1] == 3:
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
                # Calculate frames for scene duration
                duration = prompt_data.get('duration', 5)
                fps = self.config.video.get('fps', 30)
                num_frames = int(duration * fps)

                path = self.generate(
                    image_path=img_path,
                    prompt=prompt_data['prompt'],
                    output_path=output_path,
                    num_frames=num_frames,
                    fps=fps,
                    seed=seed,
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
