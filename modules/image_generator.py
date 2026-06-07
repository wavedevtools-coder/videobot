# modules/image_generator.py
"""Image generation using FLUX.1 Dev via Diffusers.

Changes v2.1.0:
  - Character reference image (dinocharacter.png) loaded via IP-Adapter when
    available, so every generated scene has consistent Tiny Dino appearance.
  - Falls back gracefully to prompt-only generation when IP-Adapter weights
    are not installed (cloud environments without ip_adapter package).
  
Usage with FLUX.1-dev:
  1. Get a Hugging Face token from https://huggingface.co/settings/tokens
  2. Accept the model license at https://huggingface.co/black-forest-labs/FLUX.1-dev
  3. Set HF_TOKEN environment variable OR run: huggingface-cli login
  4. Run the pipeline
  
Alternative models (also require authentication):
  - black-forest-labs/FLUX.1-schnell (faster, fewer steps)
  - black-forest-labs/FLUX.1-pro (highest quality, requires Pro subscription)
"""

import logging
import os
import torch
from pathlib import Path
from typing import Optional, List, Dict, Any
from PIL import Image

from .config_loader import Config
from .character_manager import CharacterManager

logger = logging.getLogger('image_generator')


class FLUXImageGenerator:
    """FLUX.1 Dev/Schnell image generation with character reference consistency.
    
    Supports both FLUX.1-dev and FLUX.1-schnell models from Black Forest Labs.
    Requires Hugging Face authentication for access to gated models.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.img_config = self.config.models.get('image', {})
        self.model_id = self.img_config.get('model', 'black-forest-labs/FLUX.1-dev')
        dtype_name = self.img_config.get('dtype', 'bfloat16')
        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(dtype_name, torch.bfloat16)

        self.guidance_scale = self.img_config.get('guidance_scale', 3.5)
        self.num_steps = self.img_config.get('num_inference_steps', 28)
        self.max_seq_length = self.img_config.get('max_sequence_length', 512)
        self.enable_cpu_offload = self.img_config.get('enable_cpu_offload', True)

        # Character reference image (dinocharacter.png)
        assets = self.config.section('assets')
        self.char_ref_path = assets.get('character_ref', 'assets/dinocharacter.png')
        self._char_ref_image: Optional[Image.Image] = None
        self._ip_scale = self.img_config.get('ip_adapter_scale', 0.6)

        self._pipe = None
        self._loaded = False
        self._ip_adapter_ready = False

    # ------------------------------------------------------------------
    # Character reference loader
    # ------------------------------------------------------------------
    def _load_char_ref(self) -> Optional[Image.Image]:
        """Load dinocharacter.png as a PIL image (cached after first load)."""
        if self._char_ref_image is not None:
            return self._char_ref_image
        if os.path.exists(self.char_ref_path):
            try:
                img = Image.open(self.char_ref_path).convert("RGB")
                self._char_ref_image = img
                logger.info(f"Character reference image loaded: {self.char_ref_path}")
            except Exception as e:
                logger.warning(f"Could not load character ref image: {e}")
        else:
            logger.warning(
                f"Character ref image not found at {self.char_ref_path}. "
                "Generating without visual reference (prompt-only)."
            )
        return self._char_ref_image

    # ------------------------------------------------------------------
    # Pipeline loader
    # ------------------------------------------------------------------
    def _load_pipeline(self):
        """Lazy-load FLUX pipeline + optional IP-Adapter for char consistency."""
        if self._loaded:
            return

        try:
            from diffusers import FluxPipeline

            logger.info(f"Loading FLUX.1 Dev model: {self.model_id}")
            
            # Check if use_auth_token is enabled and HF token is available
            use_token = self.img_config.get('use_auth_token', False)
            load_kwargs = {
                'torch_dtype': self.dtype,
            }
            
            if use_token:
                import os
                hf_token = os.getenv('HF_TOKEN')
                if hf_token:
                    load_kwargs['use_auth_token'] = hf_token
                    logger.info("Using Hugging Face authentication token")
                else:
                    logger.warning(
                        "use_auth_token is enabled but HF_TOKEN environment variable not set. "
                        "If the model is gated, you need to either:\n"
                        "  1. Set HF_TOKEN environment variable with your Hugging Face token\n"
                        "  2. Run 'huggingface-cli login' to authenticate\n"
                        "  3. Use an open alternative like 'black-forest-labs/FLUX.1-schnell'"
                    )

            self._pipe = FluxPipeline.from_pretrained(
                self.model_id,
                **load_kwargs
            )

            if self.enable_cpu_offload:
                self._pipe.enable_model_cpu_offload()
                logger.info("Enabled model CPU offload")
            else:
                self._pipe = self._pipe.to("cuda")

            self._pipe.vae.enable_slicing()
            self._pipe.vae.enable_tiling()

            # ── Try to load IP-Adapter for character consistency ──────
            char_ref = self._load_char_ref()
            if char_ref is not None:
                try:
                    self._pipe.load_ip_adapter(
                        "h94/IP-Adapter",
                        subfolder="models",
                        weight_name="ip-adapter_sd15.bin",
                    )
                    self._pipe.set_ip_adapter_scale(self._ip_scale)
                    self._ip_adapter_ready = True
                    logger.info(
                        f"IP-Adapter loaded (scale={self._ip_scale}). "
                        "Tiny Dino reference image will guide every scene."
                    )
                except Exception as e:
                    logger.warning(
                        f"IP-Adapter not available ({e}). "
                        "Falling back to prompt-only character description. "
                        "Install ip_adapter for stronger visual consistency."
                    )

            self._loaded = True
            logger.info("FLUX.1 Dev pipeline ready")

        except ImportError:
            raise RuntimeError(
                "diffusers not installed. Install with:\n"
                "  pip install diffusers transformers accelerate"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load FLUX pipeline: {e}")

    # ------------------------------------------------------------------
    # Prompt enrichment with character description fallback
    # ------------------------------------------------------------------
    def _enrich_prompt_with_char(self, prompt: str) -> str:
        """
        When IP-Adapter is NOT available, prepend a hard-coded visual
        description of Tiny Dino so at least the text prompt drives
        character consistency.
        """
        if self._ip_adapter_ready:
            return prompt   # IP-Adapter handles visual reference

        char_desc = (
            "Tiny Dino character: small cute baby dinosaur, bright lime green "
            "skin (#7ED957), yellow-cream belly, large expressive brown eyes, "
            "orange spiky back ridges, smooth rounded body, chibi cartoon style, "
            "3D Pixar-quality render, consistent character design. "
        )
        return char_desc + prompt

    # ------------------------------------------------------------------
    # Core generate()
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        output_path: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Generate a single scene image with Tiny Dino character reference."""
        self._load_pipeline()

        width  = width  or self.config.video.get('width',  720)
        height = height or self.config.video.get('height', 1280)

        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)

        # Enrich prompt with character description if no IP-Adapter
        enriched_prompt = self._enrich_prompt_with_char(prompt)
        logger.info(f"Generating image [{width}x{height}]: {enriched_prompt[:100]}...")

        try:
            # Build kwargs — add ip_adapter_image when available
            kwargs: Dict[str, Any] = dict(
                prompt=enriched_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_steps,
                max_sequence_length=self.max_seq_length,
                generator=generator,
            )

            char_ref = self._load_char_ref()
            if self._ip_adapter_ready and char_ref is not None:
                kwargs['ip_adapter_image'] = char_ref

            result = self._pipe(**kwargs)
            image  = result.images[0]

            if output_path:
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                image.save(output_path, quality=95)
                logger.info(f"Image saved: {output_path}")
                return output_path

            return image

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA OOM! Clearing cache and retrying with CPU offload...")
            torch.cuda.empty_cache()
            if not self.enable_cpu_offload:
                self._pipe.enable_model_cpu_offload()
            raise

    # ------------------------------------------------------------------
    # Batch scene generation
    # ------------------------------------------------------------------
    def generate_scenes(
        self,
        prompts: List[Dict[str, str]],
        output_dir: str,
        base_seed: Optional[int] = None,
    ) -> List[str]:
        """Generate images for all scenes in a story."""
        os.makedirs(output_dir, exist_ok=True)
        generated_paths = []

        for i, prompt_data in enumerate(prompts):
            scene_num = prompt_data.get('scene_number', i + 1)
            output_path = os.path.join(output_dir, f"scene_{scene_num:02d}.png")

            if os.path.exists(output_path):
                logger.info(f"Scene {scene_num} already exists, skipping")
                generated_paths.append(output_path)
                continue

            seed = (base_seed + scene_num) if base_seed else None

            try:
                path = self.generate(
                    prompt=prompt_data['prompt'],
                    negative_prompt=prompt_data.get('negative_prompt', ''),
                    output_path=output_path,
                    seed=seed,
                )
                generated_paths.append(path)
            except Exception as e:
                logger.error(f"Failed to generate scene {scene_num}: {e}")
                raise

        return generated_paths

    @staticmethod
    def verify_image(path: str) -> bool:
        """Verify generated image is valid and not corrupted."""
        try:
            img = Image.open(path)
            img.verify()
            if img.size[0] < 100 or img.size[1] < 100:
                return False
            return True
        except Exception as e:
            logger.error(f"Image verification failed for {path}: {e}")
            return False