# modules/image_generator.py
"""Image generation via Diffusers — auto-selects pipeline class from model ID.

Changes v2.1.2 (fix):
  - Replaced hardcoded FluxPipeline with AutoPipelineForText2Image so the
    correct pipeline class is chosen automatically from the model ID.
    This fixes the crash when config switches between FLUX, SDXL-Turbo,
    SDXL, SD1.5, etc. without requiring code changes.
  - Removed ignore_missing_keys kwarg — not accepted by diffusers pipelines;
    was silently ignored and gave a noisy warning.
  - FLUX-specific handling preserved: negative_prompt omitted for FLUX models,
    IP-Adapter skipped (XLabs weights optional), prompt enriched with char desc.
  - SDXL-Turbo handling: guidance_scale forced to 0.0, num_steps clamped to
    config value (default 4), negative_prompt supported.
  - Character reference image (dinocharacter.png) enriches prompts via
    IP-Adapter when available, falls back to text-only description.
"""

import logging
import os
import torch
from typing import Optional, List, Dict, Any
from PIL import Image

from .config_loader import Config
from .character_manager import CharacterManager

logger = logging.getLogger('image_generator')

# Models that don't support negative_prompt
_FLUX_MODEL_PREFIXES = (
    "black-forest-labs/flux",
    "flux.1",
    "flux-",
)

def _is_flux_model(model_id: str) -> bool:
    low = model_id.lower()
    return any(low.startswith(p) or p in low for p in _FLUX_MODEL_PREFIXES)


class FLUXImageGenerator:
    """Diffusers image generation — auto-selects pipeline from model ID.

    Works with FLUX.1-dev, FLUX.1-schnell, SDXL-Turbo, SDXL, SD1.5, etc.
    The model is chosen entirely from config.yaml models.image.model.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config      = config or Config()
        self.img_config  = self.config.models.get('image', {})
        self.model_id    = self.img_config.get('model', 'black-forest-labs/FLUX.1-dev')

        dtype_name = self.img_config.get('dtype', 'bfloat16')
        dtype_map = {
            "bf16":     torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16":     torch.float16,
            "float16":  torch.float16,
            "fp32":     torch.float32,
            "float32":  torch.float32,
        }
        self.dtype = dtype_map.get(dtype_name, torch.bfloat16)

        self.guidance_scale     = self.img_config.get('guidance_scale', 3.5)
        self.num_steps          = self.img_config.get('num_inference_steps', 28)
        self.max_seq_length     = self.img_config.get('max_sequence_length', 512)
        self.enable_cpu_offload = self.img_config.get('enable_cpu_offload', True)

        # Character reference image
        assets = self.config.section('assets')
        self.char_ref_path          = assets.get('character_ref', 'assets/dinocharacter.png')
        self._char_ref_image: Optional[Image.Image] = None
        self._ip_scale              = self.img_config.get('ip_adapter_scale', 0.6)

        self._pipe              = None
        self._loaded            = False
        self._ip_adapter_ready  = False
        self._is_flux           = _is_flux_model(self.model_id)

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
    # Pipeline loader — uses AutoPipelineForText2Image
    # ------------------------------------------------------------------
    def _load_pipeline(self):
        """Lazy-load pipeline using AutoPipelineForText2Image.

        AutoPipeline inspects the model's config.json and instantiates the
        correct class (FluxPipeline, StableDiffusionXLPipeline, etc.)
        automatically — no hardcoding needed.
        """
        if self._loaded:
            return

        try:
            from diffusers import AutoPipelineForText2Image

            logger.info(f"Loading model via AutoPipeline: {self.model_id}")
            self._pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
            )

            if self.enable_cpu_offload:
                self._pipe.enable_model_cpu_offload()
                logger.info("Enabled model CPU offload")
            else:
                self._pipe = self._pipe.to("cuda")

            # VAE optimisations (not all pipelines expose these — guard them)
            if hasattr(self._pipe, 'vae') and self._pipe.vae is not None:
                try:
                    self._pipe.vae.enable_slicing()
                    self._pipe.vae.enable_tiling()
                except Exception:
                    pass

            # ── IP-Adapter (FLUX-native only; skipped for other models) ──
            if self._is_flux:
                char_ref = self._load_char_ref()
                if char_ref is not None:
                    try:
                        self._pipe.load_ip_adapter(
                            "XLabs-AI/flux-ip-adapter",
                            weight_name="ip_adapter.safetensors",
                        )
                        self._pipe.set_ip_adapter_scale(self._ip_scale)
                        self._ip_adapter_ready = True
                        logger.info(
                            f"FLUX IP-Adapter loaded (scale={self._ip_scale}). "
                            "Tiny Dino reference image will guide every scene."
                        )
                    except Exception as e:
                        logger.warning(
                            f"FLUX IP-Adapter not available ({e}). "
                            "Falling back to prompt-only character description. "
                            "To enable: download XLabs-AI/flux-ip-adapter weights."
                        )

            self._loaded = True
            logger.info(
                f"Pipeline ready: {type(self._pipe).__name__} "
                f"| model={self.model_id} | flux={self._is_flux}"
            )

        except ImportError:
            raise RuntimeError(
                "diffusers not installed. Run:\n"
                "  pip install diffusers transformers accelerate"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load image pipeline: {e}")

    # ------------------------------------------------------------------
    # Prompt enrichment
    # ------------------------------------------------------------------
    def _enrich_prompt_with_char(self, prompt: str) -> str:
        """Prepend Tiny Dino visual description when IP-Adapter is unavailable."""
        if self._ip_adapter_ready:
            return prompt

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
        """Generate a single scene image.

        Builds pipeline kwargs appropriate for the loaded model:
        - FLUX models: no negative_prompt, uses max_sequence_length
        - SDXL-Turbo: guidance_scale=0.0 (required by the model)
        - All others: standard kwargs
        """
        self._load_pipeline()

        width  = width  or self.config.video.get('width',  720)
        height = height or self.config.video.get('height', 1280)

        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)

        enriched_prompt = self._enrich_prompt_with_char(prompt)
        logger.info(f"Generating image [{width}x{height}] with {type(self._pipe).__name__}: "
                    f"{enriched_prompt[:80]}...")

        try:
            kwargs: Dict[str, Any] = dict(
                prompt=enriched_prompt,
                width=width,
                height=height,
                num_inference_steps=self.num_steps,
                generator=generator,
            )

            model_lower = self.model_id.lower()

            if self._is_flux:
                # FLUX does not support negative_prompt
                kwargs['guidance_scale']     = self.guidance_scale
                kwargs['max_sequence_length'] = self.max_seq_length
                char_ref = self._load_char_ref()
                if self._ip_adapter_ready and char_ref is not None:
                    kwargs['ip_adapter_image'] = char_ref

            elif "turbo" in model_lower:
                # SDXL-Turbo requires guidance_scale=0.0
                kwargs['guidance_scale']  = 0.0
                if negative_prompt:
                    kwargs['negative_prompt'] = negative_prompt

            else:
                # Standard SD / SDXL
                kwargs['guidance_scale'] = self.guidance_scale
                if negative_prompt:
                    kwargs['negative_prompt'] = negative_prompt

            result = self._pipe(**kwargs)
            image  = result.images[0]

            if output_path:
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                image.save(output_path, quality=95)
                logger.info(f"Image saved: {output_path}")
                return output_path

            return image

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA OOM — clearing cache and enabling CPU offload...")
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
            scene_num   = prompt_data.get('scene_number', i + 1)
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
            return img.size[0] >= 100 and img.size[1] >= 100
        except Exception as e:
            logger.error(f"Image verification failed for {path}: {e}")
            return False
