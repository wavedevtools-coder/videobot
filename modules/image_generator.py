# modules/image_generator.py
"""Image generation via Diffusers — auto-selects pipeline class from model ID.

Changes v2.1.3:
  - Config switched to stabilityai/sdxl-turbo (Apache 2.0, no HF auth needed,
    commercial-safe for YouTube monetization, 4-step fast generation).
  - Added hf_token config key: set token for gated models (FLUX.1-schnell/dev)
    or leave blank for open models (sdxl-turbo, sdxl-base).
  - local_files_only fast-path: if model is already cached locally, load it
    without any network call (works fully offline after first download).
  - Clear actionable error when model needs auth: tells user exactly what to do.
  - AutoPipelineForText2Image auto-selects correct class (no hardcoded FluxPipeline).
  - Model-specific inference kwargs: turbo forces guidance_scale=0.0,
    FLUX omits negative_prompt, all others use standard kwargs.
"""

import logging
import os
import torch
from typing import Optional, List, Dict, Any
from PIL import Image

from .config_loader import Config

logger = logging.getLogger('image_generator')

_FLUX_PREFIXES = ("flux",)
_TURBO_KEYWORDS = ("turbo",)

def _is_flux(model_id: str) -> bool:
    return any(p in model_id.lower() for p in _FLUX_PREFIXES)

def _is_turbo(model_id: str) -> bool:
    return any(k in model_id.lower() for k in _TURBO_KEYWORDS)


class FLUXImageGenerator:
    """Diffusers image generation — model-agnostic via AutoPipelineForText2Image.

    Set config.yaml  models.image.model  to any diffusers-compatible model ID.
    No code changes needed when switching models.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config      = config or Config()
        self.img_config  = self.config.models.get('image', {})
        self.model_id    = self.img_config.get('model', 'stabilityai/sdxl-turbo')

        dtype_name = self.img_config.get('dtype', 'bfloat16')
        self.dtype = {
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp16": torch.float16,  "float16":  torch.float16,
            "fp32": torch.float32,  "float32":  torch.float32,
        }.get(dtype_name, torch.bfloat16)

        self.guidance_scale     = self.img_config.get('guidance_scale', 0.0)
        self.num_steps          = self.img_config.get('num_inference_steps', 4)
        self.max_seq_length     = self.img_config.get('max_sequence_length', 512)
        self.enable_cpu_offload = self.img_config.get('enable_cpu_offload', True)

        # HuggingFace token — needed for gated models (FLUX.1-schnell/dev)
        # Set in config.yaml models.image.hf_token or HF_TOKEN env var
        self.hf_token = (
            self.img_config.get('hf_token', '')
            or os.environ.get('HF_TOKEN', '')
            or os.environ.get('HUGGING_FACE_HUB_TOKEN', '')
        ).strip() or None

        assets = self.config.section('assets')
        self.char_ref_path = assets.get('character_ref', 'assets/dinocharacter.png')
        self._char_ref_image: Optional[Image.Image] = None
        self._ip_scale = self.img_config.get('ip_adapter_scale', 0.6)

        self._pipe             = None
        self._loaded           = False
        self._ip_adapter_ready = False

    # ------------------------------------------------------------------
    # Character reference
    # ------------------------------------------------------------------
    def _load_char_ref(self) -> Optional[Image.Image]:
        if self._char_ref_image is not None:
            return self._char_ref_image
        if os.path.exists(self.char_ref_path):
            try:
                self._char_ref_image = Image.open(self.char_ref_path).convert("RGB")
                logger.info(f"Character reference loaded: {self.char_ref_path}")
            except Exception as e:
                logger.warning(f"Could not load character ref: {e}")
        else:
            logger.warning(f"Character ref not found at {self.char_ref_path} — using prompt-only.")
        return self._char_ref_image

    # ------------------------------------------------------------------
    # Pipeline loader
    # ------------------------------------------------------------------
    def _load_pipeline(self):
        """Load pipeline using AutoPipelineForText2Image.

        Strategy:
          1. Try local_files_only=True first (instant, no network, works offline)
          2. Fall back to download with token if provided
          3. Fall back to download without token (open models)
          4. Raise a clear actionable error if still failing
        """
        if self._loaded:
            return

        try:
            from diffusers import AutoPipelineForText2Image

            load_kwargs = dict(torch_dtype=self.dtype)
            if self.hf_token:
                load_kwargs['token'] = self.hf_token

            # ── 1. Try local cache first (fast path, no network) ──────────
            pipe = None
            try:
                logger.info(f"Checking local cache for: {self.model_id}")
                pipe = AutoPipelineForText2Image.from_pretrained(
                    self.model_id,
                    local_files_only=True,
                    **load_kwargs,
                )
                logger.info("Loaded from local cache (offline mode).")
            except Exception:
                pass  # Not cached — will download below

            # ── 2. Download if not cached ──────────────────────────────────
            if pipe is None:
                logger.info(f"Downloading model: {self.model_id}")
                try:
                    pipe = AutoPipelineForText2Image.from_pretrained(
                        self.model_id,
                        **load_kwargs,
                    )
                except Exception as e:
                    err = str(e)
                    if "401" in err or "Unauthorized" in err or "gated" in err.lower():
                        raise RuntimeError(
                            f"Model '{self.model_id}' requires a HuggingFace token.\n\n"
                            "Fix options:\n"
                            "  A) Use a free model — set in config.yaml:\n"
                            "       models.image.model: stabilityai/sdxl-turbo\n"
                            "  B) Add your HF token to config.yaml:\n"
                            "       models.image.hf_token: hf_xxxxxxxxxxxx\n"
                            "  C) Run once in terminal: huggingface-cli login\n\n"
                            f"Original error: {e}"
                        )
                    raise

            self._pipe = pipe

            # ── Memory optimisations ───────────────────────────────────────
            if self.enable_cpu_offload:
                self._pipe.enable_model_cpu_offload()
                logger.info("CPU offload enabled (lower VRAM usage).")
            else:
                self._pipe = self._pipe.to("cuda")

            if hasattr(self._pipe, 'vae') and self._pipe.vae is not None:
                try:
                    self._pipe.vae.enable_slicing()
                    self._pipe.vae.enable_tiling()
                except Exception:
                    pass

            # ── FLUX IP-Adapter (optional, FLUX models only) ───────────────
            if _is_flux(self.model_id):
                char_ref = self._load_char_ref()
                if char_ref is not None:
                    try:
                        self._pipe.load_ip_adapter(
                            "XLabs-AI/flux-ip-adapter",
                            weight_name="ip_adapter.safetensors",
                        )
                        self._pipe.set_ip_adapter_scale(self._ip_scale)
                        self._ip_adapter_ready = True
                        logger.info(f"FLUX IP-Adapter loaded (scale={self._ip_scale}).")
                    except Exception as e:
                        logger.warning(
                            f"FLUX IP-Adapter not available ({e}). "
                            "Using prompt-only character description."
                        )

            self._loaded = True
            logger.info(
                f"Pipeline ready: {type(self._pipe).__name__} | "
                f"model={self.model_id} | steps={self.num_steps}"
            )

        except ImportError:
            raise RuntimeError(
                "diffusers not installed. Run:\n"
                "  pip install diffusers transformers accelerate"
            )
        except RuntimeError:
            raise   # re-raise our own clear errors
        except Exception as e:
            raise RuntimeError(f"Failed to load image pipeline: {e}")

    # ------------------------------------------------------------------
    # Prompt enrichment
    # ------------------------------------------------------------------
    def _enrich_prompt(self, prompt: str) -> str:
        """Prepend Tiny Dino visual description when IP-Adapter is unavailable."""
        if self._ip_adapter_ready:
            return prompt
        return (
            "Tiny Dino: small baby T-rex, bright lime-green body, light yellow belly, "
            "large expressive eyes, cute chibi cartoon style, 3D Pixar render, "
            "centered in frame, bright well-lit scene. "
            + prompt
        )

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
        """Generate one scene image. Kwargs adapt to the loaded model type."""
        self._load_pipeline()

        width  = width  or self.config.video.get('width',  720)
        height = height or self.config.video.get('height', 1280)

        generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None
        enriched  = self._enrich_prompt(prompt)

        logger.info(
            f"Generating [{width}x{height}] {type(self._pipe).__name__}: "
            f"{enriched[:80]}..."
        )

        try:
            kwargs: Dict[str, Any] = dict(
                prompt=enriched,
                width=width,
                height=height,
                num_inference_steps=self.num_steps,
                generator=generator,
            )

            if _is_flux(self.model_id):
                # FLUX: no negative_prompt; needs max_sequence_length
                kwargs['guidance_scale']      = self.guidance_scale
                kwargs['max_sequence_length'] = self.max_seq_length
                char_ref = self._load_char_ref()
                if self._ip_adapter_ready and char_ref is not None:
                    kwargs['ip_adapter_image'] = char_ref

            elif _is_turbo(self.model_id):
                # SDXL-Turbo: guidance_scale MUST be 0.0
                kwargs['guidance_scale'] = 0.0
                if negative_prompt:
                    kwargs['negative_prompt'] = negative_prompt

            else:
                # Standard SDXL / SD1.5 etc.
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
            logger.error("CUDA OOM — clearing cache, enabling CPU offload...")
            torch.cuda.empty_cache()
            if not self.enable_cpu_offload:
                self._pipe.enable_model_cpu_offload()
            raise

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------
    def generate_scenes(
        self,
        prompts: List[Dict[str, str]],
        output_dir: str,
        base_seed: Optional[int] = None,
    ) -> List[str]:
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        for i, pd in enumerate(prompts):
            scene_num   = pd.get('scene_number', i + 1)
            output_path = os.path.join(output_dir, f"scene_{scene_num:02d}.png")
            if os.path.exists(output_path):
                logger.info(f"Scene {scene_num} cached, skipping")
                paths.append(output_path)
                continue
            seed = (base_seed + scene_num) if base_seed else None
            try:
                paths.append(self.generate(
                    prompt=pd['prompt'],
                    negative_prompt=pd.get('negative_prompt', ''),
                    output_path=output_path,
                    seed=seed,
                ))
            except Exception as e:
                logger.error(f"Scene {scene_num} failed: {e}")
                raise
        return paths

    @staticmethod
    def verify_image(path: str) -> bool:
        try:
            img = Image.open(path)
            img.verify()
            return img.size[0] >= 100 and img.size[1] >= 100
        except Exception as e:
            logger.error(f"Image verify failed {path}: {e}")
            return False
