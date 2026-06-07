# modules/audio_generator.py
"""Audio generation using Stable Audio Open for sound effects and music."""

import logging
import os
import torch
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

from .config_loader import Config

logger = logging.getLogger('audio_generator')


class StableAudioGenerator:
    """Stable Audio Open for generating sound effects and background music."""

    # Mood-to-audio prompt mapping
    MOOD_PROMPTS = {
        'curious': 'gentle mysterious music box, light percussion, soft strings, wonder, discovery, magical atmosphere',
        'excited': 'upbeat bouncy marimba, fast tempo, energetic drums, happy playful melody, children adventure',
        'chaos': 'slapstick comedy music, trombone slides, xylophone runs, boing sounds, cartoon chase music',
        'determined': 'heroic light orchestra, building drums, optimistic brass, adventure theme, rising intensity',
        'triumph': 'celebratory fanfare, bright bells, joyful strings, victory theme, sparkling sounds',
        'warm': 'soft acoustic guitar, gentle piano, warm strings, cozy lullaby, peaceful bedtime',
        'scared': 'tense pizzicato, low rumbling, mysterious woodwinds, slight dissonance, spooky but not terrifying',
        'silly': 'funny kazoo, bouncy bassoon, comedic percussion, cartoon sound effects, playful waltz',
        'happy': 'bright glockenspiel, cheerful flute, light percussion, sunny melody, children playing',
        'sad': 'gentle solo piano, soft cello, melancholic but hopeful, warm tones',
        'surprised': 'orchestral hit, sudden cymbal, dramatic pause, comedic sting, cartoon surprise',
    }

    # Sound effect prompts for specific actions
    SFX_PROMPTS = {
        'jump': 'cartoon bounce boing, spring sound, playful hop',
        'fall': 'comedic fall whistle, soft thud, cartoon crash',
        'splash': 'water splash, playful splashing, bubble sounds',
        'run': 'quick footsteps, patter sounds, cartoon running',
        'roar': 'cute baby dinosaur roar, tiny growl, adorable rumble',
        'chirp': 'happy bird-like chirp, cute squeak, friendly tweet',
        'eat': 'munching sounds, happy eating, yum sounds',
        'sleep': 'gentle snoring, soft breathing, cozy night sounds',
        'discovery': 'magical chime, sparkly sound, aha moment',
        'celebration': 'party horn, confetti sound, happy cheer',
    }

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.audio_config = self.config.models.get('audio', {})
        self.model_id = self.audio_config.get('model', 'stabilityai/stable-audio-open-1.0')
        self.dtype = getattr(torch, self.audio_config.get('dtype', 'float16'))
        self.audio_length = self.audio_config.get('audio_length', 8.0)
        self.num_steps = self.audio_config.get('num_inference_steps', 20)
        self.cfg_scale = self.audio_config.get('cfg_scale', 7.0)

        self._model = None
        self._processor = None
        self._loaded = False

    def _load_model(self):
        """Lazy-load Stable Audio Open model."""
        if self._loaded:
            return

        try:
            from diffusers import StableAudioPipeline

            logger.info(f"Loading Stable Audio Open: {self.model_id}")

            self._model = StableAudioPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
            )
            self._model = self._model.to("cuda")

            self._loaded = True
            logger.info("Stable Audio Open loaded successfully")

        except ImportError:
            raise RuntimeError(
                "transformers not installed. Install with: "
                "pip install transformers torchaudio"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Stable Audio: {e}")

    def generate(
        self,
        prompt: str,
        output_path: str,
        duration: Optional[float] = None,
        negative_prompt: str = "low quality, distorted, noisy, harsh, loud",
        seed: Optional[int] = None,
    ) -> str:
        """Generate audio clip from text prompt."""
        self._load_model()

        duration = duration or self.audio_length
        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)

        logger.info(f"Generating audio: {prompt[:80]}...")

        try:
            audio = self._model(
                prompt,
                negative_prompt=negative_prompt,
                audio_end_in_s=duration,
                num_inference_steps=self.num_steps,
                guidance_scale=self.cfg_scale,
                generator=generator,
            ).audios[0]

            # Save as WAV
            self._save_audio(audio, output_path)
            logger.info(f"Audio saved: {output_path}")
            return output_path

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA OOM during audio generation!")
            torch.cuda.empty_cache()
            raise

    def _save_audio(self, audio_tensor, output_path: str):
        """Save audio tensor as WAV file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            import torchaudio
            # Ensure tensor is on CPU and detached
            if hasattr(audio_tensor, 'detach'):
                audio_tensor = audio_tensor.detach().cpu()
            else:
                audio_tensor = torch.tensor(audio_tensor).cpu()

            # Normalize to [-1, 1]
            if audio_tensor.abs().max() > 1:
                audio_tensor = audio_tensor / audio_tensor.abs().max()

            torchaudio.save(output_path, audio_tensor.unsqueeze(0), sample_rate=44100)
        except ImportError:
            # Fallback to scipy
            from scipy.io import wavfile
            audio_np = np.array(audio_tensor)
            if audio_np.abs().max() > 1:
                audio_np = audio_np / audio_np.abs().max()
            wavfile.write(output_path, 44100, (audio_np * 32767).astype(np.int16))

    def generate_for_scene(
        self,
        mood: str,
        action: Optional[str] = None,
        output_path: str = "",
        duration: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Generate scene-appropriate audio."""
        # Build prompt from mood and action
        prompt_parts = []

        if action and action in self.SFX_PROMPTS:
            prompt_parts.append(self.SFX_PROMPTS[action])

        mood_prompt = self.MOOD_PROMPTS.get(mood, self.MOOD_PROMPTS['happy'])
        prompt_parts.append(mood_prompt)

        prompt = ", ".join(prompt_parts)

        return self.generate(
            prompt=prompt,
            output_path=output_path,
            duration=duration,
            seed=seed,
        )

    def generate_background_music(
        self,
        story_duration: float,
        output_path: str,
        seed: Optional[int] = None,
    ) -> str:
        """Generate continuous background music for full story."""
        # Create a consistent theme
        base_prompt = (
            "gentle adventurous children's music, "
            "marimba and flute melody, light percussion, "
            "happy playful atmosphere, seamless loop friendly, "
            "no sudden changes, consistent tempo"
        )

        # For long stories, generate in segments and concatenate
        max_segment = 10.0  # Max generation length
        if story_duration <= max_segment:
            return self.generate(
                prompt=base_prompt,
                output_path=output_path,
                duration=story_duration,
                seed=seed,
            )

        # Generate segments
        import tempfile
        segments = []
        remaining = story_duration
        segment_num = 0

        while remaining > 0:
            seg_duration = min(max_segment, remaining)
            seg_path = tempfile.mktemp(suffix='.wav')
            seg_seed = (seed + segment_num) if seed else None

            self.generate(
                prompt=base_prompt,
                output_path=seg_path,
                duration=seg_duration,
                seed=seg_seed,
            )
            segments.append(seg_path)
            remaining -= seg_duration
            segment_num += 1

        # Concatenate segments
        self._concatenate_audio(segments, output_path)

        # Cleanup temp files
        for seg in segments:
            os.remove(seg)

        return output_path

    def _concatenate_audio(self, segment_paths: list, output_path: str):
        """Concatenate audio segments with crossfade."""
        try:
            import torchaudio
            import torch

            all_audio = []
            for path in segment_paths:
                waveform, sr = torchaudio.load(path)
                all_audio.append(waveform)

            # Simple concatenation (crossfade can be added)
            combined = torch.cat(all_audio, dim=-1)
            torchaudio.save(output_path, combined, sr)
        except ImportError:
            # Fallback: just copy first segment
            import shutil
            shutil.copy(segment_paths[0], output_path)

    def generate_sfx_mix(
        self,
        scenes: list,
        output_dir: str,
        base_seed: Optional[int] = None,
    ) -> list:
        """Generate individual SFX clips for each scene."""
        os.makedirs(output_dir, exist_ok=True)
        sfx_paths = []

        for i, scene in enumerate(scenes):
            scene_num = scene.get('scene_number', i + 1)
            mood = scene.get('mood', 'happy')
            action = scene.get('action', '')

            output_path = os.path.join(output_dir, f"sfx_{scene_num:02d}.wav")

            if os.path.exists(output_path):
                sfx_paths.append(output_path)
                continue

            seed = (base_seed + scene_num * 50) if base_seed else None
            duration = scene.get('duration_seconds', 5)

            try:
                path = self.generate_for_scene(
                    mood=mood,
                    action=action,
                    output_path=output_path,
                    duration=duration,
                    seed=seed,
                )
                sfx_paths.append(path)
            except Exception as e:
                logger.error(f"SFX generation failed for scene {scene_num}: {e}")
                sfx_paths.append(None)

        return sfx_paths
