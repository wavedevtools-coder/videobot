# modules/scene_generator.py
"""Scene prompt engineering for image generation with character consistency."""

import json
import logging
from typing import Dict, Any, List, Optional
from .config_loader import Config
from .character_manager import CharacterManager

logger = logging.getLogger('scene_generator')


class ScenePromptBuilder:
    """Builds optimized prompts for each scene ensuring character consistency."""

    # Negative prompts to avoid common AI issues
    DEFAULT_NEGATIVE = (
        "ugly, deformed, blurry, low quality, distorted, "
        "disfigured, poorly drawn face, mutation, mutated, "
        "extra limbs, extra fingers, malformed limbs, "
        "missing arms, missing legs, extra arms, extra legs, "
        "fused fingers, too many fingers, long neck, "
        "cross-eyed, mutated hands, polar lowres, bad face, "
        "out of frame, oversaturated, overexposed"
    )

    # Style presets for different moods
    STYLE_PRESETS = {
        'curious': 'soft lighting, warm tones, shallow depth of field, magical atmosphere',
        'excited': 'bright vibrant colors, dynamic lighting, energetic composition',
        'chaos': 'motion blur, exaggerated angles, comedic timing visual, slapstick energy',
        'determined': 'dramatic side lighting, heroic angle, focused composition',
        'triumph': 'golden hour lighting, celebratory particles, epic framing',
        'warm': 'soft golden glow, cozy atmosphere, gentle bokeh, heartwarming',
        'scared': 'dramatic shadows, wide angle, tense atmosphere',
        'silly': 'bright flat lighting, cartoon proportions, playful colors',
        'sleepy': 'soft blue hour lighting, gentle atmosphere, dreamy bokeh',
    }

    CAMERA_SHOTS = {
        'close-up': 'extreme close-up on face, detailed expression, shallow depth of field',
        'medium': 'medium shot, character in environment, balanced framing',
        'wide': 'wide establishing shot, full environment, epic scale',
        'overhead': 'top-down view, bird eye perspective, full scene layout',
        'low': 'low angle shot, looking up at character, heroic perspective',
        'dynamic': 'dutch angle, dynamic composition, action lines',
        'insert': 'insert shot, focus on specific object or detail',
    }

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.char_manager = CharacterManager(self.config)
        self.video_config = self.config.video

    def build_image_prompt(self, scene: Dict[str, Any]) -> Dict[str, str]:
        """Build a complete image generation prompt for a scene."""
        mood = scene.get('mood', 'happy')
        camera = scene.get('camera', 'medium shot')
        description = scene.get('description', '')
        scene_type = scene.get('type', '')

        # Get character-consistent description
        char_enriched = self.char_manager.enrich_scene_prompt(
            scene_description=description,
            mood=mood,
            style='default'
        )

        # Add style preset for mood
        style = self.STYLE_PRESETS.get(mood, self.STYLE_PRESETS['curious'])

        # Add camera direction
        cam = self.CAMERA_SHOTS.get(camera.lower().split()[0], self.CAMERA_SHOTS['medium'])

        # Combine
        prompt = (
            f"{char_enriched}, {style}, {cam}, "
            f"3D Pixar-style animation, soft lighting, vibrant colors, "
            f"highly detailed, cinematic composition, "
            f"vertical 9:16 format, {self.video_config.get('width', 720)}x{self.video_config.get('height', 1280)}, "
            f"professional quality, render"
        )

        return {
            'prompt': prompt,
            'negative_prompt': self.DEFAULT_NEGATIVE,
            'mood': mood,
            'camera': camera,
            'scene_type': scene_type,
        }

    def build_all_prompts(self, story: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build prompts for all scenes in a story."""
        prompts = []
        scenes = story.get('scenes', [])

        for i, scene in enumerate(scenes):
            prompt_data = self.build_image_prompt(scene)
            prompt_data['scene_number'] = i + 1
            prompt_data['duration'] = scene.get('duration_seconds', self.video_config.get('scene_duration', 5))
            prompts.append(prompt_data)
            logger.debug(f"Scene {i+1} prompt built: {prompt_data['prompt'][:100]}...")

        return prompts

    def estimate_generation_cost(self, num_scenes: int) -> Dict[str, float]:
        """Estimate GPU cost for scene generation."""
        # Rough estimates for RTX 3090 24GB
        per_scene = {
            'image_generation_minutes': 0.5,  # FLUX.1 dev ~30s per image
            'video_generation_minutes': 2.0,  # LTX ~2min per 5s clip
            'audio_generation_minutes': 0.3,  # Stable Audio ~20s per clip
        }

        total = {k: v * num_scenes for k, v in per_scene.items()}
        total['total_gpu_minutes'] = sum(per_scene.values()) * num_scenes
        # Rough INR estimate (cloud GPU pricing)
        total['estimated_cost_inr'] = total['total_gpu_minutes'] * 0.5  # ~₹0.5 per GPU min

        return total

    def create_storyboard_plan(self, story: Dict[str, Any]) -> Dict[str, Any]:
        """Create a full storyboard plan with all prompts and metadata."""
        scenes = story.get('scenes', [])
        prompts = self.build_all_prompts(story)
        cost_estimate = self.estimate_generation_cost(len(scenes))

        storyboard = {
            'title': story.get('title', 'Untitled'),
            'theme': story.get('theme', 'general'),
            'total_scenes': len(scenes),
            'total_duration_seconds': sum(s.get('duration_seconds', 5) for s in scenes) + self.video_config.get('outro_duration', 4),
            'scenes': [],
            'cost_estimate': cost_estimate,
            'character_version': self.char_manager.version,
        }

        for i, (scene, prompt) in enumerate(zip(scenes, prompts)):
            storyboard['scenes'].append({
                'scene_number': i + 1,
                'type': scene.get('type', ''),
                'mood': scene.get('mood', ''),
                'duration_seconds': scene.get('duration_seconds', 5),
                'description': scene.get('description', ''),
                'prompt': prompt['prompt'],
                'negative_prompt': prompt['negative_prompt'],
                'camera': prompt['camera'],
            })

        return storyboard
