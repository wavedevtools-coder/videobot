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

    DEFAULT_NEGATIVE = (
        "ugly, deformed, blurry, low quality, distorted, disfigured, "
        "poorly drawn face, mutation, extra limbs, extra fingers, "
        "malformed limbs, missing arms, missing legs, fused fingers, "
        "cross-eyed, text, watermark, logo, empty black background, "
        "dark underexposed, cropped character, out of frame, oversaturated"
    )

    VIDEO_NEGATIVE = (
        "fade to black, darkening, black frames, static freeze, flickering, "
        "glitch, blur, inconsistent motion, morphing, distortion, jittery, "
        "low quality, sudden scene change, empty void, washed out"
    )

    STYLE_PRESETS = {
        'curious': 'soft warm lighting, shallow depth of field, magical glow, lush colorful background',
        'excited': 'bright vibrant colors, dynamic rim lighting, energetic composition, saturated palette',
        'chaos': 'exaggerated cartoon angles, motion lines, slapstick energy, comedic timing',
        'determined': 'dramatic side lighting, heroic low angle, focused composition, vivid contrast',
        'triumph': 'golden hour lighting, sparkling particles, celebratory atmosphere, epic framing',
        'warm': 'soft golden glow, cozy atmosphere, gentle bokeh, heartwarming tones',
        'scared': 'dramatic but friendly shadows, wide angle, tense but cute atmosphere',
        'silly': 'bright flat cartoon lighting, playful colors, exaggerated proportions',
        'happy': 'cheerful sunny lighting, vivid greens and blues, inviting environment',
        'sleepy': 'soft blue hour lighting, dreamy bokeh, peaceful atmosphere',
    }

    CAMERA_SHOTS = {
        'close-up': 'extreme close-up on Tiny Dino face, expressive eyes, shallow depth of field',
        'medium': 'medium shot, Tiny Dino full body in colorful environment, balanced framing',
        'wide': 'wide establishing shot, Tiny Dino small in vivid prehistoric landscape',
        'overhead': 'top-down view, full scene layout, playful perspective',
        'low': 'low angle hero shot, looking up at Tiny Dino, empowering framing',
        'dynamic': 'dynamic dutch angle, action composition, sense of movement',
        'insert': 'insert shot, focus on key prop or detail Tiny Dino interacts with',
    }

    MOTION_BY_TYPE = {
        'hook': 'Tiny Dino notices something, head tilts with curiosity, eyes widen, subtle lean forward',
        'first_try': 'Tiny Dino bounces eagerly toward goal, arms flail, tail wags, hopeful energy',
        'funny_fail': 'Tiny Dino stumbles in slapstick motion, exaggerated bounce, comedic reaction',
        'fail': 'Tiny Dino stumbles in slapstick motion, exaggerated bounce, comedic reaction',
        'second_try': 'Tiny Dino tries again with focused determination, careful deliberate movement',
        'funny_win': 'Tiny Dino celebrates with joyful bounce, tail wagging, happy chirping body language',
        'win': 'Tiny Dino celebrates with joyful bounce, tail wagging, happy body language',
        'ending': 'Tiny Dino settles contentedly, gentle breathing, soft satisfied expression',
        'heartwarming': 'Tiny Dino relaxes happily, slow blink, cozy peaceful micro-movements',
    }

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.char_manager = CharacterManager(self.config)
        self.video_config = self.config.video

    def build_image_prompt(self, scene: Dict[str, Any]) -> Dict[str, str]:
        """Build a still-frame prompt optimized for SDXL / FLUX image generation."""
        mood = scene.get('mood', 'happy')
        camera = scene.get('camera', 'medium shot')
        description = scene.get('description', '')
        scene_type = scene.get('type', '')

        char_enriched = self.char_manager.enrich_scene_prompt(
            scene_description=description,
            mood=mood,
            style='default',
        )

        style = self.STYLE_PRESETS.get(mood, self.STYLE_PRESETS['curious'])
        cam_key = camera.lower().split()[0] if camera else 'medium'
        cam = self.CAMERA_SHOTS.get(cam_key, self.CAMERA_SHOTS['medium'])

        prompt = (
            f"{char_enriched}. "
            f"Single frozen keyframe moment, {cam}. {style}. "
            f"Tiny Dino clearly visible and centered, full body in frame, "
            f"rich detailed background fills the scene, bright well-lit, no empty darkness. "
            f"3D Pixar-style cartoon render, soft subsurface skin, vibrant colors, "
            f"cinematic composition, vertical 9:16 mobile format, sharp focus, 8K detail"
        )

        return {
            'prompt': prompt,
            'negative_prompt': self.DEFAULT_NEGATIVE,
            'mood': mood,
            'camera': camera,
            'scene_type': scene_type,
        }

    def build_video_prompt(self, scene: Dict[str, Any]) -> Dict[str, str]:
        """Build a motion-focused prompt for LTX image-to-video."""
        mood = scene.get('mood', 'happy')
        description = scene.get('description', '')
        scene_type = scene.get('type', '').lower()
        motion = scene.get('motion', '').strip()

        if not motion:
            motion = self.MOTION_BY_TYPE.get(
                scene_type,
                'Tiny Dino moves naturally with smooth subtle animation, expressive body language',
            )

        style = self.STYLE_PRESETS.get(mood, self.STYLE_PRESETS['curious'])

        prompt = (
            f"Smooth cinematic animation of Tiny Dino, an adorable vibrant lime green 3D Pixar-style baby dinosaur. "
            f"He has a smooth yellow belly running up to his chin, huge round eyes with bright green irises, and a wide friendly smile. "
            f"Scene: {description}. "
            f"Motion: {motion}. "
            f"{style}. "
            f"Consistent character appearance throughout, fluid natural movement, "
            f"stable camera, well-lit colorful scene stays visible, "
            f"3D Pixar cartoon style, vertical 9:16, masterpiece, ultra-high resolution, extremely detailed, sharp focus, 8K, highly clear, no fade to black"
        )

        return {
            'prompt': prompt,
            'negative_prompt': self.VIDEO_NEGATIVE,
        }

    def build_all_prompts(self, story: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build image and video prompts for all scenes in a story."""
        prompts = []
        scenes = story.get('scenes', [])

        for i, scene in enumerate(scenes):
            image_data = self.build_image_prompt(scene)
            video_data = self.build_video_prompt(scene)

            prompt_data = {
                **image_data,
                'video_prompt': video_data['prompt'],
                'video_negative_prompt': video_data['negative_prompt'],
                'scene_number': i + 1,
                'duration': scene.get(
                    'duration_seconds',
                    self.video_config.get('scene_duration', 5),
                ),
            }
            prompts.append(prompt_data)
            logger.debug(
                f"Scene {i+1} image: {prompt_data['prompt'][:90]}... | "
                f"video: {prompt_data['video_prompt'][:90]}..."
            )

        return prompts

    def estimate_generation_cost(self, num_scenes: int) -> Dict[str, float]:
        """Estimate GPU cost for scene generation."""
        per_scene = {
            'image_generation_minutes': 0.5,
            'video_generation_minutes': 2.0,
            'audio_generation_minutes': 0.3,
        }

        total = {k: v * num_scenes for k, v in per_scene.items()}
        total['total_gpu_minutes'] = sum(per_scene.values()) * num_scenes
        total['estimated_cost_inr'] = total['total_gpu_minutes'] * 0.5

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
            'total_duration_seconds': sum(s.get('duration_seconds', 5) for s in scenes)
                + self.video_config.get('outro_duration', 4),
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
                'motion': scene.get('motion', ''),
                'prompt': prompt['prompt'],
                'video_prompt': prompt['video_prompt'],
                'negative_prompt': prompt['negative_prompt'],
                'video_negative_prompt': prompt['video_negative_prompt'],
                'camera': prompt['camera'],
            })

        return storyboard
