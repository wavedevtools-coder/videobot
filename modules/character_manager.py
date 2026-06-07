# modules/character_manager.py
"""Character profile management for Tiny Dino consistency."""

import json
import os
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from .config_loader import Config


class CharacterManager:
    """Manages Tiny Dino character profile for consistent scene generation."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.profile_path = self.config.get('assets', 'outro_metadata', default='assets/character_profile.json')
        # Try multiple locations for character profile
        possible_paths = [
            'assets/character_profile.json',
            '../assets/character_profile.json',
            os.path.join(self.config.get('project_root') or '.', 'assets', 'character_profile.json'),
        ]
        self.profile_path = next((p for p in possible_paths if os.path.exists(p)), self.profile_path)
        self._profile: Optional[Dict[str, Any]] = None
        self._version_hash: Optional[str] = None

    def load_profile(self) -> Dict[str, Any]:
        """Load character profile from JSON."""
        if self._profile is not None:
            return self._profile

        if not os.path.exists(self.profile_path):
            raise FileNotFoundError(f"Character profile not found: {self.profile_path}")

        with open(self.profile_path, 'r', encoding='utf-8') as f:
            self._profile = json.load(f)

        self._version_hash = self._compute_hash(self._profile)
        return self._profile

    @staticmethod
    def _compute_hash(profile: Dict[str, Any]) -> str:
        """Compute hash of profile for cache invalidation."""
        content = json.dumps(profile, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:8]

    @property
    def version(self) -> str:
        """Get character version string."""
        profile = self.load_profile()
        return profile.get('version', '1.0')

    @property
    def version_hash(self) -> str:
        """Get computed version hash."""
        if self._version_hash is None:
            self.load_profile()
        return self._version_hash

    def get_base_description(self, style: str = 'default') -> str:
        """Get Tiny Dino base description for image generation prompts."""
        profile = self.load_profile()
        appearance = profile['appearance']
        presets = profile.get('prompt_presets', {})

        base = presets.get('base_description', '')
        style_mod = presets.get('style_modifiers', {}).get(style, presets.get('style_modifiers', {}).get('default', ''))

        return f"{base}, {style_mod}".strip(', ')

    def get_appearance_description(self) -> str:
        """Get detailed appearance description."""
        profile = self.load_profile()
        app = profile['appearance']
        parts = [
            f"{app['size']}",
            f"{app['color']['primary']} colored {app['species']}",
            f"{app['belly']['color']} belly",
            f"{app['eyes']['description']}",
            f"{app['skin_texture']} skin texture",
            ", ".join(app.get('features', []))
        ]
        return ", ".join(parts)

    def get_mood_modifier(self, mood: str) -> str:
        """Get mood-specific prompt modifier."""
        profile = self.load_profile()
        presets = profile.get('prompt_presets', {})
        mood_mods = presets.get('mood_modifiers', {})
        return mood_mods.get(mood, mood_mods.get('happy', ''))

    def get_personality_traits(self) -> str:
        """Get personality traits as comma-separated string."""
        profile = self.load_profile()
        traits = profile.get('personality', {}).get('traits', [])
        return ", ".join(traits)

    def enrich_scene_prompt(self, scene_description: str, mood: str = 'happy', style: str = 'default') -> str:
        """Enrich a scene description with Tiny Dino character details."""
        base = self.get_base_description(style)
        mood_mod = self.get_mood_modifier(mood)
        appearance = self.get_appearance_description()

        enriched = (
            f"{base}. {appearance}. {mood_mod}. "
            f"Scene: {scene_description}. "
            f"Vertical 9:16 format, high quality, detailed, cinematic lighting."
        )
        return enriched

    def get_outro_metadata(self) -> Dict[str, str]:
        """Get metadata for outro cache validation."""
        profile = self.load_profile()
        return {
            "character_version": profile.get('version', '1.0'),
            "character_hash": self.version_hash,
            "logo_version": "1.0",
            "resolution": f"{self.config.video.get('width', 720)}x{self.config.video.get('height', 1280)}",
        }

    def validate_profile(self) -> bool:
        """Validate character profile has all required fields."""
        profile = self.load_profile()
        required = ['name', 'version', 'appearance', 'personality', 'prompt_presets']
        for key in required:
            if key not in profile:
                return False
        appearance_required = ['species', 'color', 'eyes', 'skin_texture']
        for key in appearance_required:
            if key not in profile.get('appearance', {}):
                return False
        return True
