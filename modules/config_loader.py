# modules/config_loader.py
"""Configuration management for Tiny Dino AI Shorts Factory."""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Singleton configuration manager."""

    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load(config_path or cls._find_config())
        return cls._instance

    @staticmethod
    def _find_config() -> str:
        """Find config.yaml in project root or current directory."""
        search_paths = [
            "config.yaml",
            "../config.yaml",
            "../../config.yaml",
            str(Path.home() / "dino" / "config.yaml"),
            "D:/dino/config.yaml",
        ]
        for path in search_paths:
            if os.path.exists(path):
                return path
        raise FileNotFoundError("config.yaml not found. Please create it in project root.")

    def _load(self, path: str):
        """Load YAML configuration."""
        with open(path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        self._validate()
        self._resolve_paths()

    def _validate(self):
        """Validate required configuration keys."""
        required = ['video', 'generation', 'quality', 'budget', 'models']
        for key in required:
            if key not in self._config:
                raise ValueError(f"Missing required config section: {key}")

    def _resolve_paths(self):
        """Resolve relative paths to absolute paths."""
        root = self._config.get('project_root', '.')
        os.makedirs(root, exist_ok=True)
        for key in ['assets', 'logging', 'story']:
            if key in self._config:
                for subkey in ['file', 'history_file', 'outro_video', 'outro_metadata', 'logo', 'font', 'watermark']:
                    if subkey in self._config[key]:
                        path = self._config[key][subkey]
                        if not os.path.isabs(path):
                            self._config[key][subkey] = os.path.join(root, path)

    def get(self, *keys: str, default=None) -> Any:
        """Get nested config value: config.get('video', 'width')."""
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def section(self, name: str) -> Dict[str, Any]:
        """Get a config section as dictionary."""
        return self._config.get(name, {})

    @property
    def video(self) -> Dict[str, Any]:
        return self.section('video')

    @property
    def generation(self) -> Dict[str, Any]:
        return self.section('generation')

    @property
    def quality(self) -> Dict[str, Any]:
        return self.section('quality')

    @property
    def budget(self) -> Dict[str, Any]:
        return self.section('budget')

    @property
    def upload(self) -> Dict[str, Any]:
        return self.section('upload')

    @property
    def models(self) -> Dict[str, Any]:
        return self.section('models')

    @property
    def story(self) -> Dict[str, Any]:
        return self.section('story')

    def as_dict(self) -> Dict[str, Any]:
        """Return full configuration as dictionary."""
        return self._config.copy()
