# modules/story_generator.py
"""Story generation using Qwen3 14B via Ollama."""

import json
import logging
import os
import random
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime

import requests
from .config_loader import Config
from .character_manager import CharacterManager
from .story_scorer import StoryScorer, StoryScore

logger = logging.getLogger('story_generator')


class AntiRepeatTracker:
    """Tracks story history to prevent repetition."""

    def __init__(self, history_file: str, similarity_threshold: float = 0.75, max_history: int = 1000):
        self.history_file = history_file
        self.similarity_threshold = similarity_threshold
        self.max_history = max_history
        self._history: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        """Load story history from file."""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r', encoding='utf-8') as f:
                self._history = json.load(f)
        logger.info(f"Loaded {len(self._history)} stories in history")

    def _save(self):
        """Save story history to file."""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        # Keep only recent history
        trimmed = self._history[-self.max_history:]
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(trimmed, f, indent=2, ensure_ascii=False)

    def compute_fingerprint(self, story: Dict[str, Any]) -> str:
        """Create fingerprint for similarity comparison."""
        text = f"{story.get('title', '')} " + " ".join(
            s.get('description', '') for s in story.get('scenes', [])
        )
        return hashlib.md5(text.lower().encode()).hexdigest()

    def check_similarity(self, story: Dict[str, Any]) -> tuple[bool, float]:
        """Check if story is too similar to any in history."""
        new_fp = self.compute_fingerprint(story)
        new_text = f"{story.get('title', '')} " + " ".join(
            s.get('description', '') for s in story.get('scenes', [])
        ).lower()

        for entry in self._history:
            if entry.get('fingerprint') == new_fp:
                return False, 1.0  # Exact match

            # Word overlap similarity
            old_text = entry.get('text_preview', '')
            if old_text:
                new_words = set(new_text.split())
                old_words = set(old_text.split())
                if new_words and old_words:
                    jaccard = len(new_words & old_words) / len(new_words | old_words)
                    if jaccard > self.similarity_threshold:
                        return False, jaccard

        return True, 0.0

    def add(self, story: Dict[str, Any]):
        """Add story to history."""
        entry = {
            'fingerprint': self.compute_fingerprint(story),
            'title': story.get('title', ''),
            'text_preview': f"{story.get('title', '')} " + " ".join(
                s.get('description', '') for s in story.get('scenes', [])
            )[:200],
            'timestamp': datetime.now().isoformat(),
        }
        self._history.append(entry)
        self._save()

    def get_used_themes(self) -> set:
        """Get set of recently used themes/topics."""
        themes = set()
        for entry in self._history[-50:]:
            themes.add(entry.get('title', '').lower().split()[0] if entry.get('title') else '')
        return themes


class StoryGenerator:
    """Generates Tiny Dino stories using local LLM via Ollama."""

    # Story formula: Hook → Try → Fail (funny) → Try → Win → Lesson
    STORY_PROMPT_TEMPLATE = """You are an expert children's story writer specializing in 30-second animated shorts.
Create a funny, heartwarming Tiny Dino story following this exact structure:

CHARACTER: {character_description}
PERSONALITY: {personality}

STORY FORMULA (MANDATORY - exactly 6 scenes):
Scene 1 - HOOK: Dino discovers something or faces a silly problem. Must start with immediate curiosity or surprise.
Scene 2 - FIRST TRY: Dino attempts to solve it in a naive/enthusiastic way. Setup for comedy.
Scene 3 - FUNNY FAIL: The attempt backfires in a visually hilarious, slapstick way. This is the comedy peak.
Scene 4 - SECOND TRY: Dino tries a clever/different approach, learns from mistake.
Scene 5 - FUNNY WIN: Success happens in an unexpected, adorable way. Visual payoff.
Scene 6 - HEARTWARMING ENDING: A tiny life lesson or sweet moment. Dino happy/satisfied.

RULES:
- Each scene MUST be visually descriptive for AI image generation
- Include camera directions (close-up, wide shot, etc.)
- Focus on physical comedy and exaggerated reactions
- Tiny Dino never speaks human language - only squeaks, roars, body language
- Setting: colorful prehistoric/fantasy world
- Mood progression: curious → excited → chaos → determined → triumph → warm
- Avoid: scary content, complex dialogue, dark themes, repetitive gags
- Target: ages 3-8, universally funny

{anti_repeat_prompt}

OUTPUT FORMAT - JSON only:
{{
  "title": "Catchy 3-5 word title",
  "theme": "one-word theme tag",
  "scenes": [
    {{
      "scene_number": 1,
      "type": "hook",
      "description": "Detailed visual description for image generation...",
      "mood": "curious",
      "camera": "close-up/medium/wide",
      "duration_seconds": 5
    }},
    ...
  ]
}}

Generate ONE story now. Return ONLY valid JSON, no markdown, no explanation."""

    ANTI_REPEAT_PROMPT = """
AVOID these recently used themes: {used_themes}
Create something COMPLETELY DIFFERENT. Use an unexpected combination of:
- Setting: {setting}
- Object: {object}
- Situation: {situation}
"""

    SETTINGS = [
        'crystal cave', 'bubble jungle', 'candy volcano', 'cloud island',
        'glowing mushroom forest', 'rainbow waterfall', 'floating rock garden',
        'starlight beach', 'butterfly meadow', 'giant flower field',
        'honey tree village', 'snowy hot spring', 'musical bamboo forest',
        'whispering dunes', 'coral tree park',
    ]

    OBJECTS = [
        'a magical seed', 'a squeaky toy', 'a flying pancake',
        'a shrinking berry', 'a dancing shadow', 'a talking cloud',
        'a slippery slime', 'a glowing pebble', 'a grumpy chicken',
        'a giant spoon', 'a bubble wand', 'a wobbly jelly',
        'a sleeping butterfly', 'a mysterious egg', 'a ticklish feather',
        'a runaway snowball', 'a friendly ghost', 'a bouncing mushroom',
    ]

    SITUATIONS = [
        'must deliver before sunset', 'accidentally broke something precious',
        'made a promise to a friend', 'discovered something that should not exist',
        'got challenged to a silly contest', 'found a map to invisible treasure',
        'has to hide from a parade', 'must learn a skill in one hour',
        'became unexpectedly tiny/giant', 'switched places with a chicken',
    ]

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.char_manager = CharacterManager(self.config)
        self.scorer = StoryScorer(self.config.quality.get('min_story_score', 70))
        self.llm_config = self.config.models.get('llm', {})
        self.base_url = self.llm_config.get('base_url', 'http://localhost:11434')
        self.model = self.llm_config.get('model', 'qwen3:14b')
        self.temperature = self.llm_config.get('temperature', 0.8)
        self.max_tokens = self.llm_config.get('max_tokens', 2048)

        # Anti-repeat
        story_cfg = self.config.story
        self.tracker = AntiRepeatTracker(
            history_file=story_cfg.get('history_file', 'data/story_history.json'),
            similarity_threshold=story_cfg.get('similarity_threshold', 0.75),
            max_history=story_cfg.get('history_limit', 1000),
        )

        self._generation_count = 0
        self._max_regenerations = self.config.quality.get('max_regenerations', 3)

    def _build_prompt(self) -> str:
        """Build the story generation prompt."""
        profile = self.char_manager.load_profile()
        appearance = self.char_manager.get_appearance_description()
        personality = self.char_manager.get_personality_traits()

        # Anti-repeat elements
        used_themes = self.tracker.get_used_themes()
        setting = random.choice(self.SETTINGS)
        object_item = random.choice(self.OBJECTS)
        situation = random.choice(self.SITUATIONS)

        anti_repeat = ""
        if self.config.story.get('anti_repeat', True) and used_themes:
            anti_repeat = self.ANTI_REPEAT_PROMPT.format(
                used_themes=', '.join(list(used_themes)[:10]),
                setting=setting,
                object=object_item,
                situation=situation,
            )

        return self.STORY_PROMPT_TEMPLATE.format(
            character_description=appearance,
            personality=personality,
            anti_repeat_prompt=anti_repeat,
        )

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API for story generation."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '')
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Ensure Ollama is running: ollama run {self.model}"
            )
        except requests.exceptions.Timeout:
            raise RuntimeError("Ollama request timed out after 120s")
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")

    def _parse_story(self, raw_text: str) -> Dict[str, Any]:
        """Parse LLM output into structured story."""
        # Extract JSON from response
        text = raw_text.strip()

        # Remove markdown code blocks if present
        if text.startswith('```'):
            lines = text.split('\n')
            # Find JSON content between code fences
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith('```json') or line.startswith('```'):
                    in_json = not in_json
                    continue
                if in_json or (not line.startswith('```') and '{' in line):
                    json_lines.append(line)
            text = '\n'.join(json_lines)

        # Try to find JSON object
        try:
            # Direct parse
            story = json.loads(text)
        except json.JSONDecodeError:
            # Extract JSON from text
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                try:
                    story = json.loads(text[start:end])
                except json.JSONDecodeError as e:
                    raise ValueError(f"Cannot parse story JSON: {e}\nRaw: {text[:500]}")
            else:
                raise ValueError(f"No JSON found in response:\n{text[:500]}")

        # Validate structure
        if 'title' not in story or 'scenes' not in story:
            raise ValueError(f"Story missing required fields: {list(story.keys())}")

        if len(story['scenes']) < 4:
            raise ValueError(f"Story has only {len(story['scenes'])} scenes, need at least 4")

        # Normalize scenes
        for i, scene in enumerate(story['scenes']):
            scene['scene_number'] = i + 1
            if 'mood' not in scene:
                moods = ['curious', 'excited', 'chaos', 'determined', 'triumph', 'warm']
                scene['mood'] = moods[min(i, len(moods)-1)]
            if 'camera' not in scene:
                scene['camera'] = 'medium shot'
            if 'duration_seconds' not in scene:
                scene['duration_seconds'] = self.config.video.get('scene_duration', 5)

        return story

    def generate(self, max_retries: Optional[int] = None) -> Dict[str, Any]:
        """Generate a high-quality story with scoring and validation."""
        max_retries = max_retries or self._max_regenerations

        for attempt in range(1, max_retries + 1):
            logger.info(f"Story generation attempt {attempt}/{max_retries}")

            try:
                # Step 1: Generate raw story
                prompt = self._build_prompt()
                raw = self._call_ollama(prompt)

                # Step 2: Parse
                story = self._parse_story(raw)
                logger.info(f"Generated story: '{story.get('title', 'Untitled')}'")

                # Step 3: Quick check
                story_text = json.dumps(story)
                passed, reason = self.scorer.quick_check(story_text)
                if not passed:
                    logger.warning(f"Quick check failed: {reason}")
                    continue

                # Step 4: Full scoring
                score = self.scorer.score(story)
                logger.info(f"Story score: {score.total}/100")

                # Step 5: Anti-repeat check
                if self.config.story.get('anti_repeat', True):
                    is_unique, sim = self.tracker.check_similarity(story)
                    if not is_unique:
                        logger.warning(f"Story too similar to history (similarity: {sim:.2f})")
                        continue

                # Step 6: Validation
                if score.passed:
                    # Add to history
                    self.tracker.add(story)
                    story['_score'] = {
                        'total': score.total,
                        'hook': score.hook_strength,
                        'comedy': score.comedy_potential,
                        'visual': score.visual_clarity,
                        'unique': score.uniqueness,
                        'ending': score.ending_quality,
                    }
                    logger.info(f"Story ACCEPTED with score {score.total}")
                    return story
                else:
                    logger.warning(f"Story REJECTED (score {score.total} < {self.scorer.min_score})")
                    logger.debug(self.scorer.format_score_report(score))

            except Exception as e:
                logger.error(f"Generation attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    raise

        raise RuntimeError(f"Failed to generate acceptable story after {max_retries} attempts")
