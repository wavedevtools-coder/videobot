# modules/story_scorer.py
"""Story Quality Filter - reject weak stories before GPU spend."""

import re
import json
import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger('story_scorer')


@dataclass
class StoryScore:
    """Story quality score breakdown."""
    hook_strength: float = 0.0
    comedy_potential: float = 0.0
    visual_clarity: float = 0.0
    uniqueness: float = 0.0
    ending_quality: float = 0.0
    total: float = 0.0
    passed: bool = False
    feedback: List[str] = field(default_factory=list)


class StoryScorer:
    """Scores stories across 5 dimensions before GPU generation."""

    # Comedy keywords that indicate humor potential
    COMEDY_INDICATORS = [
        'slapstick', 'chase', 'sneez', 'trip', 'slip', 'wobble', 'bounce',
        'mishap', 'accident', 'funny', 'hilarious', 'silly', 'goofy', 'clumsy',
        'mistake', 'mess', 'confused', 'surprised', 'shock', 'oops', 'whoops',
        'pratfall', 'tangle', 'stuck', 'slippery', 'waddle', 'stumble',
        'chocolate', 'cake', 'ice cream', 'mud', 'water', 'puddle', 'balloon',
        'chicken', 'bee', 'spider', 'mouse', 'shadow', 'mirror', 'costume',
    ]

    # Visual scene indicators
    VISUAL_INDICATORS = [
        'close-up', 'wide shot', 'zoom', 'pan', 'slow motion', 'fast motion',
        'explosion', 'splash', 'chase', 'run', 'jump', 'fly', 'fall', 'roll',
        'spin', 'bounce', 'shatter', 'transform', 'glow', 'sparkle', 'rain',
        'sunset', 'forest', 'cave', 'mountain', 'river', 'storm', 'fireworks',
        'butterfly', 'balloon', 'snow', 'bubble', 'rainbow', 'waterfall',
    ]

    # Weak/poor story patterns
    WEAK_PATTERNS = [
        r'(and then.*and then.*and then)',  # repetitive structure
        r'(suddenly.*suddenly)',            # overuse of suddenly
        r'(very.*very.*very)',              # repetitive intensifiers
        r'\b(bad|sad|angry|mad|upset)\b.*\b(bad|sad|angry|mad|upset)\b',  # too negative
    ]

    def __init__(self, min_score: float = 70.0):
        self.min_score = min_score

    def score(self, story: Dict[str, Any]) -> StoryScore:
        """Score a story and return detailed breakdown."""
        title = story.get('title', '')
        scenes = story.get('scenes', [])
        story_text = ' '.join([s.get('description', '') for s in scenes])
        full_text = f"{title} {story_text}".lower()

        result = StoryScore()
        result.feedback = []

        # 1. Hook Strength (0-100)
        result.hook_strength = self._score_hook(story)

        # 2. Comedy Potential (0-100)
        result.comedy_potential = self._score_comedy(full_text, scenes)

        # 3. Visual Clarity (0-100)
        result.visual_clarity = self._score_visual(story_text, scenes)

        # 4. Uniqueness (0-100)
        result.uniqueness = self._score_uniqueness(full_text, story)

        # 5. Ending Quality (0-100)
        result.ending_quality = self._score_ending(scenes)

        # Weighted total
        result.total = round(
            result.hook_strength * 0.25 +
            result.comedy_potential * 0.25 +
            result.visual_clarity * 0.20 +
            result.uniqueness * 0.15 +
            result.ending_quality * 0.15,
            1
        )

        result.passed = result.total >= self.min_score

        if not result.passed:
            result.feedback.append(
                f"Story score {result.total} below minimum {self.min_score}"
            )

        return result

    def _score_hook(self, story: Dict[str, Any]) -> float:
        """Score opening hook strength (0-100)."""
        scenes = story.get('scenes', [])
        if not scenes:
            return 0.0

        first_scene = scenes[0].get('description', '').lower()
        score = 50.0  # base

        # Strong opening patterns
        strong_hooks = [
            r'^(one day|suddenly|out of nowhere|little did|when|just as)',
            r'(discover|found|notice|realize|wonder|curious|mysterious)',
            r'(strange|weird|odd|unusual|magical|glowing|shiny)',
            r'(?!.*(wake up|open eyes|start day))',  # avoid weak openings
        ]

        for pattern in strong_hooks:
            if re.search(pattern, first_scene):
                score += 15

        # Penalize boring openings
        boring = ['wake up', 'get out of bed', 'start morning', 'open eyes', 'yawn']
        for b in boring:
            if b in first_scene:
                score -= 25

        # Action in first scene bonus
        action_words = ['run', 'chase', 'jump', 'fall', 'crash', 'bounce', 'roll']
        for word in action_words:
            if word in first_scene:
                score += 5

        return max(0, min(100, score))

    def _score_comedy(self, text: str, scenes: List[Dict]) -> float:
        """Score comedy potential (0-100)."""
        score = 30.0  # base

        # Count comedy indicators
        indicator_count = sum(1 for ind in self.COMEDY_INDICATORS if ind in text)
        score += min(indicator_count * 8, 40)

        # Scene count variety (more scenes = more comedy beats)
        if len(scenes) >= 4:
            score += 10
        if len(scenes) >= 5:
            score += 5

        # Physical comedy potential
        physical = ['slip', 'trip', 'fall', 'bounce', 'roll', 'crash', 'spin']
        phys_count = sum(1 for p in physical if p in text)
        score += min(phys_count * 5, 15)

        # Reaction-based humor
        reactions = ['gasp', 'blink', 'jaw drop', 'stare', 'confused', 'surprised']
        react_count = sum(1 for r in reactions if r in text)
        score += min(react_count * 3, 10)

        return max(0, min(100, score))

    def _score_visual(self, text: str, scenes: List[Dict]) -> float:
        """Score visual clarity for AI generation (0-100)."""
        score = 40.0  # base

        # Visual indicator keywords
        vis_count = sum(1 for vis in self.VISUAL_INDICATORS if vis in text)
        score += min(vis_count * 3, 25)

        # Scene specificity check
        for scene in scenes:
            desc = scene.get('description', '').lower()
            # Check for setting
            if any(word in desc for word in ['forest', 'cave', 'beach', 'mountain', 'river', 'meadow']):
                score += 3
            # Check for clear subject
            if 'dino' in desc or 'tiny' in desc:
                score += 3
            # Check for action
            if any(word in desc for word in ['run', 'jump', 'chase', 'fall', 'bounce']):
                score += 2

        # Penalize abstract/internal descriptions
        abstract = ['think', 'feel', 'remember', 'wonder internally', 'imagine']
        for abs_word in abstract:
            if abs_word in text:
                score -= 8

        return max(0, min(100, score))

    def _score_uniqueness(self, text: str, story: Dict) -> float:
        """Score story uniqueness (0-100)."""
        score = 50.0  # base

        title = story.get('title', '').lower()

        # Check for common/generic titles
        generic = ['adventure', 'day', 'story', 'journey', 'trip']
        for g in generic:
            if title == g or title == f"the {g}" or title == f"a {g}":
                score -= 20

        # Specific elements boost uniqueness
        unique_elements = [
            'chicken', 'robot', 'alien', 'pirate', 'ninja', 'wizard',
            'cookie', 'pizza', 'ice cream', 'chocolate', 'spaghetti',
            'umbrella', 'kite', 'balloon', 'slide', 'swing', 'trampoline',
            'parrot', 'penguin', 'octopus', 'unicorn', 'dragon',
        ]
        unique_count = sum(1 for u in unique_elements if u in text)
        score += min(unique_count * 8, 30)

        # Unexpected combinations
        combos = [
            ('dino', 'dance'), ('dino', 'cook'), ('dino', 'paint'),
            ('dino', 'sing'), ('dino', 'skate'), ('dino', 'surf'),
        ]
        for a, b in combos:
            if a in text and b in text:
                score += 10

        return max(0, min(100, score))

    def _score_ending(self, scenes: List[Dict]) -> float:
        """Score ending quality (0-100)."""
        if not scenes:
            return 0.0

        last_scene = scenes[-1].get('description', '').lower()
        score = 50.0  # base

        # Strong ending patterns
        good_endings = [
            r'(laugh|giggle|smile|happy|joy|cheer|celebrate)',
            r'(hug|cuddle|nap|sleep|dream|cozy|warm)',
            r'(lesson|learn|realize|understand|friend)',
            r'(home|safe|together|family|love)',
        ]
        for pattern in good_endings:
            if re.search(pattern, last_scene):
                score += 12

        # Action endings are good
        action_end = ['jump', 'dance', 'spin', 'bounce', 'waddle']
        for a in action_end:
            if a in last_scene:
                score += 8

        # Weak endings
        weak = ['to be continued', 'the end abruptly', 'suddenly stop', 'just stand']
        for w in weak:
            if w in last_scene:
                score -= 20

        # Resolution check - does it feel complete?
        if len(scenes) >= 3:
            score += 10

        return max(0, min(100, score))

    def quick_check(self, story_text: str) -> Tuple[bool, str]:
        """Quick pass/fail check with reason."""
        # Check for minimum requirements
        if len(story_text) < 100:
            return False, "Story too short (< 100 chars)"

        if len(story_text) > 2000:
            return False, "Story too long (> 2000 chars)"

        # Check for weak patterns
        for pattern in self.WEAK_PATTERNS:
            if re.search(pattern, story_text, re.IGNORECASE):
                return False, f"Weak pattern detected: {pattern}"

        # Require Tiny Dino presence
        if 'dino' not in story_text.lower():
            return False, "Tiny Dino not mentioned in story"

        # Require at least 2 scenes implied
        scene_indicators = ['scene', 'then', 'suddenly', 'next', 'meanwhile', 'after']
        if not any(s in story_text.lower() for s in scene_indicators):
            return False, "No scene structure detected"

        return True, "Passed quick check"

    def format_score_report(self, score: StoryScore) -> str:
        """Format score as readable report."""
        lines = [
            "=" * 50,
            "STORY QUALITY REPORT",
            "=" * 50,
            f"Hook Strength:     {score.hook_strength:5.1f}/100",
            f"Comedy Potential:  {score.comedy_potential:5.1f}/100",
            f"Visual Clarity:    {score.visual_clarity:5.1f}/100",
            f"Uniqueness:        {score.uniqueness:5.1f}/100",
            f"Ending Quality:    {score.ending_quality:5.1f}/100",
            "-" * 50,
            f"TOTAL SCORE:       {score.total:5.1f}/100 (min: {self.min_score})",
            f"RESULT:            {'PASS' if score.passed else 'FAIL'}",
            "=" * 50,
        ]
        if score.feedback:
            lines.append("Feedback:")
            for fb in score.feedback:
                lines.append(f"  - {fb}")
        return "\n".join(lines)
