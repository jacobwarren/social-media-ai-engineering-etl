from __future__ import annotations
import re
import emojis
from typing import List
from training.rewards.base import detect_bullet_styles


def bullet_style_reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """GRPO-compatible bullet style reward (extracted from 26-train-grpo).
    Keeps the same signature and scoring.
    """
    scores: List[float] = []
    for i, c in enumerate(completions):
        user_prompt = prompts[i][-1] if isinstance(prompts[i], list) else prompts[i]
        bullet_style_match = re.search(r"(?i)Bullet\s+Styles?:\s*(.*)", user_prompt)
        if not bullet_style_match:
            bullet_style_match = re.search(r"(?i)\*\*Bullet Styles\*\*:\s*(.*?)(?:\n|$)", user_prompt)
        bullet_style_info = bullet_style_match.group(1).lower().strip() if bullet_style_match else ""

        desired_styles = []
        if "•" in bullet_style_info or "dot" in bullet_style_info:
            desired_styles.append("•")
        if "differing emojis" in bullet_style_info:
            desired_styles.append("Differing Emojis")
        if "emoji" in bullet_style_info:
            desired_styles.append("Emoji")
        if "numbers" in bullet_style_info:
            desired_styles.append("Numbers")
        if "letters" in bullet_style_info:
            desired_styles.append("Letters")

        completion_text = c if isinstance(c, str) else (c[0] if c else "")
        style_detected = detect_bullet_styles(completion_text)

        if not desired_styles:
            scores.append(1.0 if style_detected else 0.0)
            continue
        if not style_detected:
            scores.append(0.0)
            continue

        style_detected_lower = style_detected.lower() if isinstance(style_detected, str) else ""
        match_score = 0.0
        for ds in desired_styles:
            if ds == "•" and (style_detected == "•" or "•" in completion_text):
                match_score = max(match_score, 1.0)
            elif ds == "Differing Emojis" and any(emojis.count(em) > 0 for em in ["🔥", "✅", "🚀", "💡", "📌", "⭐", "⚡"]):
                match_score = max(match_score, 0.8)
            elif ds == "Emoji" and emojis.count(completion_text) > 0:
                match_score = max(match_score, 0.7)
            elif isinstance(style_detected_lower, str) and ds.lower() in style_detected_lower:
                match_score = max(match_score, 0.9)
        scores.append(match_score)
    return scores

