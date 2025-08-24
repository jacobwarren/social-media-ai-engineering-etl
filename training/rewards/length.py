from __future__ import annotations
import re
from typing import List


def precise_post_length_reward(required_length: str, completion_text: str) -> float:
    if not required_length:
        return 0.5
    length = len(completion_text)
    rl = required_length.lower()
    if "up to 750" in rl:
        max_chars = 750
        if length <= max_chars:
            return 0.7 + (0.3 * min(1.0, length / max_chars))
        else:
            over = (length - max_chars) / max_chars
            if over <= 0.1:
                return 0.6
            elif over <= 0.25:
                return 0.3
            else:
                return 0.0
    elif "between 750 and 1,500" in rl:
        min_chars = 750
        max_chars = 1500
        if length < min_chars:
            return 0.7 * (length / min_chars)
        elif length <= max_chars:
            rng = max_chars - min_chars
            pos = length - min_chars
            return 0.7 + (0.3 * (pos / rng))
        else:
            over = (length - max_chars) / max_chars
            if over <= 0.1:
                return 0.6
            elif over <= 0.25:
                return 0.3
            else:
                return 0.0
    elif "between 1,500 and 3,000" in rl:
        min_chars = 1500
        max_chars = 3000
        if length < min_chars:
            return 0.7 * (length / min_chars)
        elif length <= max_chars:
            rng = max_chars - min_chars
            pos = length - min_chars
            return 0.7 + (0.3 * (pos / rng))
        else:
            over = (length - max_chars) / max_chars
            if over <= 0.1:
                return 0.6
            elif over <= 0.25:
                return 0.3
            else:
                return 0.0
    else:
        return post_length_reward(required_length, completion_text)


def post_length_reward(required_length: str, completion_text: str) -> float:
    if not required_length:
        return 0.5
    length = len(completion_text)
    m = re.search(r"up\s+to\s+(\d+(,\d+)?)\s+characters", required_length, flags=re.IGNORECASE)
    if not m:
        return 0.5
    max_chars_str = m.group(1).replace(",", "")
    max_chars = int(max_chars_str)
    if length <= max_chars:
        return 1.0
    over = (length - max_chars) / max_chars
    if over <= 0.1:
        return 0.7
    elif over <= 0.25:
        return 0.4
    else:
        return 0.0


def precise_post_length(prompts: List[str], completions: List[str]) -> List[float]:
    scores: List[float] = []
    for i, text in enumerate(completions):
        m = re.search(r"\*\*Suggested Post Length\*\*:\s*(.*?)(?:\n|$)", prompts[i])
        req = m.group(1).strip() if m else ""
        scores.append(precise_post_length_reward(req, text if isinstance(text, str) else text[0]))
    return scores

