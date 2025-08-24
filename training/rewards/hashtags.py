from __future__ import annotations
import re
from typing import List


def hashtag_limit_reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    scores: List[float] = []
    for i, completion in enumerate(completions):
        text = completion if isinstance(completion, str) else (completion[0] if completion else "")
        tail = text.split('\n')[-1]
        hashtags = re.findall(r"#[A-Za-z0-9_]+", tail)
        score = 1.0 if len(hashtags) <= 3 else max(0.0, 1.0 - 0.2 * (len(hashtags) - 3))
        scores.append(score)
    return scores

