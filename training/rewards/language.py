from __future__ import annotations
import re
from typing import List


def chinese_character_reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    scores: List[float] = []
    pattern = re.compile(r"[\u4e00-\u9fff]")
    for i, completion in enumerate(completions):
        text = completion if isinstance(completion, str) else (completion[0] if completion else "")
        scores.append(0.0 if pattern.search(text) else 1.0)
    return scores

