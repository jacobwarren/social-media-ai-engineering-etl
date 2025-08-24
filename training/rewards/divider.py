from __future__ import annotations
from typing import List
import re


def divider_style_reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    scores: List[float] = []
    for i, completion in enumerate(completions):
        user_prompt = prompts[i][-1] if isinstance(prompts[i], list) else prompts[i]
        m = re.search(r"\*\*Section Divider\*\*:\s*`([^\`]+)`", user_prompt)
        requested_divider = m.group(1) if m else None
        raw_text = completion if isinstance(completion, str) else completion[0]
        answer_text = raw_text if raw_text.strip() else raw_text
        if not requested_divider:
            scores.append(0.5)
            continue
        lines = answer_text.split('\n')
        found_dividers: List[str] = []
        for line in lines:
            line = line.strip()
            if line and len(line) >= 3:
                char = line[0]
                if all(c == char for c in line) and len(line) >= 3:
                    found_dividers.append(char)
                elif len(line) >= 5 and line[0] == line[2] and all(line[i] == ' ' for i in range(1, len(line), 2)):
                    found_dividers.append(line[0])
        if not found_dividers:
            scores.append(0.0)
        elif requested_divider in found_dividers:
            scores.append(1.0)
        else:
            scores.append(0.3)
    return scores

