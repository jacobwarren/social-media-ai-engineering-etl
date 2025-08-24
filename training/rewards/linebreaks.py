from __future__ import annotations
from typing import List
import re


def line_break_reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    scores: List[float] = []
    for i, completion in enumerate(completions):
        user_prompt = prompts[i][-1] if isinstance(prompts[i], list) else prompts[i]
        m = re.search(r"\*\*Line Break Usage\*\*:\s*(.*?)(?:\n|$)", user_prompt)
        requested_style = None
        if m:
            t = m.group(1).lower()
            if "frequent" in t:
                requested_style = "frequent"
            elif "fewer" in t or "compact" in t:
                requested_style = "fewer"
            elif "no " in t or "continuous" in t:
                requested_style = "none"
            elif "moderate" in t:
                requested_style = "moderate"
        raw_text = completion if isinstance(completion, str) else completion[0]
        answer_text = raw_text if raw_text.strip() else raw_text
        if not requested_style:
            scores.append(0.5)
            continue
        lines = answer_text.split('\n')
        text_length = len(answer_text)
        line_count = len(lines)
        if text_length == 0 or line_count <= 1:
            line_break_ratio = 0
        else:
            line_break_ratio = (line_count - 1) / text_length * 100
        if requested_style == "frequent":
            if line_break_ratio > 2:
                score = 1.0
            elif line_break_ratio > 1.5:
                score = 0.8
            elif line_break_ratio > 1:
                score = 0.6
            elif line_break_ratio > 0.5:
                score = 0.4
            else:
                score = 0.2
        elif requested_style == "fewer":
            if 0.2 < line_break_ratio <= 0.8:
                score = 1.0
            elif 0 < line_break_ratio <= 0.2 or 0.8 < line_break_ratio <= 1.2:
                score = 0.7
            elif line_break_ratio > 1.2:
                score = 0.3
            else:
                score = 0.5
        elif requested_style == "none":
            if line_break_ratio == 0:
                score = 1.0
            elif line_break_ratio <= 0.2:
                score = 0.7
            elif line_break_ratio <= 0.5:
                score = 0.4
            else:
                score = 0.2
        else:
            if 0.8 < line_break_ratio <= 1.5:
                score = 1.0
            elif 0.5 < line_break_ratio <= 0.8 or 1.5 < line_break_ratio <= 2:
                score = 0.8
            elif 0.2 < line_break_ratio <= 0.5 or 2 < line_break_ratio <= 2.5:
                score = 0.5
            else:
                score = 0.3
        scores.append(score)
    return scores

