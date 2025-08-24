from __future__ import annotations
from typing import List
from training.rewards.base import analyze_narrative_structure


def narrative_structure_reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    scores: List[float] = []
    for completion in completions:
        raw_text = completion if isinstance(completion, str) else completion[0]
        answer_text = raw_text
        if not answer_text.strip():
            answer_text = raw_text
        flow, pacing, sentiment_arc = analyze_narrative_structure(answer_text)
        structure_score = 0.0
        if pacing not in ["Short/Not Enough Data"]:
            structure_score += 0.4
        if sentiment_arc not in ["Neutral"]:
            structure_score += 0.3
        if flow and flow[0] not in ["Short Content", "Mixed Flow"]:
            structure_score += 0.3
        scores.append(structure_score)
    return scores

