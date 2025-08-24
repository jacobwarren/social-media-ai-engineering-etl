from __future__ import annotations
from typing import List
from nltk.tokenize import sent_tokenize
from training.rewards.base import analyze_narrative_structure


def semantic_coherence_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    scores: List[float] = []
    for completion in completions:
        raw_text = completion if isinstance(completion, str) else completion[0]
        answer_text = raw_text if raw_text.strip() else raw_text
        sentences = sent_tokenize(answer_text)
        if len(sentences) < 3:
            # Heuristic: short but on-topic single-sentence should get some credit
            scores.append(0.6 if len(sentences) >= 1 else 0.5)
            continue
        flow, pacing, sentiment_arc = analyze_narrative_structure(answer_text)
        score = 0.5
        if pacing not in ["Short/Not Enough Data"]:
            score += 0.25
        if sentiment_arc not in ["Neutral"]:
            score += 0.25
        scores.append(min(1.0, max(0.0, score)))
    return scores

