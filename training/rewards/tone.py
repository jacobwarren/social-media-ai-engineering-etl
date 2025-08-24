from __future__ import annotations
import re
from typing import List
from training.rewards.base import get_sentiment_scores, analyze_sentiment_arc


def tone_alignment_reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    scores: List[float] = []
    for i, completion in enumerate(completions):
        user_prompt = prompts[i][-1] if isinstance(prompts[i], list) else prompts[i]
        answer_text = completion if isinstance(completion, str) else (completion[0] if completion else "")
        tone_match = re.search(r"(?i)\*\*Tone\*\*: \s*([^\n]+)", user_prompt)
        if not tone_match:
            tone_match = re.search(r"(?i)-\s*Tone:\s*([^\n]+)", user_prompt)
        required_tones = [t.strip().lower() for t in (tone_match.group(1) if tone_match else "").split(',') if t.strip()]
        if not required_tones:
            scores.append(0.5)
            continue
        sentiment_scores = get_sentiment_scores(answer_text)
        arc = analyze_sentiment_arc(sentiment_scores)
        tone_map = {
            "friendly": "positive", "cheerful": "positive", "charming": "positive",
            "professional": "neutral", "informative": "neutral", "scholarly": "neutral",
            "serious": "negative", "rebellious": "negative", "sarcastic": "negative",
        }
        per_tone_rewards = []
        for t in required_tones:
            desired = tone_map.get(t, "neutral")
            if desired == "positive":
                per_tone_rewards.append(1.0 if arc in ("Rising",) else 0.6)
            elif desired == "negative":
                per_tone_rewards.append(1.0 if arc in ("Falling",) else 0.6)
            else:
                per_tone_rewards.append(1.0 if arc in ("Flat", "Neutral") else 0.6)
        final_score = sum(per_tone_rewards) / len(per_tone_rewards)
        scores.append(final_score)
    return scores

