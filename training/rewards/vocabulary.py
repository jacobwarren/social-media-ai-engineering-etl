from __future__ import annotations
from typing import List
import re
from nltk.tokenize import word_tokenize


def vocabulary_usage_reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    scores: List[float] = []
    for i, completion in enumerate(completions):
        user_prompt = prompts[i][-1] if isinstance(prompts[i], list) else prompts[i]
        m = re.search(r"\*\*Vocabulary Usage\*\*:\s*(.*?)(?:\n|$)", user_prompt)
        requested_vocab = None
        if m:
            t = m.group(1).lower()
            if "rich" in t:
                requested_vocab = "rich"
            elif "developed" in t:
                requested_vocab = "developed"
            elif "normal" in t:
                requested_vocab = "normal"
            elif "conservative" in t or "narrow" in t:
                requested_vocab = "conservative"
        raw_text = completion if isinstance(completion, str) else completion[0]
        answer_text = raw_text if raw_text.strip() else raw_text
        if not requested_vocab:
            scores.append(0.5)
            continue
        words = [w.lower() for w in word_tokenize(answer_text) if w.isalpha()]
        if not words:
            scores.append(0.2)
            continue
        total_words = len(words)
        unique_words = len(set(words))
        vocab_ratio = unique_words / total_words
        if requested_vocab == "rich":
            if vocab_ratio > 0.5:
                score = 1.0
            elif vocab_ratio > 0.4:
                score = 0.8
            elif vocab_ratio > 0.3:
                score = 0.5
            else:
                score = 0.3
        elif requested_vocab == "developed":
            if 0.35 < vocab_ratio <= 0.5:
                score = 1.0
            elif 0.3 < vocab_ratio <= 0.35 or 0.5 < vocab_ratio <= 0.6:
                score = 0.8
            elif 0.25 < vocab_ratio <= 0.3 or 0.6 < vocab_ratio:
                score = 0.5
            else:
                score = 0.3
        elif requested_vocab == "normal":
            if 0.25 < vocab_ratio <= 0.35:
                score = 1.0
            elif 0.2 < vocab_ratio <= 0.25 or 0.35 < vocab_ratio <= 0.4:
                score = 0.8
            elif 0.15 < vocab_ratio <= 0.2 or 0.4 < vocab_ratio <= 0.5:
                score = 0.5
            else:
                score = 0.3
        else:
            if vocab_ratio <= 0.25:
                score = 1.0
            elif vocab_ratio <= 0.3:
                score = 0.8
            elif vocab_ratio <= 0.35:
                score = 0.5
            else:
                score = 0.3
        scores.append(score)
    return scores

