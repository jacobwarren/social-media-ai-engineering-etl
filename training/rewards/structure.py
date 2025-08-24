from __future__ import annotations
from typing import List
import re
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize


def sentence_structure_reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    scores: List[float] = []
    for i, completion in enumerate(completions):
        user_prompt = prompts[i][-1] if isinstance(prompts[i], list) else prompts[i]
        m = re.search(r"\*\*Sentence Structure\*\*:\s*(.*?)(?:\n|$)", user_prompt)
        requested_structure = None
        if m:
            txt = m.group(1).lower()
            if "short" in txt and "sentences" in txt:
                requested_structure = "short"
            elif "long" in txt and "complex" in txt:
                requested_structure = "long"
            elif "mix" in txt or "balanced" in txt:
                requested_structure = "balanced"
        raw_text = completion if isinstance(completion, str) else completion[0]
        answer_text = raw_text if raw_text.strip() else raw_text
        if not requested_structure:
            # Heuristic fallback: reward reasonable paragraph/sentence structure
            paragraphs = [p.strip() for p in answer_text.split('\n\n') if p.strip()]
            sentences_all = sent_tokenize(answer_text)
            if len(paragraphs) >= 2 and len(sentences_all) >= 2:
                scores.append(0.9)
            elif len(sentences_all) >= 2:
                scores.append(0.7)
            else:
                scores.append(0.5)
            continue
        sentences = sent_tokenize(answer_text)
        if len(sentences) < 2:
            scores.append(0.3)
            continue
        sentence_lengths = [len(word_tokenize(s)) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        length_variance = np.var(sentence_lengths)
        if requested_structure == "short":
            if avg_length < 10:
                score = 1.0
            elif avg_length < 15:
                score = 0.7
            elif avg_length < 20:
                score = 0.4
            else:
                score = 0.2
        elif requested_structure == "long":
            if avg_length > 20:
                score = 1.0
            elif avg_length > 15:
                score = 0.7
            elif avg_length > 10:
                score = 0.4
            else:
                score = 0.2
        else:
            if 10 <= avg_length <= 20 and length_variance > 20:
                score = 1.0
            elif 10 <= avg_length <= 20:
                score = 0.7
            elif length_variance > 20:
                score = 0.6
            else:
                score = 0.4
        scores.append(score)
    return scores

