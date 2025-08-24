from __future__ import annotations
from typing import List
from nltk.tokenize import sent_tokenize

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None


def topic_shifts_reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    scores: List[float] = []
    for completion in completions:
        raw_text = completion if isinstance(completion, str) else completion[0]
        answer_text = raw_text if raw_text.strip() else raw_text
        sentences = sent_tokenize(answer_text)
        if len(sentences) < 3:
            scores.append(0.5)
            continue
        paragraphs = [p.strip() for p in answer_text.split('\n\n') if p.strip()]
        if nlp:
            paragraph_docs = [nlp(p[:1000]) for p in paragraphs]
            similarities = []
            for i in range(1, len(paragraph_docs)):
                similarities.append(paragraph_docs[i].similarity(paragraph_docs[i-1]))
            avg_sim = sum(similarities) / len(similarities) if similarities else 0.5
        else:
            avg_sim = 0.5
        score = 1.0 - abs(0.5 - avg_sim)
        scores.append(max(0.0, min(1.0, score)))
    return scores

