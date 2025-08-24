from __future__ import annotations

from typing import List
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize

from .context import FeatureContext


def analyze_narrative_flow(text: str, max_sentences: int = 20) -> List[str]:
    sentences = sent_tokenize(text[:5000])
    sentences = sentences[:max_sentences]
    if not sentences:
        return []
    intro_cues = {"today", "announce", "we're", "introduce", "sharing"}
    outro_cues = {"follow", "check out", "sign up", "learn more", "share", "comment"}
    labels: List[str] = []
    for i, s in enumerate(sentences):
        ls = s.lower()
        if i <= 1 and any(c in ls for c in intro_cues):
            labels.append("Introduction/Setup")
        elif i >= len(sentences) - 2 and any(c in ls for c in outro_cues):
            labels.append("Outro/CTA")
        else:
            labels.append("Content")
    return labels


def analyze_pacing(text: str, max_sentences: int = 50) -> str:
    sentences = sent_tokenize(text[:5000])
    if len(sentences) < 3:
        return "Short/Not Enough Data"
    sentences = sentences[:max_sentences]
    lens = [len(word_tokenize(s)) for s in sentences]
    p75 = np.percentile(lens, 75)
    if p75 <= 10:
        return "Fast"
    elif p75 > 20:
        return "Slow"
    else:
        return "Moderate"


def rolling_average(values, window=3):
    if len(values) < window:
        return values
    return [float(np.mean(values[i:i+window])) for i in range(len(values) - window + 1)]


def analyze_sentiment_arc(sentiment_scores, window: int = 3, short_text_threshold: int = 4) -> str:
    n = len(sentiment_scores)
    if n < short_text_threshold:
        return "Short/Not Enough Data for Arc"
    smoothed = rolling_average(sentiment_scores, window) if window and n >= window else sentiment_scores
    t = np.arange(len(smoothed))
    if len(t) < 2:
        return "Short/Not Enough Data for Arc"
    try:
        slope = float(np.polyfit(t, smoothed, 1)[0])
    except Exception:
        slope = 0.0
    if slope > 0.03:
        return "Rising"
    elif slope < -0.03:
        return "Falling"
    else:
        return "Flat"

