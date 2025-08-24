from __future__ import annotations

import re
from collections import Counter
from typing import List, Set, Tuple
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize

from training.grpo.nlp_setup import init_nlp_context

# Shared NLP context (mirrors legacy globals)
_ctx = init_nlp_context()
lemmatizer = _ctx.lemmatizer
stop_words = _ctx.stop_words
sia = _ctx.sia
nlp = _ctx.nlp


def extract_keywords(text: str, max_keywords: int = 15) -> Set[str]:
    if not nlp:
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 3]
        return set(tokens[:max_keywords])
    doc = nlp(text[:10000])
    entities = [ent.text.lower() for ent in doc.ents if len(ent.text) > 3]
    important_words = [
        token.text.lower()
        for token in doc
        if token.pos_ in ["NOUN", "PROPN", "ADJ"]
        and token.text.lower() not in stop_words
        and len(token.text) > 3
    ]
    all_keywords = entities + important_words
    counter = Counter(all_keywords)
    return set([word for word, _ in counter.most_common(max_keywords)])


def detect_bullet_styles(text: str) -> str | None:
    lines = text.split('\n')
    bullet_list: List[str] = []
    numbered_pattern = re.compile(r'^\s*\d+[\.\)]\s+')
    lettered_pattern = re.compile(r'^\s*[a-zA-Z]+[\.\)]\s+')
    symbolic_pattern = re.compile(r'^\s*([^\w\s])')

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        if numbered_pattern.match(stripped_line):
            bullet_list.append('Numbers')
        elif lettered_pattern.match(stripped_line):
            bullet_list.append('Letters')
        else:
            m = symbolic_pattern.match(stripped_line)
            if m:
                bullet_list.append(m.group(1))

    if not bullet_list:
        return None

    counts = Counter(bullet_list)
    if len(counts) > 1:
        return "Mixed Bullet Styles"
    style, _ = counts.most_common(1)[0]
    return style


def get_sentiment_scores(text: str) -> List[float]:
    sentences = sent_tokenize(text[:5000])
    if len(sentences) > 10:
        step = max(1, len(sentences) // 10)
        sentences = sentences[::step]
    sentiment_scores: List[float] = []
    for sentence in sentences:
        if not sia:
            sentiment_scores.append(0.0)
            continue
        score = sia.polarity_scores(sentence)['compound']
        sentiment_scores.append(score)
    return sentiment_scores


def analyze_sentiment_arc(sentiment_scores: List[float]) -> str:
    if len(sentiment_scores) < 3:
        return "Neutral"
    first, middle, last = sentiment_scores[0], sentiment_scores[len(sentiment_scores)//2], sentiment_scores[-1]
    if first < middle < last and last > 0.2:
        return "Rising"
    elif first > middle > last and last < -0.2:
        return "Falling"
    elif abs(last - first) < 0.1 and abs(middle) < 0.1:
        return "Flat"
    else:
        return "Variable"


def analyze_narrative_flow(text: str) -> str:
    sentences = sent_tokenize(text[:5000])
    if len(sentences) < 3:
        return "Short/Not Enough Data"
    keywords = extract_keywords(text)
    transitions = 0
    prev_keywords = None
    for s in sentences:
        s_k = extract_keywords(s)
        if prev_keywords is not None:
            overlap = len(keywords.intersection(s_k))
            if overlap < 2:
                transitions += 1
        prev_keywords = s_k
    if transitions <= 1:
        return "Smooth"
    elif transitions <= 3:
        return "Moderate"
    else:
        return "Choppy"


def analyze_pacing(text: str) -> str:
    sentences = sent_tokenize(text[:5000])
    if len(sentences) < 3:
        return "Short/Not Enough Data"
    sentence_lengths = [len(word_tokenize(s)) for s in sentences]
    avg_len = np.mean(sentence_lengths)
    stddev = np.std(sentence_lengths)
    if stddev > 7:
        return "Variable"
    elif avg_len < 10:
        return "Fast"
    elif avg_len > 20:
        return "Slow"
    else:
        return "Moderate"


def analyze_narrative_structure(text: str) -> tuple[str, str, str]:
    flow = analyze_narrative_flow(text)
    pacing = analyze_pacing(text)
    sentiment_scores = get_sentiment_scores(text)
    sentiment_arc = analyze_sentiment_arc(sentiment_scores)
    return flow, pacing, sentiment_arc

