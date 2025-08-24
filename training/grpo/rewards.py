# Extracted reward functions from 26-train-grpo.py (PR3)
# No behavior changes intended.

from __future__ import annotations

import re
import numpy as np
import emojis
from typing import List
from nltk.tokenize import word_tokenize, sent_tokenize

from training.grpo.nlp_setup import init_nlp_context
from training.grpo.prompt_parsing import (
    parse_writing_style_block,
    extract_prompt_content,
    detect_urls,
    detect_potential_people_names,
    detect_organization_names,
)
from training.grpo.scenarios import get_scenario_type, normalize_scenario_score

# Initialize NLP context local to this module to mirror original globals
_ctx = init_nlp_context()
lemmatizer = _ctx.lemmatizer
stop_words = _ctx.stop_words
sia = _ctx.sia
nlp = _ctx.nlp


# ---------- Helper analysis functions (verbatim logic) ----------

def extract_keywords(text, max_keywords=15):
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
    from collections import Counter
    counter = Counter(all_keywords)
    return set([word for word, _ in counter.most_common(max_keywords)])


def detect_bullet_styles(text):
    lines = text.split('\n')
    bullet_list = []
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

    from collections import Counter
    counts = Counter(bullet_list)
    if len(counts) > 1:
        return "Mixed Bullet Styles"
    style, _ = counts.most_common(1)[0]
    return style


def get_sentiment_scores(text):
    sentences = sent_tokenize(text[:5000])
    if len(sentences) > 10:
        step = max(1, len(sentences) // 10)
        sentences = sentences[::step]
    sentiment_scores = []
    for sentence in sentences:
        if not sia:
            sentiment_scores.append(0.0)
            continue
        score = sia.polarity_scores(sentence)['compound']
        sentiment_scores.append(score)
    return sentiment_scores


def analyze_sentiment_arc(sentiment_scores: List[float]):
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


def analyze_narrative_flow(text):
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


def analyze_pacing(text):
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


def analyze_narrative_structure(text):
    flow = analyze_narrative_flow(text)
    pacing = analyze_pacing(text)
    sentiment_scores = get_sentiment_scores(text)
    sentiment_arc = analyze_sentiment_arc(sentiment_scores)
    return flow, pacing, sentiment_arc


# ---------- Reward functions (verbatim signatures) ----------

def bullet_style_reward_func(prompts, completions, **kwargs) -> list[float]:
    scores = []
    for i, c in enumerate(completions):
        user_prompt = prompts[i][-1] if isinstance(prompts[i], list) else prompts[i]
        bullet_style_match = re.search(r"(?i)Bullet\s+Styles?:\s*(.*)", user_prompt)
        if not bullet_style_match:
            bullet_style_match = re.search(r"(?i)\*\*Bullet Styles\*\*:\s*(.*?)(?:\n|$)", user_prompt)
        bullet_style_info = bullet_style_match.group(1).lower().strip() if bullet_style_match else ""
        desired_styles = []
        if "•" in bullet_style_info or "dot" in bullet_style_info:
            desired_styles.append("•")
        if "differing emojis" in bullet_style_info:
            desired_styles.append("Differing Emojis")
        if "emoji" in bullet_style_info:
            desired_styles.append("Emoji")
        if "numbers" in bullet_style_info:
            desired_styles.append("Numbers")
        if "letters" in bullet_style_info:
            desired_styles.append("Letters")
        completion_text = c if isinstance(c, str) else (c[0] if c else "")
        style_detected = detect_bullet_styles(completion_text)
        if not desired_styles:
            scores.append(1.0 if style_detected else 0.0)
            continue
        if not style_detected:
            scores.append(0.0)
            continue
        style_detected_lower = style_detected.lower()
        match_score = 0.0
        for ds in desired_styles:
            if ds == "•" and (style_detected == "•" or "•" in completion_text):
                match_score = max(match_score, 1.0)
            elif ds == "Differing Emojis" and any(emojis.count(em) > 0 for em in ["🔥", "✅", "🚀", "💡", "📌", "⭐", "⚡"]):
                match_score = max(match_score, 0.8)
            elif ds == "Emoji" and emojis.count(completion_text) > 0:
                match_score = max(match_score, 0.7)
            elif ds.lower() in style_detected_lower:
                match_score = max(match_score, 0.9)
        scores.append(match_score)
    return scores


def tone_alignment_reward_func(prompts, completions, **kwargs) -> list[float]:
    scores = []
    for i, completion in enumerate(completions):
        user_prompt = prompts[i][-1] if isinstance(prompts[i], list) else prompts[i]
        answer_text = completion if isinstance(completion, str) else (completion[0] if completion else "")
        # Extract tone requirement
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


def hashtag_limit_reward_func(prompts, completions, **kwargs) -> list[float]:
    scores = []
    for i, completion in enumerate(completions):
        text = completion if isinstance(completion, str) else (completion[0] if completion else "")
        # Count hashtags at end
        tail = text.split('\n')[-1]
        hashtags = re.findall(r"#[A-Za-z0-9_]+", tail)
        score = 1.0 if len(hashtags) <= 3 else max(0.0, 1.0 - 0.2 * (len(hashtags) - 3))
        scores.append(score)
    return scores


def chinese_character_reward_func(prompts, completions, **kwargs) -> list[float]:
    scores = []
    pattern = re.compile(r"[\u4e00-\u9fff]")
    for i, completion in enumerate(completions):
        text = completion if isinstance(completion, str) else (completion[0] if completion else "")
        if pattern.search(text):
            scores.append(0.0)
        else:
            scores.append(1.0)
    return scores



__all__ = [
    'bullet_style_reward_func',
    'tone_alignment_reward_func',
    'hashtag_limit_reward_func',
    'chinese_character_reward_func',
    'extract_keywords',
    'detect_bullet_styles',
    'get_sentiment_scores',
    'analyze_sentiment_arc',
    'analyze_narrative_flow',
    'analyze_pacing',
    'analyze_narrative_structure',
    # Expose names used elsewhere
    'narrative_structure_reward_func',
    'emoji_variety_reward',
    'semantic_coherence_reward',
    'sentence_structure_reward_func',
    'vocabulary_usage_reward_func',
    'line_break_reward_func',
    'punctuation_usage_reward_func',
    'divider_style_reward_func',
    'topic_shifts_reward_func',
]
