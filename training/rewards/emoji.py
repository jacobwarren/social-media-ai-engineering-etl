from __future__ import annotations
import re
from typing import List
import emojis


def categorize_emoji_usage(frequency, bins=None):
    if bins is None:
        bins = [
            (0, "none"),
            (0.0005, "very low"),
            (0.001, "low"),
            (0.005, "medium"),
            (0.01, "high"),
            (1.0, "extreme"),
        ]
    last_label = "none"
    for threshold, label in bins:
        if frequency <= threshold:
            return label
        last_label = label
    return last_label


def emoji_frequency_analysis(post: str):
    emoji_count = emojis.count(post)
    text_length = len(post)
    frequency = 0.0 if text_length == 0 else (emoji_count / text_length)
    usage_label = categorize_emoji_usage(frequency)
    return {
        "emoji_count": emoji_count,
        "text_length": text_length,
        "frequency": frequency,
        "usage": usage_label,
    }


def enhanced_emoji_usage_reward(required_usage: str, completion_text: str) -> float:
    if not required_usage:
        return 0.5
    req = required_usage.lower()
    if req in ["none", "very low", "low", "medium", "high", "extreme"]:
        required_category = req
    elif req == "infrequent":
        required_category = "low"
    elif req == "frequent":
        required_category = "high"
    else:
        required_category = "medium"
    emoji_count = emojis.count(completion_text)
    text_length = len(completion_text)
    frequency = 0 if text_length == 0 else (emoji_count / text_length)
    actual_category = categorize_emoji_usage(frequency)
    order = ["none", "very low", "low", "medium", "high", "extreme"]
    if actual_category == required_category:
        return 1.0
    try:
        d = abs(order.index(required_category) - order.index(actual_category))
        if d == 1:
            return 0.7
        elif d == 2:
            return 0.4
        else:
            return 0.0
    except ValueError:
        return 0.3


def emoji_usage_reward(required_usage: str, completion_text: str) -> float:
    if not required_usage:
        return 0.5
    analysis = emoji_frequency_analysis(completion_text)
    actual_usage = analysis["usage"]
    if actual_usage in ["none"]:
        simple_actual = "none"
    elif actual_usage in ["very low", "low"]:
        simple_actual = "infrequent"
    elif actual_usage in ["medium", "high", "extreme"]:
        simple_actual = "frequent"
    else:
        simple_actual = "infrequent"
    rl = required_usage.lower()
    if rl in ["none"]:
        simple_required = "none"
    elif rl in ["infrequent", "very low", "low"]:
        simple_required = "infrequent"
    elif rl in ["frequent", "medium", "high", "extreme"]:
        simple_required = "frequent"
    else:
        simple_required = "infrequent"
    if simple_required == simple_actual:
        return 1.0
    if simple_required == "none":
        return 0.5 if simple_actual == "infrequent" else 0.0
    elif simple_required == "infrequent":
        if simple_actual == "none":
            return 0.5
        elif simple_actual == "frequent":
            return 0.3
        else:
            return 1.0
    elif simple_required == "frequent":
        if simple_actual == "infrequent":
            return 0.7
        elif simple_actual == "none":
            return 0.0
        else:
            return 1.0
    return 0.0


def enhanced_emoji_usage(prompts: List[str], completions: List[str], ctx=None) -> List[float]:
    scores: List[float] = []
    for i, text in enumerate(completions):
        prompt = prompts[i]
        m = re.search(r"\*\*Emoji Usage\*\*:\s*(.*?)(?:\n|$)", prompt)
        required = m.group(1).strip() if m else ""
        scores.append(enhanced_emoji_usage_reward(required, text if isinstance(text, str) else text[0]))
    return scores

