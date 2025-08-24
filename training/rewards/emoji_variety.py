from __future__ import annotations
import emojis


def emoji_variety_reward(completion_text: str) -> float:
    all_emoji = [c for c in completion_text if emojis.count(c) > 0]
    total_emoji = len(all_emoji)
    if total_emoji == 0:
        return 0.0
    unique_emoji = set(all_emoji)
    unique_count = len(unique_emoji)
    variety_ratio = unique_count / total_emoji
    if variety_ratio >= 0.9:
        return 1.0
    elif variety_ratio >= 0.7:
        return 0.8
    elif variety_ratio >= 0.5:
        return 0.6
    else:
        return 0.4

