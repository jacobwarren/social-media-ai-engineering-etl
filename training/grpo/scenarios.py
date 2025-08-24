from typing import Dict


def normalize_scenario_score(score: float, scenario_id: int) -> float:
    """Standardize scores across scenarios to a 0-1 range.
    Uses the same max table as in 26-train-grpo.py.
    """
    max_scores: Dict[int, float] = {
        0: 10.0,  # LinkedIn post (sum of all weights)
        1: 10.0,  # Topic classification
        2: 10.0,  # Opinion extraction
        3: 10.0,  # Tone analysis
        4: 10.0,  # Structure classification
        5: 10.0,  # Generic scenario
    }
    normalized = score / max_scores.get(scenario_id, 10.0)
    return min(normalized, 1.0)


def get_scenario_type(user_prompt: str) -> int:
    """Identify the scenario type from the markdown-formatted prompt.
    Mirrors the logic in 26-train-grpo.py.
    """
    # Look across the whole prompt for the request phrase to be robust to formatting
    lp = user_prompt.lower()
    if "create a linkedin post that" in lp:
        return 0

    if "analyze the following social media post and identify its primary topic" in lp:
        return 1
    if "extract the core opinion from this social media post and present it in first person" in lp:
        return 2
    if "analyze this social media post and identify up to three primary tones" in lp:
        return 3
    if "classify the structural format of this social media post" in lp:
        return 4

    return 5

