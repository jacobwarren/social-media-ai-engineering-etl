from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Union

ScoreList = List[float]
FuncMap = Dict[str, Callable[[List[str], List[str]], ScoreList]]
WeightMap = Dict[str, float]


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def aggregate_rewards(
    prompts: List[str],
    completions: List[str],
    funcs: FuncMap,
    weights: WeightMap,
    normalize: Union[None, str, Callable[[float], float]] = None,
    return_components: bool = False,
) -> Union[ScoreList, Tuple[ScoreList, Dict[str, ScoreList]]]:
    """
    Combine multiple reward signals into a single scalar per sample.

    - funcs: mapping name -> callable(prompts, completions) -> List[float]
    - weights: mapping name -> float
    - normalize: None | 'clip' | callable(float)->float applied element-wise
    - return_components: if True, return (aggregate, per_component_scores)
    """
    # Compute individual scores
    per_scores: Dict[str, ScoreList] = {}
    for name, fn in funcs.items():
        try:
            vals = fn(prompts, completions)
        except Exception:
            # On failure, degrade gracefully with zeros the right length
            vals = [0.0 for _ in range(len(completions))]
        # Optional normalization
        if normalize == 'clip':
            vals = [_clip01(float(v)) for v in vals]
        elif callable(normalize):
            vals = [float(normalize(float(v))) for v in vals]
        else:
            vals = [float(v) for v in vals]
        per_scores[name] = vals

    # Weighted aggregate
    agg: ScoreList = []
    for i in range(len(completions)):
        total = 0.0
        wsum = 0.0
        for name, vals in per_scores.items():
            if i >= len(vals):
                continue
            w = float(weights.get(name, 1.0))
            total += vals[i] * w
            wsum += w
        agg.append(total / wsum if wsum else 0.0)

    return (agg, per_scores) if return_components else agg

