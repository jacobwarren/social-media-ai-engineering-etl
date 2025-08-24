import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd

from training.rewards.bullet_style import bullet_style_reward_func
from training.rewards.tone import tone_alignment_reward_func
from training.rewards.hashtags import hashtag_limit_reward_func
from training.rewards.emoji import enhanced_emoji_usage
from training.rewards.length import precise_post_length
from training.rewards.structure import sentence_structure_reward_func
from training.rewards.semantic import semantic_coherence_reward

from training.rewards.aggregator import aggregate_rewards


def sample_prompts() -> list[dict]:
    # Minimal prompt set across constraints
    return [
        {"name": "bullet_numbered", "prompt": "**Bullet Styles**: Numbered"},
        {"name": "tone_serious", "prompt": "**Tone**: Serious"},
        {"name": "hashtags_limit", "prompt": "Limit hashtags to 3"},
        {"name": "length_short", "prompt": "**Suggested Post Length**: Up to 100 characters"},
        {"name": "emoji_none", "prompt": "**Emoji Usage**: none"},
        {"name": "structure", "prompt": "Write in two short paragraphs."},
        {"name": "coherence", "prompt": "Discuss features X and Y for developers."},
    ]


def generate_completion(prompt: str) -> str:
    # Extremely simple deterministic generator for CPU-only evaluation
    base = prompt[:60]
    rand = random.Random(3407)
    suffix = [" This is a serious note.", " 1. First\n 2. Second", " #ai #ml #prod", " No emoji."]
    return base + rand.choice(suffix)


def main():
    ap = argparse.ArgumentParser(description="Evaluate modular rewards and aggregator")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--base-dir", default="reports")
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    prompts = sample_prompts()
    names = [p["name"] for p in prompts]
    texts = [p["prompt"] for p in prompts]
    completions = [generate_completion(p) for p in texts]

    # Individual rewards
    results = {
        "bullet": bullet_style_reward_func(texts, completions),
        "tone": tone_alignment_reward_func(texts, completions),
        "hashtags": hashtag_limit_reward_func(texts, completions),
        "length": precise_post_length(texts, completions),
        "emoji": enhanced_emoji_usage(texts, completions),
        "structure": sentence_structure_reward_func(texts, completions),
        "coherence": semantic_coherence_reward(texts, completions),
    }

    # Aggregated
    funcs = {
        k: (lambda name: (lambda p, c: results[name]))(k) for k in results.keys()
    }
    weights = {"bullet": 1.0, "tone": 1.0, "hashtags": 1.0, "length": 1.0, "emoji": 1.0, "structure": 0.5, "coherence": 0.5}
    if args.weights and os.path.exists(args.weights):
        try:
            with open(args.weights, "r", encoding="utf-8") as f:
                user_w = json.load(f)
            for k in weights:
                if k in user_w:
                    weights[k] = float(user_w[k])
        except Exception:
            pass

    agg = aggregate_rewards(texts, completions, funcs, weights)

    # Save outputs
    out_dir = Path(args.base_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"name": names, "prompt": texts, "completion": completions, **{f"reward_{k}": v for k, v in results.items()}, "aggregate": agg})
    df.to_csv(out_dir / "rewards.csv", index=False)
    try:
        df.to_parquet(out_dir / "rewards.parquet", index=False)
    except Exception:
        pass

    # Markdown report
    md = ["# Rewards Evaluation", "", f"Run: {args.run_id}", "", "## Aggregate scores", ""]
    md.append(df[["name", "aggregate"]].to_markdown(index=False))
    md.append("\n\n## Weights\n")
    md.append("```json\n" + json.dumps(weights, indent=2) + "\n```")
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote evaluation to {out_dir}")


if __name__ == "__main__":
    main()

