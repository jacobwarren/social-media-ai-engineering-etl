import json
import argparse
from pathlib import Path

import streamlit as st

# Updated to use GRPO-compatible modular rewards directly
from training.rewards.bullet_style import bullet_style_reward_func
from training.rewards.tone import tone_alignment_reward_func
from training.rewards.hashtags import hashtag_limit_reward_func
from training.rewards.emoji import enhanced_emoji_usage
from training.rewards.length import precise_post_length
from training.rewards.structure import sentence_structure_reward_func
from training.rewards.semantic import semantic_coherence_reward

from training.rewards.aggregator import aggregate_rewards


def load_weights(weights_path: str | None):
    default = {"bullet": 1.0, "tone": 1.0, "hashtags": 1.0, "length": 1.0, "emoji": 1.0, "structure": 0.5, "coherence": 0.5}
    if not weights_path:
        return default
    try:
        with open(weights_path, "r", encoding="utf-8") as f:
            user = json.load(f)
        for k in list(default.keys()):
            if k in user:
                default[k] = float(user[k])
    except Exception:
        pass
    return default


def main(weights_path: str | None = None):
    st.set_page_config(page_title="Reward Scoring Demo", layout="centered")
    st.title("Reward Scoring Demo")

    weights = load_weights(weights_path)

    with st.sidebar:
        st.header("Weights")
        for k in list(weights.keys()):
            weights[k] = st.slider(k, min_value=0.0, max_value=2.0, value=float(weights[k]), step=0.1)
        if st.button("Reset to defaults"):
            st.session_state.clear()

    st.subheader("Prompt (constraints)")
    prompt = st.text_area("Enter a prompt with constraints (e.g., tone, length, emoji usage)", height=200, value="""### Request\nCreate a LinkedIn post on the topic of: `AI tools`\n\n### Writing Constraints\n- **Suggested Post Length**: Up to 100 characters\n- **Emoji Usage**: none\n- **Tone**: Serious\n""")

    st.subheader("Completion")
    completion = st.text_area("Paste a completion to score", height=200, value="This is a serious note. No emoji.")

    if st.button("Score"):
        prompts = [prompt]
        completions = [completion]

        # Individual rewards (GRPO-compatible functions)
        scores = {
            "bullet": bullet_style_reward_func(prompts, completions),
            "tone": tone_alignment_reward_func(prompts, completions),
            "hashtags": hashtag_limit_reward_func(prompts, completions),
            # For demo: approximate length via simple cap (use precise_post_length from 26 if needed)
            "length": precise_post_length(prompts, completions),
            "emoji": enhanced_emoji_usage(prompts, completions),
            "structure": sentence_structure_reward_func(prompts, completions),
            "coherence": semantic_coherence_reward(prompts, completions),
        }

        funcs = {k: (lambda name: (lambda p, c: scores[name]))(k) for k in scores.keys()}
        agg = aggregate_rewards(prompts, completions, funcs, weights)

        st.write("### Results")
        for k, v in scores.items():
            st.write(f"{k}: {float(v[0]):.3f}")
        st.write(f"\n**Aggregate:** {float(agg[0]):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()
    main(args.weights)

