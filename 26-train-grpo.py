import argparse
parser = argparse.ArgumentParser(description="GRPO training")
parser.add_argument("--run-id", default=None)
parser.add_argument("--base-dir", default="data/processed")
parser.add_argument("--models-dir", default="models")
parser.add_argument("--use-aggregator", action="store_true", default=False)
parser.add_argument("--weights", default=None, help="Path to JSON file with reward weights")
parser.add_argument("--seed", type=int, default=3407)
args, _ = parser.parse_known_args()

import os


# ETL cohesion: logging, manifest, run-id latest resolution
from utils.logging_setup import init_pipeline_logging
from utils.manifest import read_manifest, discover_input, should_skip, compute_hash, update_stage
from utils.run_id import get_last_run_id

logger = init_pipeline_logging("phase3.train_grpo", None, "26-train-grpo")

if args.run_id == "latest":
    latest = get_last_run_id(args.base_dir)
    if latest:
        args.run_id = latest

# Early idempotent skip if model already exists when run-id provided
if args.run_id:
    out_dir = os.path.join(args.models_dir, args.run_id, "grpo-model")
    if os.path.isdir(out_dir):
        logger.info(f"Model already exists at {out_dir}; skipping training")
        raise SystemExit(0)


# -*- coding: utf-8 -*-
import os
import re
import torch
import nltk
import emojis
import numpy as np
import spacy
from thefuzz import process, fuzz
from utils.seed import set_global_seed
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import wandb
from datasets import load_dataset, Dataset

# Patching TRL for GRPO must happen before trainer/model construction
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from unsloth import is_bfloat16_supported

# Centralized NLP setup (preserve globals for downstream calls)
from training.grpo.nlp_setup import init_nlp_context
_ctx = init_nlp_context()
lemmatizer = _ctx.lemmatizer
stop_words = _ctx.stop_words
sia = _ctx.sia
nlp = _ctx.nlp

# Import reward functions and helpers from pipe/training/rewards (authoritative)
from training.rewards.bullet_style import bullet_style_reward_func
from training.rewards.tone import tone_alignment_reward_func
from training.rewards.hashtags import hashtag_limit_reward_func
from training.rewards.language import chinese_character_reward_func
from training.rewards.length import precise_post_length_reward, post_length_reward
from training.rewards.emoji import enhanced_emoji_usage_reward, emoji_usage_reward, emoji_frequency_analysis
from training.rewards.emoji_variety import emoji_variety_reward
from training.rewards.vocabulary import vocabulary_usage_reward_func
from training.rewards.linebreaks import line_break_reward_func
from training.rewards.punctuation import punctuation_usage_reward_func
from training.rewards.divider import divider_style_reward_func
from training.rewards.semantic import semantic_coherence_reward
from training.rewards.fabrication import fabrication_detection_reward_func
from training.rewards.structure import sentence_structure_reward_func
from training.rewards.topics import topic_shifts_reward_func
from training.rewards.narrative import narrative_structure_reward_func

# Parsing helpers
from training.grpo.prompt_parsing import (
    parse_writing_style_block,
    extract_prompt_content,
    extract_analysis_content,
    detect_urls,
    detect_potential_people_names,
    detect_organization_names,
)

# Scenario helpers
from training.grpo.scenarios import get_scenario_type, normalize_scenario_score

set_global_seed(3407)
# --- 2) Load the Base Model with Unsloth (4-bit + LoRA for GRPO) ---


def main():
    """Main training function to run the enhanced GRPO training."""
    print("Starting GRPO Training...")
    from training.grpo.model import load_base_model_and_tokenizer

    # Resolve SFT model directory: use models/<run-id>/sft-model when available, else ./sft-model
    sft_model_dir = "./sft-model"
    try:
        if args.run_id:
            candidate = os.path.join(args.models_dir, args.run_id, "sft-model")
            if os.path.isdir(candidate):
                sft_model_dir = candidate
                logger.info(f"Using SFT model from {sft_model_dir}")
            else:
                logger.warning(f"Expected SFT model at {candidate} not found; falling back to {sft_model_dir}")
    except Exception:
        pass

    # Use absolute path to ensure Unsloth treats it as a local directory
    sft_model_dir = os.path.abspath(sft_model_dir)
    model, tokenizer, is_vllm_capable = load_base_model_and_tokenizer(
        model_name=sft_model_dir,
        max_seq_length=4096,
        lora_rank=64,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=64,
        gpu_memory_utilization=0.6,
    )


    def get_balanced_sft(split="train") -> Dataset:
        # Discover DPO CSV via manifest if run-id provided, else fallback
        csv_path = "dpo.csv"
        try:
            if args.run_id:
                manifest = read_manifest(args.run_id, args.base_dir)
                out_paths = []
                try:
                    stage_meta = manifest.get("stages", {}).get("24-add-negatives")
                    if stage_meta:
                        outs = stage_meta.get("outputs") or stage_meta.get("output") or []
                        out_paths.extend(outs if isinstance(outs, list) else [outs])
                except Exception:
                    pass
                if not out_paths:
                    try:
                        stage_meta = manifest.get("stages", {}).get("23-split")
                        if stage_meta:
                            outs = stage_meta.get("outputs") or stage_meta.get("output") or []
                            out_paths.extend(outs if isinstance(outs, list) else [outs])
                    except Exception:
                        pass
                # Prefer 24-dpo-ready.csv, else 23-dpo.csv, else first discovered
                candidates = [p for p in out_paths if isinstance(p, str)]
                preferred = [p for p in candidates if p.endswith("24-dpo-ready.csv")] or [p for p in candidates if p.endswith("23-dpo.csv")] or candidates
                if preferred:
                    csv_path = preferred[0]
                logger.info(f"Using DPO CSV: {csv_path}")
        except Exception:
            pass

        data = load_dataset("csv", data_files=csv_path)["train"]

        # Filter for rows that have a 'prompt' and 'chosen'
        def filter_samples(sample):
            return sample.get("prompt") is not None and sample.get("chosen") is not None

        data = data.filter(filter_samples)

        def map_to_grpo(sample):
            # We'll build a user prompt
            chat_prompt = [
                {"role": "user", "content": sample["prompt"]},
            ]
            # Flatten into a single string
            combined_string = tokenizer.apply_chat_template(
                chat_prompt, tokenize=False, add_generation_prompt=True
            )

            # Instead of adding <|start_header_id|> etc.
            # we'll just store the "answer" as the raw text (sample["chosen"])
            return {
                "prompt": combined_string,
                "answer": sample['chosen']  # raw text
            }
        data = data.map(map_to_grpo)
        return data


    # Define a default aggregated reward available regardless of --use-aggregator
    def default_aggregated_reward(prompts, completions, **kwargs):
        # Import inside to avoid import-time issues if this path changes
        from training.rewards.aggregator import aggregate_rewards
        funcs = {
            "bullet": lambda p, c: bullet_style_reward_func(p, c, **kwargs),
            "tone": lambda p, c: tone_alignment_reward_func(p, c, **kwargs),
            "hashtags": lambda p, c: hashtag_limit_reward_func(p, c, **kwargs),
            "length": lambda p, c: [
                precise_post_length_reward(
                    parse_writing_style_block(p[i]).get("post_length_requirement"),
                    (c[i] if isinstance(c[i], str) else c[i][0])
                ) for i in range(len(c))
            ],
            "emoji": lambda p, c: [
                enhanced_emoji_usage_reward(
                    parse_writing_style_block(p[i]).get("emoji_usage_requirement"),
                    (c[i] if isinstance(c[i], str) else c[i][0])
                ) for i in range(len(c))
            ],
            "structure": lambda p, c: sentence_structure_reward_func(p, c, **kwargs),
            "coherence": lambda p, c: semantic_coherence_reward(p, c, **kwargs),
        }
        import json
        weights = {"bullet": 1.0, "tone": 1.0, "hashtags": 1.0, "length": 1.0, "emoji": 1.0, "structure": 0.5, "coherence": 0.5}
        if args.weights:
            try:
                with open(args.weights, "r", encoding="utf-8") as f:
                    user_weights = json.load(f)
                for k in list(weights.keys()):
                    if k in user_weights:
                        weights[k] = float(user_weights[k])
            except Exception:
                pass
        return aggregate_rewards(prompts, completions, funcs, weights)

    # Initialize with default; may be overridden under --use-aggregator block
    aggregated_reward = default_aggregated_reward




    dataset = get_balanced_sft()
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42, shuffle=True)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    from training.grpo.trainer import build_training_args
    # Use vLLM only if the loader returned an Unsloth-backed model
    training_args = build_training_args(output_dir="grpo-results", use_vllm=is_vllm_capable)
    # Optional: use aggregator to blend reward signals from multiple simple rewards
    if args.use_aggregator:
        try:
            # Build aggregate reward function over individual reward components
            def aggregated_reward(prompts, completions, **kwargs):
                from training.rewards.aggregator import aggregate_rewards
                funcs = {
                    "bullet": lambda p, c: bullet_style_reward_func(p, c, **kwargs),
                    "tone": lambda p, c: tone_alignment_reward_func(p, c, **kwargs),
                    "hashtags": lambda p, c: hashtag_limit_reward_func(p, c, **kwargs),
                    "length": lambda p, c: [
                        precise_post_length_reward(
                            parse_writing_style_block(p[i]).get("post_length_requirement"),
                            (c[i] if isinstance(c[i], str) else c[i][0])
                        ) for i in range(len(c))
                    ],
                    "emoji": lambda p, c: [
                        enhanced_emoji_usage_reward(
                            parse_writing_style_block(p[i]).get("emoji_usage_requirement"),
                            (c[i] if isinstance(c[i], str) else c[i][0])
                        ) for i in range(len(c))
                    ],
                    "vocabulary": lambda p, c: vocabulary_usage_reward_func(p, c, **kwargs),
                    "linebreaks": lambda p, c: line_break_reward_func(p, c, **kwargs),
                    "punctuation": lambda p, c: punctuation_usage_reward_func(p, c, **kwargs),
                    "divider": lambda p, c: divider_style_reward_func(p, c, **kwargs),
                    "topics": lambda p, c: topic_shifts_reward_func(p, c, **kwargs),
                    "narrative": lambda p, c: narrative_structure_reward_func(p, c, **kwargs),
                    "coherence": lambda p, c: semantic_coherence_reward(p, c, **kwargs),
                    "emoji_variety": lambda p, c: [emoji_variety_reward((c[i] if isinstance(c[i], str) else c[i][0])) for i in range(len(c))],
                    "language": lambda p, c: chinese_character_reward_func(p, c, **kwargs),
                    "fabrication": lambda p, c: fabrication_detection_reward_func(p, c, **kwargs),
                }
                import json
                weights = {
                    "bullet": 1.0,
                    "tone": 1.0,
                    "hashtags": 1.0,
                    "length": 1.0,
                    "emoji": 1.0,
                    "structure": 0.5,
                    "coherence": 0.5,
                    "vocabulary": 0.5,
                    "linebreaks": 0.5,
                    "punctuation": 0.5,
                    "divider": 0.3,
                    "topics": 0.5,
                    "narrative": 0.5,
                    "emoji_variety": 0.3,
                    "language": 1.0,
                    "fabrication": 2.0,
                }
                if args.weights:
                    try:
                        with open(args.weights, "r", encoding="utf-8") as f:
                            user_weights = json.load(f)
                        for k in list(weights.keys()):
                            if k in user_weights:
                                weights[k] = float(user_weights[k])
                    except Exception:
                        pass
                return aggregate_rewards(prompts, completions, funcs, weights)

            reward_list = [aggregated_reward]
        except Exception:
            # Fallback to simple sum of two robust rewards
            def aggregated_reward(prompts, completions, **kwargs):
                from training.rewards.aggregator import aggregate_rewards
                funcs = {
                    "bullet": lambda p, c: bullet_style_reward_func(p, c, **kwargs),
                    "tone": lambda p, c: tone_alignment_reward_func(p, c, **kwargs),
                    "coherence": lambda p, c: semantic_coherence_reward(p, c, **kwargs),
                }
                return aggregate_rewards(prompts, completions, funcs, {"bullet": 1.0, "tone": 1.0, "coherence": 0.5})
            reward_list = [aggregated_reward]
    else:
        reward_list = [aggregated_reward]

    from training.grpo.trainer import build_trainer

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_list,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if test_dataset else None,
    )
    # Write a minimal model card in output_dir
    try:
        import json, time
        card = {
            "run_id": args.run_id,
            "seed": 3407,
            "base_model": getattr(model, 'name_or_path', 'unknown'),
            "trainer": "GRPOTrainer",
            "report_to": "wandb",
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        out_dir = getattr(training_args, 'output_dir', 'grpo-results')
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "model_card.json"), "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2)
    except Exception:
        pass


    print("Trainer model:", trainer.model)
    print("Trainer tokenizer:", trainer.tokenizer)

    # Train the model
    print("Starting training...")
    trainer.train()

    wandb.finish()

    # Resolve output model path and idempotent skip
    output_dir = "grpo-model"
    if args.run_id:
        os.makedirs(os.path.join(args.models_dir, args.run_id), exist_ok=True)
        output_dir = os.path.join(args.models_dir, args.run_id, output_dir)
        # Manifest idempotent skip was already checked pre-training if you add it above

    # Save the merged model and tokenizer
    print(f"Saving merged model to {output_dir}...")
    # Try Unsloth merged save; fallback to standard HF save
    try:
        model.save_pretrained_merged(output_dir, tokenizer)
    except Exception:
        from transformers import PreTrainedModel, PreTrainedTokenizerBase
        try:
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        except Exception as e:
            print(f"Standard save failed: {e}")
            raise
    print(f"Successfully saved model to {output_dir}")

    # Update manifest
    try:
        if args.run_id:
            m = read_manifest(args.run_id, args.base_dir)
            sig = compute_hash([], {"stage": 26})
            update_stage(args.run_id, args.base_dir, m, "26-train-grpo", input_path=None, outputs=[output_dir], signature=sig, extra={})
    except Exception:
        pass

    return trainer

if __name__ == "__main__":
    main()