import json
import os
import random
from collections import defaultdict
from typing import Dict, Tuple

# pip install nlpaug
import nlpaug.augmenter.word as naw

import nltk

# Compatibility bootstrap: allow running this file directly or as part of the package
if __package__ is None or __package__ == "":
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from utils.logging_setup import init_pipeline_logging
from utils.seed import set_global_seed
from utils.manifest import read_manifest, compute_hash, should_skip, update_stage, discover_input
from utils.version import STAGE_VERSION

logger = init_pipeline_logging("phase2.balance", None, "05-balance")

##############################################################################
# 1. Set up the nlpaug synonym augmenter
##############################################################################

synonym_aug = naw.SynonymAug(aug_src='wordnet')

def replace_synonyms_nlpaug(text):
    """
    Use the nlpaug SynonymAug to replace synonyms in text.
    """
    if not text:
        return text
    return synonym_aug.augment(text)

##############################################################################
# 2. Helper to bucket posts by combo
##############################################################################

def bucket_by_combo(
    posts,
    structure_field='structure',
    emoji_usage_field='emoji_usage',
    max_length_field='max_length'
):
    """
    Returns a dict of (structure, emoji_usage, max_length) -> [posts].
    """
    combo_dict = defaultdict(list)
    for post in posts:
        structure_val = post.get(structure_field)
        emoji_val = post.get(emoji_usage_field)
        max_len_val = post.get(max_length_field)
        if structure_val is not None and emoji_val is not None and max_len_val is not None:
            combo_dict[(structure_val, emoji_val, max_len_val)].append(post)
    return combo_dict

##############################################################################
# 3. Two-pass balancing
##############################################################################

def two_pass_balance_dataset(
    input_file=None,
    output_file=None,
    run_id=None,
    base_dir="data/processed",
    structure_field='structure',
    emoji_usage_field='emoji_usage',
    max_length_field='max_length',
    key_message_field='post_text',
    target_cap: int | None = None,
    augment_fraction: float = 1.0,
    seed: int | None = None,
    debug: bool = False,
):
    """
    1) Stream input_file (JSONL).
    2) Pass 1: compute counts per combo; compute avg target (bounded by target_cap).
    3) Pass 2: down-sample to the target using reservoir per combo; write intermediate.
    4) Pass 3: up-sample combos below target with optional augmentation fraction; write final.
    """

    set_global_seed(seed)

    # Resolve input/output via run-id or explicit paths
    if run_id:
        if run_id == "latest":
            from utils.run_id import get_last_run_id
            rid = get_last_run_id(base_dir)
            if not rid:
                raise ValueError("No .last_run_id found; run previous stages first or provide --run-id")
            run_id = rid
        manifest = read_manifest(run_id, base_dir)
        discovered = discover_input(manifest, "03-structures")
        if discovered and os.path.exists(discovered):
            input_file = discovered
        else:
            input_file = os.path.join(base_dir, run_id, "03-structures.jsonl")
        output_file = output_file or os.path.join(base_dir, run_id, "05-balanced.jsonl")
    else:
        if not input_file or not output_file:
            raise ValueError("When --run-id is not provided, you must specify both --input and --output")

    # Manifest skip (if run_id provided)
    if run_id:
        signature = compute_hash([input_file], {"stage": 5, "augment_fraction": augment_fraction, "cap": target_cap,
                                               "fields": [structure_field, emoji_usage_field, max_length_field], "stage_version": STAGE_VERSION})
        manifest = read_manifest(run_id, base_dir)
        if should_skip(manifest, "05-balance", signature, [output_file]):
            logger.info(f"Skipping 05-balance; up-to-date at {output_file}")
            return {"input": input_file, "output": output_file, "final_size": None, "combos": None}

    # Pass 1: counts per combo
    combo_counts = defaultdict(int)
    total = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                post = json.loads(line)
            except Exception:
                continue
            key = (post.get(structure_field), post.get(emoji_usage_field), post.get(max_length_field))
            if None in key:
                continue
            combo_counts[key] += 1
            total += 1

    if not combo_counts:
        logger.warning("No valid combos found in input!")
        return

    combos_we_have = len(combo_counts)
    avg_target = int(total / combos_we_have) if combos_we_have else 0
    if target_cap is not None:
        avg_target = min(avg_target, int(target_cap))
    avg_target = max(1, avg_target)

    if debug:
        logger.info(f"[Pass 1] combos={combos_we_have} total={total} avg_target={avg_target}")

    # Pass 2: down-sample via per-combo reservoir
    tmp_down_path = os.path.join(base_dir, run_id or "adhoc", "05a-downsampled.jsonl")
    os.makedirs(os.path.dirname(tmp_down_path), exist_ok=True)

    reservoirs: dict[tuple, list] = defaultdict(list)
    seen_per_combo: dict[tuple, int] = defaultdict(int)

    with open(input_file, 'r', encoding='utf-8') as f, open(tmp_down_path, 'w', encoding='utf-8') as fout:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                post = json.loads(line)
            except Exception:
                continue
            key = (post.get(structure_field), post.get(emoji_usage_field), post.get(max_length_field))
            if None in key:
                continue
            seen_per_combo[key] += 1
            k = avg_target
            bucket = reservoirs[key]
            if len(bucket) < k:
                bucket.append(post)
            else:
                j = random.randint(0, seen_per_combo[key] - 1)
                if j < k:
                    bucket[j] = post

        # write reservoirs
        for bucket in reservoirs.values():
            for p in bucket:
                fout.write(json.dumps(p, ensure_ascii=False) + "\n")

    if debug:
        logger.info(f"[Pass 2] Wrote down-sampled dataset to {tmp_down_path}")

    # Pass 3: up-sample to approx avg_target
    upsampled_combo_dict = defaultdict(list)
    with open(tmp_down_path, 'r', encoding='utf-8') as f:
        for line in f:
            post = json.loads(line)
            key = (post.get(structure_field), post.get(emoji_usage_field), post.get(max_length_field))
            upsampled_combo_dict[key].append(post)

    final_posts = []
    for key, posts_list in upsampled_combo_dict.items():
        count = len(posts_list)
        target = avg_target
        if count >= target:
            final_posts.extend(posts_list)
            continue
        # upsample with optional augmentation fraction
        while len(posts_list) < target:
            orig_post = random.choice(posts_list)
            new_post = dict(orig_post)
            if random.random() < augment_fraction:
                key_msg = new_post.get(key_message_field, "")
                new_post[key_message_field] = replace_synonyms_nlpaug(key_msg)
            posts_list.append(new_post)
        final_posts.extend(posts_list)

    # Write final
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as fout:
        for post in final_posts:
            fout.write(json.dumps(post, ensure_ascii=False) + "\n")

    if debug:
        logger.info(f"[Final] Dataset size: {len(final_posts)} at {output_file}")

    # Manifest update
    if run_id:
        signature = compute_hash([input_file], {"stage": 5, "avg_target": avg_target, "augment_fraction": augment_fraction, "stage_version": STAGE_VERSION})
        manifest = read_manifest(run_id, base_dir)
        update_stage(run_id, base_dir, manifest, "05-balance", input_file, [output_file], signature,
                     extra={"final_size": len(final_posts), "combos": len(upsampled_combo_dict)})

    return {
        "input": input_file,
        "output": output_file,
        "final_size": len(final_posts),
        "combos": len(upsampled_combo_dict),
    }



##############################################################################
# 4. Run it if this script is called directly
##############################################################################
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Balance dataset across structure/emoji/length combos")
    parser.add_argument("--run-id", dest="run_id", default=None, help="Use 'latest' to pick up most recent run")
    parser.add_argument("--base-dir", dest="base_dir", default="data/processed")
    parser.add_argument("--input", dest="input_file", default=None, help="Only used when --run-id is not provided")
    parser.add_argument("--output", dest="output_file", default=None, help="Only used when --run-id is not provided")
    parser.add_argument("--debug", dest="debug", action="store_true", default=False)
    args = parser.parse_args()

    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

    stats = two_pass_balance_dataset(
        input_file=args.input_file,
        output_file=args.output_file,
        run_id=args.run_id,
        base_dir=args.base_dir,
        target_cap=1000,
        augment_fraction=1.0,
        debug=args.debug,
    )
    print("Balance stats:", stats)
