import json
import random
import csv
import re
import os
import argparse
from datetime import datetime
from collections import defaultdict

# Compatibility bootstrap: allow running this file directly or as part of the package
if __package__ is None or __package__ == "":
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from pipe.utils.logging_setup import init_logging, init_pipeline_logging
from pipe.utils.manifest import read_manifest, write_manifest, compute_hash, should_skip, update_stage, discover_input
from pipe.utils.validation import validate_csv
from pipe.utils.contracts import write_contract
from pipe.utils.reports import write_summary
import pandas as pd

# For synonym augmentation (optional; will import lazily)
naw = None
try:
    import nlpaug.augmenter.word as _naw  # type: ignore
    naw = _naw
except Exception:
    naw = None

##############################################################################
# SYNONYM AUGMENTATION SETUP
##############################################################################
# This uses WordNet behind the scenes. You can tweak aug_p, aug_max, etc.
synonym_aug = None
if naw is not None:
    synonym_aug = naw.SynonymAug(
        aug_src='wordnet',
        aug_p=0.3,    # Proportion of words to attempt to replace
        aug_min=1,    # Minimum # of words replaced
        aug_max=3,    # Maximum # of words replaced
        verbose=0
    )

def replace_synonyms(text):
    """Use nlpaug to replace synonyms in text. Returns augmented text."""
    if not text:
        return text
    if synonym_aug is None:
        return text
    return synonym_aug.augment(text)

##############################################################################
# HELPER FUNCTIONS FOR ANALYZING PROMPTS
##############################################################################

def extract_prompt_type(prompt):
    """Identify the prompt type from the prompt text."""
    if not prompt:
        return "unknown"

    # Check for specific phrases that indicate prompt type
    if "identify its primary topic" in prompt.lower():
        return "topic"
    elif "extract the core opinion" in prompt.lower():
        return "opinion"
    elif "identify up to three primary tones" in prompt.lower():
        return "tone"
    elif "classify the structural format" in prompt.lower():
        return "structure"
    elif "# request" in prompt.lower() and "create a linkedin post" in prompt.lower():
        return "post-generation"
    else:
        return "other"

def extract_max_length(prompt):
    """Extract the max_length from a post generation prompt."""
    if not prompt:
        return "unknown"

    # Look for the pattern "Suggested Post Length: X"
    match = re.search(r"suggested post length[:\s]*(.+?)(?:\n|\*\*|$)", prompt.lower())
    if match:
        length_text = match.group(1).strip()

        # Check for character length ranges
        if "750" in length_text and "1,500" in length_text:
            return "medium"  # Between 750 and 1,500 characters
        elif "1,500" in length_text and "3,000" in length_text:
            return "long"    # Between 1,500 and 3,000 characters
        elif "750" in length_text:
            return "short"   # Up to 750 characters

        # Convert text lengths to standardized categories
        if any(word in length_text for word in ["short", "brief", "concise"]):
            return "short"
        elif any(word in length_text for word in ["medium", "moderate", "average"]):
            return "medium"
        elif any(word in length_text for word in ["long", "detailed", "comprehensive"]):
            return "long"
        else:
            return length_text

    return "unknown"

def extract_emoji_usage(prompt):
    """Extract emoji usage from a post generation prompt."""
    if not prompt:
        return "unknown"

    # Look for the pattern "Emoji Usage: X"
    match = re.search(r"emoji usage[:\s]*(.+?)(?:\n|\*\*|$)", prompt.lower())
    if match:
        emoji_text = match.group(1).strip()

        # Map to standardized categories
        if any(word in emoji_text for word in ["none", "no", "zero"]):
            return "none"
        elif any(word in emoji_text for word in ["very low", "minimal", "rarely"]):
            return "very low"
        elif any(word in emoji_text for word in ["low", "occasional", "sparse"]):
            return "low"
        elif any(word in emoji_text for word in ["medium", "moderate", "average"]):
            return "medium"
        elif any(word in emoji_text for word in ["high", "frequent", "many"]):
            return "high"
        elif any(word in emoji_text for word in ["extreme", "heavy", "abundant"]):
            return "extreme"
        else:
            return emoji_text

    return "unknown"

def extract_structure_from_chosen(chosen, prompt_type):
    """Extract structure category from chosen response for structure prompts."""
    if prompt_type != "structure" or not chosen:
        return None

    # The structure classification prompt asks for exactly one category name
    structure_categories = [
        "instructional", "inspirational", "controversial",
        "insightful", "comparative", "reflective", "announcement"
    ]

    chosen_lower = chosen.lower().strip()

    # Direct match with category names
    for category in structure_categories:
        if chosen_lower == category:
            return category

    # Try to find the best match if no direct match
    for category in structure_categories:
        if category in chosen_lower:
            return category

    return "other"

def extract_tones_from_chosen(chosen, prompt_type):
    """Extract primary tone from chosen response for tone prompts."""
    if prompt_type != "tone" or not chosen:
        return None

    # The tone prompt asks for a comma-separated list of up to 3 tones
    # We'll use the first tone as the key for balancing
    tones = [t.strip().lower() for t in chosen.split(',')]
    return tones[0] if tones else "unknown"

def extract_info_from_post_gen_prompt(prompt):
    """
    Extract structure from a post generation prompt.
    The format follows: "Create a LinkedIn post that **shares a step-by-step guide**"
    """
    if not prompt:
        return None

    structure_mapping = {
        "shares a step-by-step guide": "instructional",
        "reflects on an experience": "reflective",
        "inspires and motivates": "inspirational",
        "challenges popular opinions": "controversial",
        "offers keen observations": "insightful",
        "compares two or more items": "comparative",
        "announces something new": "announcement"
    }

    # Look for the pattern
    for description, structure in structure_mapping.items():
        if description in prompt.lower():
            return structure

    return None

##############################################################################
# MAIN PROCESS FUNCTION
##############################################################################

def process_csv(
    input_file=None,
    balanced_file='balanced-dataset.csv',
    sft_file='sft.csv',
    dpo_file='dpo.csv',
    run_id=None,
    base_dir="data/processed",
    seed=None,
    sft_percentage=0.80,    # 80%
    dpo_percentage=0.20,    # 20%
    lower_ratio=0.95,
    upper_ratio=1.05,
    prefer_downsampling=True,  # Prioritize downsampling
    min_entries_per_combo=1,   # Minimum entries to keep per combo
    debug=True,
    disable_augmentation=False,
):
    logger = init_pipeline_logging("phase2.split", run_id, "23-split")

    # Resolve input via manifest helper when run_id is provided unless explicit input is passed
    if run_id:
        from pipe.utils.io import resolve_input_path
        input_file = resolve_input_path(run_id=run_id, base_dir=base_dir, explicit_input=input_file, prior_stages=["22-dataset"])  # type: ignore
        logger.info(f"Resolved input: {input_file}")
    else:
        if not input_file:
            raise ValueError("When --run-id is not provided, you must specify --input")

    # Early skip if up-to-date (before reading input)
    std_balanced_path = std_sft_path = std_dpo_path = None
    std_balanced_parquet = std_sft_parquet = std_dpo_parquet = None
    if run_id:
        std_dir = os.path.join(base_dir, run_id)
        os.makedirs(std_dir, exist_ok=True)
        std_balanced_path = os.path.join(std_dir, '23-balanced-dataset.csv')
        std_sft_path = os.path.join(std_dir, '23-sft.csv')
        std_dpo_path = os.path.join(std_dir, '23-dpo.csv')
        std_balanced_parquet = os.path.join(std_dir, '23-balanced-dataset.parquet')
        std_sft_parquet = os.path.join(std_dir, '23-sft.parquet')
        std_dpo_parquet = os.path.join(std_dir, '23-dpo.parquet')
        from utils.version import STAGE_VERSION
        signature = compute_hash([input_file], config={
            "stage": 23,
            "sft_percentage": sft_percentage,
            "dpo_percentage": dpo_percentage,
            "lower_ratio": lower_ratio,
            "upper_ratio": upper_ratio,
            "prefer_downsampling": prefer_downsampling,
            "min_entries_per_combo": min_entries_per_combo,
            "stage_version": STAGE_VERSION,
        })
        manifest = read_manifest(run_id, base_dir)
        if should_skip(manifest, "23-split", signature, [std_balanced_path, std_sft_path, std_dpo_path]):
            logger.info(f"Skipping 23-split; up-to-date at {std_dir}")
            return

    # Seeding for determinism when shuffling/upsampling
    from pipe.utils.seed import set_global_seed
    set_global_seed(seed)
    """
    1) Loads all entries from `input_file` (CSV format).
    2) Extracts prompt type, structure, tone from prompt/chosen fields.
    3) Balances them prioritizing downsampling.
    4) Writes the final balanced set to `balanced_file`.
    5) Splits that balanced set into SFT and DPO sets.
    """

    ##################################################
    # 1) Load all entries from CSV into memory
    ##################################################
    all_entries = []

    with open(input_file, 'r', encoding='utf-8', newline='') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            all_entries.append(row)

    if debug:
        logger.info(f"Loaded {len(all_entries)} entries from {input_file}")

    ##################################################
    # 2) Extract metadata from prompt/chosen fields
    ##################################################
    for entry in all_entries:
        # Determine prompt type
        prompt = entry.get('prompt', '')
        entry['prompt_type'] = extract_prompt_type(prompt)

        # Extract structure based on prompt type
        if entry['prompt_type'] == 'structure':
            entry['structure'] = extract_structure_from_chosen(entry.get('chosen', ''), 'structure')
        elif entry['prompt_type'] == 'post-generation':
            entry['structure'] = extract_info_from_post_gen_prompt(prompt)
        else:
            entry['structure'] = 'N/A'

        # Extract tone if it's a tone prompt
        if entry['prompt_type'] == 'tone':
            entry['tone'] = extract_tones_from_chosen(entry.get('chosen', ''), 'tone')
        else:
            entry['tone'] = 'N/A'

    ##################################################
    # 3) Extract metadata for post-generation prompts
    ##################################################
    for entry in all_entries:
        if entry.get('prompt_type') == 'post-generation':
            prompt = entry.get('prompt', '')
            entry['max_length'] = extract_max_length(prompt)
            entry['emoji_usage'] = extract_emoji_usage(prompt)
        else:
            entry['max_length'] = None
            entry['emoji_usage'] = None

    ##################################################
    # 4) Create balanced buckets
    ##################################################
    # For post-generation prompts, group by (structure, max_length, emoji_usage)
    # For other prompts, group by prompt_type
    combo_dict = defaultdict(list)

    for entry in all_entries:
        prompt_type = entry.get('prompt_type', 'unknown')

        if prompt_type == 'post-generation':
            # Create a combo key for post-generation, including structure, length and emoji usage
            structure = entry.get('structure', 'unknown')
            max_length = entry.get('max_length', 'unknown')
            emoji_usage = entry.get('emoji_usage', 'unknown')
            combo = ('post-generation', structure, max_length, emoji_usage)
        else:
            # For other prompt types, just use the prompt_type
            combo = (prompt_type, 'N/A', 'N/A', 'N/A')

        combo_dict[combo].append(entry)

    total_entries = len(all_entries)
    combos_count = len(combo_dict)
    if combos_count == 0:
        logger.warning("No valid combinations found — no data to process.")
        return

    average_count = total_entries / combos_count
    lower_bound = lower_ratio * average_count
    upper_bound = upper_ratio * average_count

    if debug:
        logger.info(f"Number of distinct combinations: {combos_count}")
        logger.info(f"Total entries: {total_entries}")
        logger.info(f"Average count per combination: {average_count:.2f}")
        logger.info(f"±5% bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")

    ##################################################
    # 5) Balancing each combination
    ##################################################
    balanced_combo_dict = {}

    if prefer_downsampling:
        # Find the minimum count based on the lower bound
        min_count = max(min_entries_per_combo, int(lower_bound))

        # Down-sample everything to this count
        for combo, entries_list in combo_dict.items():
            count = len(entries_list)

            if count > min_count:
                # Down-sample to minimum count
                balanced_combo_dict[combo] = random.sample(entries_list, min_count)
                if debug:
                    logger.info(f"{combo} => {count} down-sampled to {min_count}.")
            elif count == 0:
                # Skip empty combos
                balanced_combo_dict[combo] = []
                if debug:
                    logger.info(f"{combo} => 0 items, skipping.")
            else:
                # Keep all entries if fewer than min_count
                balanced_combo_dict[combo] = entries_list

                # Only upsample if absolutely necessary
                if count < min_entries_per_combo:
                    # Up-sample with synonyms to reach minimum
                    balanced_list = list(entries_list)
                    while len(balanced_list) < min_entries_per_combo:
                        orig_entry = random.choice(entries_list)
                        new_entry = dict(orig_entry)

                        # Augment chosen text with synonyms
                        if (not disable_augmentation) and new_entry.get('chosen'):
                            new_entry['chosen'] = replace_synonyms(new_entry['chosen'])

                        balanced_list.append(new_entry)

                    balanced_combo_dict[combo] = balanced_list
                    if debug:
                        logger.info(f"{combo} => {count} minimally up-sampled to {len(balanced_list)}.")
                else:
                    if debug:
                        print(f"{combo} => keeping original ({count}).")
    else:
        # Original balancing strategy
        for combo, entries_list in combo_dict.items():
            count = len(entries_list)
            if lower_bound <= count <= upper_bound:
                # Already in the acceptable range
                balanced_combo_dict[combo] = entries_list
                if debug:
                    logger.info(f"{combo} => within range ({count}).")
            elif count > upper_bound:
                # Down-sample
                new_count = int(upper_bound)
                if new_count < 1:
                    new_count = 1
                balanced_combo_dict[combo] = random.sample(entries_list, new_count)
                if debug:
                    logger.info(f"{combo} => {count} down-sampled to {new_count}.")
            else:
                # Under-sampled => upsample with synonyms
                new_count = int(lower_bound)
                if new_count < 1:
                    new_count = 1

                if count == 0:
                    # If 0 entries, we can't augment
                    balanced_combo_dict[combo] = []
                    if debug:
                        logger.info(f"{combo} => 0 items, cannot up-sample from nothing.")
                else:
                    # Start with original entries
                    balanced_list = list(entries_list)
                    while len(balanced_list) < new_count:
                        orig_entry = random.choice(entries_list)
                        # Clone
                        new_entry = dict(orig_entry)
                        # Augment chosen text
                        if (not disable_augmentation) and new_entry.get('chosen'):
                            new_entry['chosen'] = replace_synonyms(new_entry['chosen'])
                        balanced_list.append(new_entry)

                    balanced_combo_dict[combo] = balanced_list
                    if debug:
                        logger.info(f"{combo} => {count} up-sampled to {len(balanced_list)}.")

    # Combine all balanced entries
    balanced_entries = []
    for combo_entries in balanced_combo_dict.values():
        balanced_entries.extend(combo_entries)

    # Shuffle if you like
    random.shuffle(balanced_entries)

    if debug:
        logger.info(f"Balanced total: {len(balanced_entries)} entries")

    ##################################################
    # 5) Write the final balanced set to a CSV
    ##################################################

    # Open std outputs only after skip decision
    std_dir = None
    std_balanced = None
    std_sft = None
    std_dpo = None
    if run_id:
        std_dir = os.path.join(base_dir, run_id)
        os.makedirs(std_dir, exist_ok=True)
        std_balanced = open(std_balanced_path, 'w', encoding='utf-8', newline='')
        std_sft = open(std_sft_path, 'w', encoding='utf-8', newline='')
        std_dpo = open(std_dpo_path, 'w', encoding='utf-8', newline='')

    with open(balanced_file, 'w', encoding='utf-8', newline='') as outf:
        # Use the same fieldnames as the input CSV - typically system, prompt, chosen, rejected
        fieldnames = list(all_entries[0].keys()) if all_entries else ['system', 'prompt', 'chosen', 'rejected']

        # Remove any metadata fields we added
        for field in ['prompt_type', 'structure', 'tone', 'max_length', 'emoji_usage']:
            if field in fieldnames and field not in ['system', 'prompt', 'chosen', 'rejected']:
                fieldnames.remove(field)

        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()

        # Prepare standardized writer once (if applicable)
        std_bal_writer = None
        if std_balanced:
            std_bal_writer = csv.DictWriter(std_balanced, fieldnames=fieldnames)
            if std_balanced.tell() == 0:
                std_bal_writer.writeheader()

        for entry in balanced_entries:
            # Only write the original fields, not our added metadata
            row_to_write = {k: v for k, v in entry.items() if k in fieldnames}
            writer.writerow(row_to_write)
            if std_bal_writer:
                std_bal_writer.writerow(row_to_write)

    ##################################################
    # 7) Split into SFT and DPO sets
    ##################################################
    # For post-generation prompts, group by structure
    # For other prompts, group by prompt_type
    balanced_by_group = defaultdict(list)
    for entry in balanced_entries:
        prompt_type = entry.get('prompt_type', 'unknown')

        if prompt_type == 'post-generation' and entry.get('structure'):
            # Use structure as the grouping key for post-generation
            group_key = f"post-{entry['structure']}"
        else:
            # Use prompt_type for other prompts
            group_key = prompt_type

        balanced_by_group[group_key].append(entry)

    sft_entries = []  # For SFT
    dpo_entries = []  # For DPO

    for group_key, entries_list in balanced_by_group.items():
        random.shuffle(entries_list)
        total_for_group = len(entries_list)

        # Take sft_percentage for SFT
        sft_count = int(total_for_group * sft_percentage)
        # The remainder goes to DPO
        dpo_count = total_for_group - sft_count

        sft_slice = entries_list[:sft_count]
        dpo_slice = entries_list[sft_count:sft_count + dpo_count]

        sft_entries.extend(sft_slice)
        dpo_entries.extend(dpo_slice)

    # Write them out
    with open(sft_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['system', 'prompt', 'chosen', 'rejected']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Prepare standardized writer if needed
        if std_sft and std_sft.tell() == 0:
            std_sft_writer = csv.DictWriter(std_sft, fieldnames=fieldnames)
            std_sft_writer.writeheader()
        for entry in sft_entries:
            # Only write the standard fields
            row_to_write = {k: v for k, v in entry.items() if k in fieldnames}
            writer.writerow(row_to_write)
            if std_sft:
                std_sft_writer.writerow(row_to_write)

    # Validate SFT and DPO
    ok_sft = False
    ok_dpo = False
    try:
        ok_sft = validate_csv(sft_file)
        logger.info(f"Validated SFT CSV {sft_file}: {ok_sft}")
    except Exception:
        logger.warning("Validation failed for SFT")

    # Write Parquet mirrors and contracts for standardized SFT and DPO
    try:
        if run_id and std_sft_path and std_sft_parquet:
            df_sft = pd.read_csv(std_sft_path)
            df_sft.to_parquet(std_sft_parquet, index=False)
            write_contract(std_sft_path, schema_version="v1", counts={"rows": len(df_sft)})
            write_contract(std_sft_parquet, schema_version="v1", counts={"rows": len(df_sft)})
            logger.info(f"Standardized SFT Parquet written to: {std_sft_parquet}")
    except Exception as e:
        logger.warning(f"Failed to write SFT Parquet/contract: {e}")

    with open(dpo_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['system', 'prompt', 'chosen', 'rejected']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Prepare standardized writer if needed
        if std_dpo and std_dpo.tell() == 0:
            std_dpo_writer = csv.DictWriter(std_dpo, fieldnames=fieldnames)
            std_dpo_writer.writeheader()
        for entry in dpo_entries:
            # Only write the standard fields
            row_to_write = {k: v for k, v in entry.items() if k in fieldnames}
            writer.writerow(row_to_write)
            if std_dpo:
                std_dpo_writer.writerow(row_to_write)

    try:
        ok_dpo = validate_csv(dpo_file)
        logger.info(f"Validated DPO CSV {dpo_file}: {ok_dpo}")
    except Exception:
        logger.warning("Validation failed for DPO")

    try:
        if run_id and std_dpo_path and std_dpo_parquet:
            df_dpo = pd.read_csv(std_dpo_path)
            df_dpo.to_parquet(std_dpo_parquet, index=False)
            write_contract(std_dpo_path, schema_version="v1", counts={"rows": len(df_dpo)})
            write_contract(std_dpo_parquet, schema_version="v1", counts={"rows": len(df_dpo)})
            logger.info(f"Standardized DPO Parquet written to: {std_dpo_parquet}")
    except Exception as e:
        logger.warning(f"Failed to write DPO Parquet/contract: {e}")

    if std_balanced:
        std_balanced.close()
    if std_sft:
        std_sft.close()
    if std_dpo:
        std_dpo.close()

    if run_id and std_dir:
        # Re-validate standardized CSVs with required columns
        from pipe.utils.validation import validate_csv
        ok_sft = validate_csv(std_sft_path)
        ok_dpo = validate_csv(std_dpo_path)

        # Gate manifest update on both standardized splits validating
        if not (ok_sft and ok_dpo):
            logger.error("Standardized split CSVs failed validation; skipping manifest update for 23-split")
            return
        # Parquet and contracts for balanced as well
        try:
            if std_balanced_path and std_balanced_parquet:
                df_bal = pd.read_csv(std_balanced_path)
                df_bal.to_parquet(std_balanced_parquet, index=False)
                write_contract(std_balanced_path, schema_version="v1", counts={"rows": len(df_bal)})
                write_contract(std_balanced_parquet, schema_version="v1", counts={"rows": len(df_bal)})
                logger.info(f"Standardized balanced Parquet written to: {std_balanced_parquet}")
        except Exception as e:
            logger.warning(f"Failed to write balanced Parquet/contract: {e}")

        manifest = read_manifest(run_id, base_dir)
        from utils.version import STAGE_VERSION
        signature = compute_hash([input_file], config={
            "stage": 23,
            "sft_percentage": sft_percentage,
            "dpo_percentage": dpo_percentage,
            "lower_ratio": lower_ratio,
            "upper_ratio": upper_ratio,
            "prefer_downsampling": prefer_downsampling,
            "min_entries_per_combo": min_entries_per_combo,
            "stage_version": STAGE_VERSION,
        })
        update_stage(
            run_id,
            base_dir,
            manifest,
            stage_name="23-split",
            input_path=input_file,
            outputs=[p for p in [std_balanced_path, std_sft_path, std_dpo_path, std_balanced_parquet, std_sft_parquet, std_dpo_parquet] if p],
            signature=signature,
            extra={},
        )
        # Write run summary (rows per artifact)
        try:
            rows = {}
            if std_balanced_path and os.path.exists(std_balanced_path):
                rows["balanced_rows"] = len(pd.read_csv(std_balanced_path))
            if std_sft_path and os.path.exists(std_sft_path):
                rows["sft_rows"] = len(pd.read_csv(std_sft_path))
            if std_dpo_path and os.path.exists(std_dpo_path):
                rows["dpo_rows"] = len(pd.read_csv(std_dpo_path))
            write_summary(run_id, "23-split", rows)
        except Exception:
            pass
        logger.info(f"Standardized outputs written to: {std_dir}")

    if debug:
        # Print a breakdown of the balanced dataset
        post_gen_structures = defaultdict(int)
        post_gen_lengths = defaultdict(int)
        post_gen_emoji = defaultdict(int)

        for entry in balanced_entries:
            if entry.get('prompt_type') == 'post-generation':
                structure = entry.get('structure', 'unknown')
                max_length = entry.get('max_length', 'unknown')
                emoji = entry.get('emoji_usage', 'unknown')

                post_gen_structures[structure] += 1
                post_gen_lengths[max_length] += 1
                post_gen_emoji[emoji] += 1

        logger.info("\nPost-Generation Breakdown:")
        logger.info(f"Structures: {dict(post_gen_structures)}")
        logger.info(f"Lengths: {dict(post_gen_lengths)}")
        logger.info(f"Emoji Usage: {dict(post_gen_emoji)}")

        logger.info(f"\n[DONE] Balanced dataset => {len(balanced_entries)} entries.")
        logger.info(f"SFT file '{sft_file}' => {len(sft_entries)} examples.")
        logger.info(f"DPO file '{dpo_file}' => {len(dpo_entries)} examples.")


##############################################################################
# Run if called directly
##############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Balance dataset and split into SFT/DPO (run-id manifest mode only)")
    parser.add_argument("--input", dest="input_path", default=None)
    parser.add_argument("--sft-percentage", dest="sft_percentage", type=float, default=0.80)
    parser.add_argument("--dpo-percentage", dest="dpo_percentage", type=float, default=0.20)
    parser.add_argument("--lower-ratio", dest="lower_ratio", type=float, default=0.95)
    parser.add_argument("--upper-ratio", dest="upper_ratio", type=float, default=1.05)
    parser.add_argument("--prefer-downsampling", dest="prefer_downsampling", action="store_true", default=True)
    parser.add_argument("--min-entries-per-combo", dest="min_entries_per_combo", type=int, default=1)
    parser.add_argument("--debug", dest="debug", action="store_true", default=True)
    parser.add_argument("--disable-augmentation", dest="disable_augmentation", action="store_true", default=False)
    from pipe.utils.cli import add_standard_args, resolve_common_args
    add_standard_args(parser, include_seed=True)
    args = parser.parse_args()

    # Strict run-id manifest mode only
    args = resolve_common_args(args, require_input_when_no_run_id=True)

    logger = init_pipeline_logging("phase2.split", args.run_id, "23-split")
    logger.info(f"Processing {args.input_path or '[manifest]'} -> [standardized] under {args.base_dir}/{args.run_id}")

    # Use standardized IO only under run-id
    from pipe.utils.artifacts import ArtifactNames
    from pipe.utils.io import resolve_io

    input_path, _, _ = resolve_io(stage="23-split", run_id=args.run_id, base_dir=args.base_dir, explicit_in=args.input_path, prior_stage="22-dataset", std_name=None)

    process_csv(
        input_file=input_path,
        balanced_file=ArtifactNames.STAGE23_BALANCED,
        sft_file=ArtifactNames.STAGE23_SFT,
        dpo_file=ArtifactNames.STAGE23_DPO,
        sft_percentage=args.sft_percentage,
        dpo_percentage=args.dpo_percentage,
        lower_ratio=args.lower_ratio,
        upper_ratio=args.upper_ratio,
        prefer_downsampling=args.prefer_downsampling,
        min_entries_per_combo=args.min_entries_per_combo,
        debug=args.debug,
        run_id=args.run_id,
        base_dir=args.base_dir,
        seed=args.seed,
    )