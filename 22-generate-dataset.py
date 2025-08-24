import json
import csv
import os
import argparse
from datetime import datetime
from tqdm import tqdm  # Optional: for progress bar, install with pip if needed
# Compatibility bootstrap: allow running this file directly or as part of the package
if __package__ is None or __package__ == "":
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from pipe.utils.logging_setup import init_pipeline_logging
from pipe.utils.manifest import read_manifest, write_manifest, compute_hash, should_skip, update_stage, discover_input
from pipe.utils.validation import validate_csv
from pipe.utils.contracts import write_contract
from pipe.utils.reports import write_summary
from pipe.utils.cli import add_standard_args, resolve_common_args
import pandas as pd

def create_topic_prompt(post_text, post):
    """Create a markdown-formatted prompt for topic classification"""
    return f"""## Prompt
Analyze the following social media post and identify its primary topic **in a single phrase or sentence**.

### Content to Analyze
```
{post_text}
```

### Writing Constraints
- **Response Type**: Topic classification
- **Format**: Single phrase or short sentence
- **Tone**: Analytical, objective
- **Length**: Keep your response under 10 words
"""

def create_opinion_prompt(post_text, post):
    """Create a markdown-formatted prompt for opinion extraction"""
    return f"""# Request
Extract the core opinion from this social media post and present it in first person.

## Content to Analyze
```
{post_text}
```

## Writing Constraints
- **Response Type**: Opinion statement
- **Format**: First-person perspective
- **Tone**: Match the author's voice
- **Length**: One or two sentences
"""

def create_tone_prompt(post_text, post):
    """Create a markdown-formatted prompt for tone analysis"""
    return f"""# Request
Analyze this social media post and identify up to three primary tones from the provided list.

## Content to Analyze
```
{post_text}
```

## Available Tones
Adventurous, Artistic, Assertive, Authoritative, Bold, Bright, Calm, Capable, Caring, Casual, Charming, Cheerful, Clever, Cocky, Colorful, Comfortable, Conversational, Creative, Daring, Delightful, Detailed, Dramatic, Dry, Eccentric, Elegant, Endearing, Energetic, Engaging, Exciting, Fabulous, Fancy, Fierce, Formal, Friendly, Fun, Futuristic, Glamorous, Honorable, Industrial, Informative, Inspiring, Intense, Inviting, Lively, Natural, No-nonsense, Persuasive, Playful, Powerful, Professional, Quirky, Rebellious, Reliable, Sarcastic, Savvy, Scholarly, Secure, Serious, Silly, Sleek, Smart, Soothing, Sophisticated, Stable, Stimulating, Strong, Swanky, Tasteful, Thoughtful, Trustworthy, Unconventional, Unique, Upbeat, Versatile, Whimsical, Witty.

## Writing Constraints
- **Response Type**: Tone classification
- **Format**: Comma-separated list
- **Maximum Selections**: Three tones
- **Prohibited**: No explanations or additional text
"""

def create_structure_prompt(post_text, post):
    """Create a markdown-formatted prompt for structure classification"""
    return f"""# Request
Classify the structural format of this social media post using the provided categories.

## Content to Analyze
```
{post_text}
```

## Structure Categories
- **Instructional**: Posts offering practical, step-by-step advice
- **Inspirational**: Posts that share success stories or words of encouragement
- **Controversial**: Posts that challenge conventional wisdom or popular opinion
- **Insightful**: Posts sharing thoughts on current events or industry changes
- **Comparative**: Posts that compare two or more things
- **Reflective**: Posts reflecting on past experiences
- **Announcement**: Posts that grow excitement for something new

## Writing Constraints
- **Response Type**: Structure classification
- **Format**: Single word (category name only)
- **Required**: Choose exactly one category
"""

def load_and_process_data(input_file, output_file='dpo.csv', run_id=None, base_dir="data/processed"):
    logger = init_pipeline_logging("phase2.dataset", run_id, "22-generate-dataset")

    # Resolve IO centrally
    from pipe.utils.io import resolve_io
    from pipe.utils.artifacts import ArtifactNames
    input_file, std_output_path, run_id = resolve_io(stage="22-dataset", run_id=run_id, base_dir=base_dir, explicit_in=input_file, prior_stage="18-prompts", std_name=ArtifactNames.STAGE22_DATASET)
    std_parquet_path = os.path.join(base_dir, run_id, ArtifactNames.STAGE22_DATASET.replace('.csv', '.parquet'))
    logger.info(f"Resolved input: {input_file}; std_out: {std_output_path}")

    # Early skip check
    std_csvfile = None
    from utils.version import STAGE_VERSION
    signature = compute_hash([input_file], config={"stage": 22, "stage_version": STAGE_VERSION})
    manifest = read_manifest(run_id, base_dir)
    if should_skip(manifest, "22-dataset", signature, [std_output_path, std_parquet_path]):
        logger.info(f"Skipping 22-dataset; up-to-date at {std_output_path}")
        return

    # Standardized system message for all prompts
    system = "Below is an instruction from the user that describes a task. It is crucial to avoid making up any facts or mentioning entities that are not explicitly stated in the instruction. Strictly adhere to the information provided and do not introduce any external or irrelevant details."

    # Prepare standardized writer only
    std_csvfile = open(std_output_path, 'w', newline='', encoding='utf-8')
    std_writer = csv.writer(std_csvfile)
    std_writer.writerow(['system', 'prompt', 'chosen', 'rejected'])

    # Legacy csvwriter removed: write only standardized CSV
    csvwriter = std_writer

    # Single-pass streaming; tqdm without fixed total
    processed = 0
    created = 0
    with open(input_file, 'r', encoding='utf-8') as file:
            for line in tqdm(file, desc="Processing posts"):
                processed += 1
                try:
                    post = json.loads(line)
                    if not post.get("post_text"):
                        continue
                    post_text = post["post_text"]

                    # Topic classification
                    if post.get('topic') is not None:
                        prompt = create_topic_prompt(post_text, post)
                        chosen = post["topic"]
                        csvwriter.writerow([system, prompt, chosen, None])
                        if std_output_path:
                            std_writer.writerow([system, prompt, chosen, None])
                        created += 1

                    # Opinion extraction
                    if post.get('opinion') is not None:
                        prompt = create_opinion_prompt(post_text, post)
                        chosen = post["opinion"]
                        csvwriter.writerow([system, prompt, chosen, None])
                        if std_output_path:
                            std_writer.writerow([system, prompt, chosen, None])
                        created += 1

                    # Tone analysis
                    if post.get('tone') is not None:
                        prompt = create_tone_prompt(post_text, post)
                        chosen = post["tone"]
                        csvwriter.writerow([system, prompt, chosen, None])
                        if std_output_path:
                            std_writer.writerow([system, prompt, chosen, None])
                        created += 1

                    # Structure classification
                    if post.get('structure') is not None:
                        prompt = create_structure_prompt(post_text, post)
                        chosen = post["structure"]
                        csvwriter.writerow([system, prompt, chosen, None])
                        if std_output_path:
                            std_writer.writerow([system, prompt, chosen, None])
                        created += 1

                    # Post generation prompt
                    if post.get('prompt') is not None:
                        prompt = post.get('prompt')
                        chosen = post_text
                        csvwriter.writerow([system, prompt, chosen, None])
                        if std_output_path:
                            std_writer.writerow([system, prompt, chosen, None])
                        created += 1
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON line; skipping")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing post: {e}")
                    continue

    logger.info(f"Created {created} training examples from {processed} lines")

    # Validate outputs (gate manifest update on standardized validation)
    ok_std = False
    try:
        ok_legacy = validate_csv(output_file)
        logger.info(f"Validated legacy CSV {output_file}: {ok_legacy}")
        if std_output_path:
            # Schema: system, prompt, chosen, rejected
            from schemas import Stage22Row
            ok_std = validate_csv(std_output_path, required_columns=["system", "prompt", "chosen", "rejected"]) and True
            logger.info(f"Validated standardized CSV {std_output_path}: {ok_std}")
    except Exception:
        logger.warning("Validation failed unexpectedly; continuing.")

    # Write Parquet mirror for standardized dataset
    if run_id and std_output_path and std_parquet_path:
        try:
            df_std = pd.read_csv(std_output_path)
            df_std.to_parquet(std_parquet_path, index=False)
            logger.info(f"Standardized Parquet written to: {std_parquet_path}")
            write_contract(std_output_path, schema_version="v1", counts={"rows": len(df_std)})
            write_contract(std_parquet_path, schema_version="v1", counts={"rows": len(df_std)})
        except Exception as e:
            logger.warning(f"Failed to write Parquet or contract for 22-dataset: {e}")

    if std_csvfile and run_id:
        std_csvfile.close()
        if not ok_std:
            logger.error("Standardized CSV failed validation; skipping manifest update for 22-dataset")
            return
        manifest = read_manifest(run_id, base_dir)
        from utils.version import STAGE_VERSION
        signature = compute_hash([input_file], config={"stage": 22, "stage_version": STAGE_VERSION})
        update_stage(
            run_id,
            base_dir,
            manifest,
            stage_name="22-dataset",
            input_path=input_file,
            outputs=[p for p in [std_output_path, std_parquet_path] if p],
            signature=signature,
            extra={"validated": bool(ok_std)},
        )
        write_summary(run_id, "22-dataset", {"rows": int(len(df_std)) if 'df_std' in locals() else None, "csv": std_output_path, "parquet": std_parquet_path})
        logger.info(f"Standardized output written to: {std_output_path}")

    logger.info(f"Training data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate standardized dataset from prompts JSONL (run-id manifest mode only)")
    parser.add_argument("--input", dest="input_path", default=None)
    add_standard_args(parser)
    args = parser.parse_args()

    # Require run-id; no ad-hoc output path support
    args = resolve_common_args(args, require_input_when_no_run_id=True)

    print(f"Processing {args.input_path or '[manifest]'} -> [standardized]")
    if args.run_id:
        print(f"Also writing standardized artifact under {args.base_dir}/{args.run_id}")
    load_and_process_data(args.input_path, None, run_id=args.run_id, base_dir=args.base_dir)