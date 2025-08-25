import argparse
import csv
import os
import re
import random
from typing import Dict

# Compatibility bootstrap: allow running this file directly or as part of the package
if __package__ is None or __package__ == "":
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from utils.logging_setup import init_pipeline_logging
from utils.manifest import read_manifest, update_stage, should_skip, compute_hash
from utils.contracts import write_contract
from utils.version import STAGE_VERSION
from utils.reports import write_summary
from utils.violations import generate_negative

logger = init_pipeline_logging("phase2.add_negatives", None, "24-add-negatives")

# Pattern and regex for metrics
metric_pattern = r'''
\b\d+(\.\d+)?% |                      # Matches percentages e.g. 50%
\b\d+(\.\d+)?\s?(times|time|x) |      # Matches multipliers e.g. 3 times, 2x
\b\d+/\d+ |                           # Matches fractions e.g. 1/2
\b\d+:\d+ |                           # Matches ratios e.g. 1:2
\b(one|two|three|four|five|           # Matches number words
   six|seven|eight|nine|ten|
   eleven|twelve|hundred|
   thousand|million|billion)\b
(percent|%)?                          # Matches 'percent' after number words
'''

def does_text_contain_url(text):
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls_in_post = re.findall(url_pattern, text)
    return len(urls_in_post) > 0

def generate_fake_url():
    random_chars = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=8))
    return f"https://lnkd.in/{random_chars}"


def process_csv(input_path: str, output_path: str, run_id: str | None, base_dir: str) -> None:
    url_phrases = [
        "\n\nLearn more at my webinar at [URL].",
        "\n\nSign up for my upcoming webinar at [URL].",
        "\n\nJoin me for an insightful webinar. Register at [URL].",
        "\n\nDon't miss out on my webinar. Secure your spot at [URL].",
        "\n\nTune in to my webinar for valuable insights. Register at [URL].",
        "\n\nSubscribe to my newsletter at [URL] for exclusive content.",
        "\n\nStay updated with my latest insights. Subscribe at [URL].",
        "\n\nGet access to valuable resources. Sign up for my newsletter at [URL].",
        "\n\nDon't miss out on my newsletter. Subscribe now at [URL].",
        "\n\nJoin my newsletter community. Sign up at [URL].",
        '\n\n[URL]',
        '\n\nHere\'s the full conversation:',
        '\n\nHere\'s the full conversation 👇',
        '\n\nHere\'s the URL: [URL]',
        '\n\nLink in the comments 👇',
    ]

    name_phrases = [
        "\n\nShout out to [NAME] for their valuable contributions!",
        "\n\nIt was great hearing [NAME] speak at the conference.",
        "\n\nFollow [NAME] for more insightful posts!",
        "\n\n[NAME] shared some incredible insights in their recent article.",
        "\n\nHad a fantastic conversation with [NAME] about industry trends.",
        "\n\nCongratulations to [NAME] on their well-deserved promotion!",
        "\n\nExcited to collaborate with [NAME] on an upcoming project.",
        "\n\n[NAME] is a true thought leader in the field.",
        "\n\nLearned so much from [NAME]'s presentation at the webinar.",
        "\n\nCan't wait to see what [NAME] accomplishes next!",
        "\n\nI'm [NAME]",
        "\n\n------\n\nI'm [NAME]",
    ]

    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = ['system', 'prompt', 'chosen', 'rejected']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        total_count = 0
        filtered_count = 0
        other_count = 0

        for row in reader:
            system = row['system']
            prompt = row['prompt']
            chosen = row['chosen']
            rejected = row['rejected']

            if "Create a LinkedIn post that" in prompt:
                total_count += 1
                key_message_start = prompt.find("### Key Message\n") + len("### Key Message\n")
                key_message_end = prompt.find("\n```\n### Writing Constraints")
                key_message = prompt[key_message_start:key_message_end]

                # Build constraints and generate a violation-aware negative from the chosen text
                constraints: Dict[str, str] = {}
                if "### Writing Constraints" in prompt:
                    length_match = re.search(r"\*\*Suggested Post Length\*\*:\s*(.*)", prompt)
                    if length_match:
                        constraints["length"] = length_match.group(1).strip()
                    tone_match = re.search(r"\*\*Tone\*\*:\s*(.*)", prompt)
                    if tone_match:
                        constraints["tone"] = tone_match.group(1).strip()
                    emoji_match = re.search(r"\*\*Emoji Usage\*\*:\s*(.*)", prompt)
                    if emoji_match:
                        constraints["emoji_usage"] = emoji_match.group(1).strip()
                constraints.setdefault("hashtag_limit", 3)
                constraints.setdefault("allow_urls", False)
                constraints.setdefault("allow_names", False)

                rejected = generate_negative(chosen, constraints)
                filtered_count += 1

                row['rejected'] = rejected

            else:
                other_count += 1

            writer.writerow(row)  # Write the updated row to the file

        print(f"Total posts: {total_count}")
        print(f"Total filtered posts: {filtered_count}")
        print(f"Total Other: {other_count}")

def main():
    parser = argparse.ArgumentParser(description="Add violation-aware negatives to DPO dataset")
    parser.add_argument("--input", dest="input_path", default="dpo.csv")
    parser.add_argument("--output", dest="output_path", default="dpo-ready.csv")
    parser.add_argument("--run-id", dest="run_id", default=None)
    parser.add_argument("--base-dir", dest="base_dir", default="data/processed")
    args = parser.parse_args()

    # Resolve latest run-id if requested
    if args.run_id == "latest":
        try:
            from utils.run_id import get_last_run_id
            latest = get_last_run_id(args.base_dir)
            if latest:
                args.run_id = latest
        except Exception:
            pass

    # Use common IO helpers for discovery and standardization
    from utils.io import resolve_io
    from utils.artifacts import ArtifactNames

    # Prefer manifest discovery of 23-split outputs; choose DPO CSV among them
    input_path = args.input_path
    output_path = args.output_path
    if args.run_id and args.base_dir:
        # Discover any output of 23-split
        discovered, _, _ = resolve_io(stage="24-add-negatives", run_id=args.run_id, base_dir=args.base_dir, explicit_in=input_path, prior_stage="23-split", std_name=None)
        # If multiple, prefer the file ending with 23-dpo.csv
        if isinstance(discovered, list):
            candidates = [p for p in discovered if p.endswith("23-dpo.csv")] or discovered
            input_path = candidates[0]
        else:
            input_path = discovered
        std_dir = os.path.join(args.base_dir, args.run_id)
        os.makedirs(std_dir, exist_ok=True)
        output_path = os.path.join(std_dir, "24-dpo-ready.csv")
        pq_expected = os.path.join(std_dir, "24-dpo-ready.parquet")
        signature = compute_hash([input_path], config={"stage": 24, "stage_version": STAGE_VERSION})
        if should_skip(read_manifest(args.run_id, args.base_dir), "24-add-negatives", signature, [output_path, pq_expected]):
            logger.info(f"Skipping 24-add-negatives; up-to-date at {output_path}")
            return

    process_csv(input_path, output_path, args.run_id, args.base_dir)

    # Parquet + contract + summary
    try:
        import pandas as pd
        pq_path = None
        if args.run_id:
            pq_path = os.path.join(args.base_dir, args.run_id, "24-dpo-ready.parquet")
        df = pd.read_csv(output_path)
        if pq_path:
            df.to_parquet(pq_path, index=False)
        write_contract(output_path, schema_version="v1", counts={"rows": len(df)})
        if pq_path:
            write_contract(pq_path, schema_version="v1", counts={"rows": len(df)})
        from utils.reports import write_summary
        write_summary(args.run_id or "no-run", "24-add-negatives", {"rows": len(df), "csv": output_path, "parquet": pq_path})
        if args.run_id:
            manifest = read_manifest(args.run_id, args.base_dir)
            update_stage(args.run_id, args.base_dir, manifest, "24-add-negatives", input_path=input_path, outputs=[p for p in [output_path, pq_path] if p], signature=signature, extra={"count": len(df)})
    except Exception:
        pass


if __name__ == "__main__":
    main()

