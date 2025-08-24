import json
import os
import re
from typing import Tuple

# Compatibility bootstrap: allow running this file directly or as part of the package
if __package__ is None or __package__ == "":
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from pipe.utils.logging_setup import init_pipeline_logging
from pipe.utils.manifest import read_manifest, compute_hash, should_skip, update_stage, discover_input
from pipe.utils.run_id import get_last_run_id
from pipe.utils.version import STAGE_VERSION

logger = init_pipeline_logging("phase2.clean_opinions", None, "12-clean-opinions")

def contains_disallowed(opinion: str) -> bool:
    """
    Returns True if the opinion contains any disallowed substrings or characters:
      - Chinese characters (in the range \u4e00-\u9fff)
      - The word "unknown" (case-insensitive, after stripping)
    """
    # Check for any Chinese character
    if re.search(r'[\u4e00-\u9fff]', opinion):
        return True
        
    # Check if the opinion is just "unknown" (case-insensitive, after stripping)
    if opinion.strip().lower() == "unknown":
        return True

    return False

def parse_field(field: str) -> str:
    """
    Attempts to parse the field if it is a JSON string containing an object.
    If it can be parsed and contains a 'opinion' key, returns that value;
    otherwise returns the original string.
    """
    try:
        parsed = json.loads(field)
        if isinstance(parsed, dict) and 'opinion' in parsed:
            return parsed['opinion']
    except (json.JSONDecodeError, TypeError):
        pass
    return field

def _resolve_io(run_id: str | None, base_dir: str, input_path: str | None, output_path: str | None) -> Tuple[str, str, str | None]:
    manifest = None
    if run_id:
        if run_id == "latest":
            rid = get_last_run_id(base_dir)
            if not rid:
                raise ValueError("No .last_run_id found; run earlier stages first or provide --run-id")
            run_id = rid
        manifest = read_manifest(run_id, base_dir)
        discovered = discover_input(manifest, "11-extract-opinion") or discover_input(manifest, "09-extract-tone")
        in_path = discovered or input_path
        if not in_path:
            raise ValueError("No input found: provide --input or ensure manifest has 11-extract-opinion or 09-extract-tone output")
        out_dir = os.path.join(base_dir, run_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "12-clean-opinion.jsonl")
        return in_path, out_path, run_id
    else:
        if not input_path or not output_path:
            raise ValueError("When --run-id is not provided, you must specify both --input and --output")
        return input_path, output_path, None


def clean_opinions(input_path: str | None,
                   output_path: str | None,
                   run_id: str | None,
                   base_dir: str) -> dict:
    in_path, out_path, resolved_run = _resolve_io(run_id, base_dir, input_path, output_path)

    # Idempotency: skip if up-to-date
    signature = compute_hash([in_path], {"stage": 12, "transform": "clean-opinions", "stage_version": STAGE_VERSION})
    if resolved_run:
        manifest = read_manifest(resolved_run, base_dir)
        if should_skip(manifest, "12-clean-opinions", signature, [out_path]):
            logger.info(f"Skipping 12-clean-opinions; up-to-date at {out_path}")
            return {"input": in_path, "output": out_path, "removed": None, "written": None}

    removed = 0
    written = 0

    with open(out_path, 'w', encoding='utf-8') as outfile:
        with open(in_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    post = json.loads(line)
                except Exception:
                    continue

                if not post.get('post_text'):
                    continue

                field = post.get('opinion', '')
                text = parse_field(field)

                if contains_disallowed(text):
                    removed += 1
                    continue

                post['opinion'] = text
                outfile.write(json.dumps(post, ensure_ascii=False) + '\n')
                written += 1

    logger.info(f"Cleaned {removed} bad records; wrote {written} records to {out_path}")

    if resolved_run:
        manifest = read_manifest(resolved_run, base_dir)
        update_stage(resolved_run, base_dir, manifest, "12-clean-opinions", in_path, [out_path], signature,
                     extra={"removed": removed, "written": written})

    return {"input": in_path, "output": out_path, "removed": removed, "written": written}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean opinion field strings and filter disallowed content")
    parser.add_argument("--run-id", dest="run_id", default=None, help="Use 'latest' to pick up the most recent run")
    parser.add_argument("--base-dir", dest="base_dir", default="data/processed")
    parser.add_argument("--input", dest="input_path", default=None, help="Only used when --run-id is not provided")
    parser.add_argument("--output", dest="output_path", default=None, help="Only used when --run-id is not provided")
    args = parser.parse_args()

    stats = clean_opinions(
        input_path=args.input_path,
        output_path=args.output_path,
        run_id=args.run_id,
        base_dir=args.base_dir,
    )
    logger.info(f"Final stats: {stats}")