import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import emojis

# Compatibility bootstrap: allow running this file directly or as part of the package
if __package__ is None or __package__ == "":
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from utils.logging_setup import init_pipeline_logging
from utils.manifest import read_manifest, compute_hash, should_skip, update_stage, discover_input
from utils.seed import set_global_seed
from utils.version import STAGE_VERSION


def load_emoji_bins(bins_path: Optional[str] = None) -> List[Tuple[float, str]]:
    """Load emoji frequency bins from JSON file or return defaults."""
    default_bins = [
        (0, "none"),
        (0.0005, "very low"),   # up to 0.05%
        (0.001, "low"),         # up to 0.1%
        (0.005, "medium"),      # up to 0.5%
        (0.01, "high"),         # up to 1%
        (1.0, "extreme")        # anything above 1% or up to 100%
    ]

    if not bins_path or not os.path.exists(bins_path):
        return default_bins

    try:
        with open(bins_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Expect format: [{"threshold": 0.0005, "label": "very low"}, ...]
        bins = []
        for item in data:
            bins.append((float(item["threshold"]), str(item["label"])))
        return sorted(bins, key=lambda x: x[0])  # Sort by threshold
    except Exception:
        return default_bins


def categorize_emoji_usage(frequency: float, bins: List[Tuple[float, str]]) -> str:
    """Return a label based on frequency thresholds."""
    last_label = "none"
    for threshold, label in bins:
        if frequency <= threshold:
            return label
        last_label = label
    return last_label


def emoji_frequency_analysis(post: str, bins: List[Tuple[float, str]]) -> Dict[str, any]:
    """Analyze emoji frequency in post text."""
    if not post:
        return {
            "emoji_count": 0,
            "text_length": 0,
            "frequency": 0.0,
            "usage": "none"
        }

    emoji_count = emojis.count(post)
    text_length = len(post)
    frequency = emoji_count / text_length if text_length > 0 else 0.0
    usage_label = categorize_emoji_usage(frequency, bins)

    return {
        "emoji_count": emoji_count,
        "text_length": text_length,
        "frequency": frequency,
        "usage": usage_label
    }


def post_length_analysis(post: str) -> str:
    """Categorize post length into predefined buckets."""
    length = len(post)

    if 750 <= length < 1501:
        return "Between 750 and 1,500 characters long"
    elif length >= 1501:
        return "Between 1,500 and 3,000 characters long"
    else:
        return "Up to 750 characters long"


def normalize_text(text: str) -> str:
    """Apply UTF-16 surrogate normalization."""
    try:
        return text.encode('utf-16', 'surrogatepass').decode('utf-16')
    except Exception:
        return text


def validate_post_fields(post: Dict) -> bool:
    """Basic validation of required post fields."""
    required_fields = ['emoji_count', 'text_length', 'emoji_frequency', 'emoji_usage', 'max_length']
    return all(field in post for field in required_fields)


def process_posts(
    input_path: str,
    output_path: str,
    run_id: Optional[str] = None,
    base_dir: str = "data/processed",
    seed: Optional[int] = None,
    normalize_text_flag: bool = True,
    emoji_bins_path: Optional[str] = None,
    logger = None
) -> Dict[str, int]:
    """Process posts with emoji and length labeling."""

    # Load emoji bins (needed for signature and processing)
    emoji_bins = load_emoji_bins(emoji_bins_path)
    logger.info(f"Using {len(emoji_bins)} emoji frequency bins")

    # Resolve IO centrally
    if run_id:
        from utils.io import resolve_io
        from utils.artifacts import ArtifactNames
        input_path, std_output_path, run_id = resolve_io(stage="02-label", run_id=run_id, base_dir=base_dir, explicit_in=input_path, prior_stage="01-find-gradient", std_name=ArtifactNames.STAGE02_LABELED)
        signature = compute_hash([input_path], config={
            "stage": 2,
            "seed": seed,
            "normalize_text": normalize_text_flag,
            "emoji_bins": emoji_bins,
            "stage_version": STAGE_VERSION,
        })
        manifest = read_manifest(run_id, base_dir)
        if should_skip(manifest, "02-label", signature, [std_output_path]):
            logger.info(f"Skipping 02-label; up-to-date at {std_output_path}")
            return {"processed": 0, "skipped": 0, "errors": 0}
    else:
        if not input_path:
            raise ValueError("When --run-id is not provided, you must specify --input (no ad-hoc outputs)")
        std_output_path = None

    # Process posts
    processed = 0
    skipped = 0
    errors = 0

    # Open standardized output when run_id present; otherwise write nothing
    out_handle = None
    try:
        if std_output_path:
            os.makedirs(os.path.dirname(std_output_path) or ".", exist_ok=True)
            out_handle = open(std_output_path, 'w', encoding='utf-8')
        else:
            out_handle = None  # No legacy writes

        with open(input_path, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    skipped += 1
                    continue

                try:
                    post = json.loads(line)

                    # Get post text
                    post_text = post.get('post_text', '')
                    if not post_text:
                        logger.warning(f"Line {line_num}: Empty post_text, skipping")
                        skipped += 1
                        continue

                    # Normalize text if requested
                    if normalize_text_flag:
                        post_text = normalize_text(post_text)
                        post['post_text'] = post_text

                    # Analyze emoji usage
                    emoji_analysis = emoji_frequency_analysis(post_text, emoji_bins)

                    # Add analysis results to post
                    post['emoji_usage'] = emoji_analysis['usage']
                    post['emoji_count'] = emoji_analysis['emoji_count']
                    post['text_length'] = emoji_analysis['text_length']
                    post['emoji_frequency'] = emoji_analysis['frequency']
                    post['max_length'] = post_length_analysis(post_text)

                    # Validate output
                    if not validate_post_fields(post):
                        logger.warning(f"Line {line_num}: Validation failed")
                        errors += 1
                        continue

                    # Write to output files
                    output_line = json.dumps(post, ensure_ascii=False) + '\n'
                    if out_handle:
                        out_handle.write(output_line)

                    processed += 1

                except json.JSONDecodeError:
                    logger.warning(f"Line {line_num}: Invalid JSON, skipping")
                    errors += 1
                    continue
                except Exception as e:
                    logger.error(f"Line {line_num}: Error processing post: {str(e)}")
                    errors += 1
                    continue

    finally:
        try:
            if out_handle:
                out_handle.close()
        except Exception:
            pass

    # Update manifest
    if std_output_path and signature and run_id:
        manifest = read_manifest(run_id, base_dir)
        update_stage(
            run_id=run_id,
            base_dir=base_dir,
            manifest=manifest,
            stage_name="02-label",
            input_path=input_path,
            outputs=[std_output_path],
            signature=signature,
            extra={"processed": processed, "skipped": skipped, "errors": errors},
        )
        logger.info(f"Standardized output written to: {std_output_path}")

    logger.info(f"Processing complete: {processed} processed, {skipped} skipped, {errors} errors")
    return {"processed": processed, "skipped": skipped, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description="Label posts with emoji and length analysis (run-id manifest mode only)")
    parser.add_argument("--run-id", dest="run_id", default=None, help="Use 'latest' to pick up the most recent run")
    parser.add_argument("--base-dir", dest="base_dir", default="data/processed")
    parser.add_argument("--input", dest="input_path", default=None, help="Input when not discoverable")
    parser.add_argument("--seed", dest="seed", type=int, default=None)
    parser.add_argument("--normalize-text", dest="normalize_text", action="store_true", default=True)
    parser.add_argument("--no-normalize-text", dest="normalize_text", action="store_false")
    parser.add_argument("--emoji-bins", dest="emoji_bins_path", default=None,
                       help="Path to JSON file with emoji frequency bins")
    args = parser.parse_args()

    logger = init_pipeline_logging("phase1.label", args.run_id, "02-label")

    # Resolve 'latest' run id ergonomics for stage 2+
    if args.run_id == "latest":
        from utils.run_id import get_last_run_id
        latest = get_last_run_id(args.base_dir)
        if not latest:
            raise ValueError("No .last_run_id found; run stage 1 first or provide --run-id")
        args.run_id = latest
        logger.info(f"Using latest run-id: {args.run_id}")

    stats = process_posts(
        input_path=args.input_path,
        output_path=None,
        run_id=args.run_id,
        base_dir=args.base_dir,
        seed=args.seed,
        normalize_text_flag=args.normalize_text,
        emoji_bins_path=args.emoji_bins_path,
        logger=logger,
    )

    logger.info(f"Final stats: {stats}")


if __name__ == "__main__":
    main()


