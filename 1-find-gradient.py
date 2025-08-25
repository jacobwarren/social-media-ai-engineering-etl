import os
import json
import argparse
from datetime import datetime
from typing import Dict, Tuple

import numpy as np

# Compatibility bootstrap: allow running this file directly or as part of the package
if __package__ is None or __package__ == "":
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from utils.logging_setup import init_pipeline_logging
from utils.manifest import read_manifest, compute_hash, should_skip, update_stage
from utils.seed import set_global_seed
from utils.version import STAGE_VERSION


def calculate_ratio(followers: float, comments: float, likes: float, shares: float, clamp: float) -> float:
    if not followers or followers <= 0:
        return 0.0
    total_engagement = max(0.0, float(comments)) + max(0.0, float(likes)) + max(0.0, float(shares))
    ratio = total_engagement / float(followers)
    if clamp is not None:
        ratio = max(0.0, min(ratio, float(clamp)))
    return float(ratio)


def compute_cutoffs(input_path: str, clamp: float, top_pct: float, bottom_pct: float, logger) -> Tuple[float, float, int]:
    ratios: list[float] = []
    posts_seen = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            posts_field = record.get("posts", None)
            if isinstance(posts_field, list):
                posts_iter = posts_field
            else:
                # Support flat JSONL where each line is a single post object
                posts_iter = [record] if ("post_text" in record or "follower_count" in record or "total_likes_count" in record) else []
            for post in posts_iter:
                followers = post.get("follower_count", 0) or 0
                comments = post.get("comments_count", 0) or 0
                likes = post.get("total_likes_count", 0) or 0
                shares = post.get("shares_count", 0) or 0
                r = calculate_ratio(followers, comments, likes, shares, clamp)
                ratios.append(r)
                posts_seen += 1
    if not ratios:
        logger.warning("No posts found to compute ratios.")
        return 0.0, 0.0, 0
    top_cut = float(np.quantile(ratios, top_pct))
    bot_cut = float(np.quantile(ratios, bottom_pct))
    return top_cut, bot_cut, posts_seen


def write_filtered(
    input_path: str,
    outputs: Dict[str, str],
    clamp: float,
    top_cut: float,
    bot_cut: float,
    keep_bottom: bool,
    logger,
) -> Dict[str, int]:
    counts = {"A": 0, "B": 0, "C": 0}
    writers = {}
    files = {}
    try:
        for name, path in outputs.items():
            if path:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                files[name] = open(path, "w", encoding="utf-8")
                writers[name] = files[name]
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                posts_field = record.get("posts", None)
                if isinstance(posts_field, list):
                    posts_iter = posts_field
                else:
                    posts_iter = [record] if ("post_text" in record or "follower_count" in record or "total_likes_count" in record) else []
                for post in posts_iter:
                    followers = post.get("follower_count", 0) or 0
                    comments = post.get("comments_count", 0) or 0
                    likes = post.get("total_likes_count", 0) or 0
                    shares = post.get("shares_count", 0) or 0
                    r = calculate_ratio(followers, comments, likes, shares, clamp)
                    if r >= top_cut:
                        tier = "Tier A (Top 20%)"
                        counts["A"] += 1
                    elif r < bot_cut:
                        tier = "Tier C (Bottom 40%)"
                        counts["C"] += 1
                    else:
                        tier = "Tier B (Middle 40%)"
                        counts["B"] += 1
                    post["engagement_ratio"] = r
                    post["tier"] = tier
                    if ("legacy" in writers) and (keep_bottom or tier != "Tier C (Bottom 40%)"):
                        writers["legacy"].write(json.dumps(post, ensure_ascii=False) + "\n")
                    if ("std" in writers) and (keep_bottom or tier != "Tier C (Bottom 40%)"):
                        writers["std"].write(json.dumps(post, ensure_ascii=False) + "\n")
    finally:
        for f in files.values():
            try:
                f.close()
            except Exception:
                pass
    logger.info(f"Tier counts: A={counts['A']} B={counts['B']} C={counts['C']}")
    return counts


def maybe_write_report(
    run_id: str | None,
    reports_dir: str,
    counts: Dict[str, int],
    top_cut: float,
    bot_cut: float,
    top_pct: float,
    bottom_pct: float,
    logger,
) -> None:
    if not run_id:
        return
    out_dir = os.path.join(reports_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)

    # Try plotting if available
    plot_path = os.path.join(out_dir, "01-engagement-tier-bars.png")
    try:
        import matplotlib.pyplot as plt  # type: ignore
        labels = ["A", "B", "C"]
        values = [counts.get("A", 0), counts.get("B", 0), counts.get("C", 0)]
        plt.figure(figsize=(5, 3))
        plt.bar(labels, values)
        plt.title("Tier counts")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)
        plt.close()
    except Exception:
        plot_path = None

    # Markdown report
    md = ["# 01 — Engagement tiering report", ""]
    md.append(f"Top percentile (A threshold): {top_pct:.2f} → cutoff={top_cut:.6f}")
    md.append(f"Bottom percentile (C threshold): {bottom_pct:.2f} → cutoff={bot_cut:.6f}")
    md.append("")
    md.append("## Tier counts")
    md.append(f"- A: {counts.get('A', 0)}")
    md.append(f"- B: {counts.get('B', 0)}")
    md.append(f"- C: {counts.get('C', 0)}")
    if plot_path:
        md.append("\n## Chart\n")
        md.append("![](01-engagement-tier-bars.png)")
    with open(os.path.join(out_dir, "01-report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    logger.info(f"Wrote report to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compute engagement tiers (run-id mode; generates run_id if missing)")
    parser.add_argument("--run-id", dest="run_id", default=None)
    parser.add_argument("--base-dir", dest="base_dir", default="data/processed")
    parser.add_argument("--reports-dir", dest="reports_dir", default="reports")
    parser.add_argument("--input", dest="input_path", default=None, help="Required: path to dataset JSONL")
    parser.add_argument("--top-pct", dest="top_pct", type=float, default=0.80)
    parser.add_argument("--bottom-pct", dest="bottom_pct", type=float, default=0.40)
    parser.add_argument("--clamp", dest="clamp", type=float, default=0.05)
    parser.add_argument("--keep-bottom", dest="keep_bottom", action="store_true", default=False)
    parser.add_argument("--seed", dest="seed", type=int, default=None)
    parser.add_argument("--report", dest="report", action="store_true", default=False)
    args = parser.parse_args()

    logger = init_pipeline_logging("phase1.find_gradient", args.run_id, "01-find-gradient")
    set_global_seed(args.seed)

    # Resolve run_id ergonomics for stage 1
    from utils.run_id import generate_run_id, save_last_run_id, write_run_metadata
    if not args.run_id:
        # Auto-generate for stage 1 and save
        args.run_id = generate_run_id()
        save_last_run_id(args.base_dir, args.run_id)
        write_run_metadata(args.base_dir, args.run_id, {"stage": 1, "script": "01-find-gradient"})

    # Resolve IO (standardized-only when run_id is set)
    std_output_path = None
    signature = None
    if args.run_id:
        if not args.input_path:
            raise ValueError("--input is required for the first stage when using --run-id")
        from utils.io import resolve_io
        from utils.artifacts import ArtifactNames
        _, std_output_path, args.run_id = resolve_io(stage="01-find-gradient", run_id=args.run_id, base_dir=args.base_dir, explicit_in=args.input_path, std_name=ArtifactNames.STAGE01_BEST_POSTS)
        signature = compute_hash([args.input_path], config={
            "stage": 1,
            "top_pct": args.top_pct,
            "bottom_pct": args.bottom_pct,
            "clamp": args.clamp,
            "keep_bottom": args.keep_bottom,
            "stage_version": STAGE_VERSION,
        })
        manifest = read_manifest(args.run_id, args.base_dir)
        if should_skip(manifest, "01-find-gradient", signature, [std_output_path]):
            logger.info(f"Skipping 01-find-gradient; up-to-date at {std_output_path}")
            return

    # Pass 1: compute cutoffs
    top_cut, bot_cut, total_posts = compute_cutoffs(args.input_path, args.clamp, args.top_pct, args.bottom_pct, logger)
    if total_posts == 0:
        logger.warning("No posts processed; exiting.")
        return

    # Pass 2: write standardized output only (no legacy outputs)
    outputs = {"std": std_output_path} if args.run_id else {}
    target_path = std_output_path if args.run_id else None

    counts = write_filtered(
        input_path=args.input_path,
        outputs=outputs,
        clamp=args.clamp,
        top_cut=top_cut,
        bot_cut=bot_cut,
        keep_bottom=args.keep_bottom,
        logger=logger,
    )

    logger.info(f"Wrote {sum(counts.values())} posts to {target_path}")

    # Validate standardized output before manifest update
    from schemas import Stage01Record
    from utils.validation import validate_jsonl_records
    ok_std = True
    if std_output_path:
        ok_std = validate_jsonl_records(std_output_path, model_cls=Stage01Record, required_keys=["post_text", "tier"])  # basic check
        if not ok_std:
            logger.error("Stage01 standardized JSONL failed validation; skipping manifest update")

    # Manifest update
    if std_output_path and signature and args.run_id and ok_std:
        manifest = read_manifest(args.run_id, args.base_dir)
        update_stage(
            run_id=args.run_id,
            base_dir=args.base_dir,
            manifest=manifest,
            stage_name="01-find-gradient",
            input_path=args.input_path,
            outputs=[std_output_path],
            signature=signature,
            extra={"counts": counts, "top_cut": top_cut, "bottom_cut": bot_cut},
        )
        logger.info(f"Standardized output written to: {std_output_path}")

    # Optional report
    if args.report:
        try:
            maybe_write_report(
                run_id=args.run_id,
                reports_dir=args.reports_dir,
                counts=counts,
                top_cut=top_cut,
                bot_cut=bot_cut,
                top_pct=args.top_pct,
                bottom_pct=args.bottom_pct,
                logger=logger,
            )
        except Exception as e:
            logger.warning(f"Failed to write report: {e}")


if __name__ == "__main__":
    main()
