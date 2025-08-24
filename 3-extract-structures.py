import json
import logging
import re
import os
from datetime import datetime
from typing import List, Dict

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

# Compatibility bootstrap: allow running this file directly or as part of the package
if __package__ is None or __package__ == "":
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from pipe.utils.logging_setup import init_pipeline_logging
from pipe.utils.manifest import read_manifest, write_manifest, compute_hash, should_skip, update_stage, discover_input
from pipe.utils.seed import set_global_seed
from pipe.utils.version import STAGE_VERSION

# Configure logging
logger = init_pipeline_logging("phase2.structures", None, "03-extract-structures")

STRUCTURE_LABELS = [
    "instructional",
    "inspirational",
    "analytical",
    "insightful",
    "controversial",
    "comparative",
    "reflective",
    "evolutionary",
    "announcement",
]

INSTR = (
    "Given the following social media post, classify the content structure using one of the available "
    "structures. Only respond with the name of the structure (a single word).\n\n"
    "**Available Structures**\n\n"
    "- instructional: Posts offering practical, step-by-step advice, providing immediate value to the reader.\n"
    "- inspirational: Posts that share success stories or words of encouragement.\n"
    "- analytical: Data-driven posts presenting insights from studies or personal experiences.\n"
    "- controversial: Posts that challenge conventional wisdom or popular opinion.\n"
    "- insightful: Posts sharing thoughts or insights on current events or industry changes.\n"
    "- comparative: Posts that compare two or more things, such as products, strategies, or concepts.\n"
    "- reflective: Posts reflecting on past experiences, failures, or successes.\n"
    "- evolutionary: Posts highlighting the contrast between past and present scenarios.\n"
    "- announcement: Posts that grow excitement for something new or upcoming.\n"
)

def _build_prompt(text: str) -> str:
    return f"<|im_start|>user\n{INSTR}\n\n**Social Media Post**\n\n{text}<|im_end|>\n<|im_start|>assistant\n"


def process_batch(batch: List[Dict], llm: LLM, sampling_params: SamplingParams) -> Dict[str, int]:
    tasks: List[str] = []
    idx_map: List[int] = []
    for i, post in enumerate(batch):
        t = (post.get('post_text') or '').strip()
        if not t:
            post['structure'] = 'unknown'
            continue
        tasks.append(_build_prompt(t))
        idx_map.append(i)

    if not tasks:
        return {"unknown": len(batch)}

    outputs = llm.generate(tasks, sampling_params)

    dist: Dict[str, int] = {}
    for map_idx, output in enumerate(outputs):
        post = batch[idx_map[map_idx]]
        text = (output.outputs[0].text.strip().lower() if output and output.outputs else "")
        label = text if text in STRUCTURE_LABELS else "unknown"
        post['structure'] = label
        dist[label] = dist.get(label, 0) + 1
    return dist

def process_posts():
    # Create an output file that we'll write to progressively
    with open('step-3-posts-with-structures.jsonl', 'w') as outfile:
        # Process each line from input file
        with open('step-2-labeled-posts.jsonl', 'r') as infile:
            # Read all lines into memory first
            all_posts = []
            for line in infile:
                # Parse the JSON object from this line
                post = json.loads(line)
                all_posts.append(post)

            # Process in batches of 200
            batch_size = 200
            for i in range(0, len(all_posts), batch_size):
                logger.info(f"Processing batch {i//batch_size + 1} of {(len(all_posts) + batch_size - 1)//batch_size}")
                batch = all_posts[i:i + batch_size]

                # Process the current batch
                process_batch(batch)

                # Write the processed batch to the output file immediately
                for post in batch:
                    outfile.write(json.dumps(post) + '\n')

                # Force flush to ensure data is written to disk
                outfile.flush()

                logger.info(f"Completed batch {i//batch_size + 1}, wrote {len(batch)} posts to output file")

    logger.info("All inference completed. Process finished successfully.")



def batch_iter(path: str, batch_size: int):
    batch = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                batch.append(json.loads(line))
            except Exception:
                continue
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def write_report(dist: Dict[str, int], examples: Dict[str, List[str]], reports_dir: str):
    os.makedirs(reports_dir, exist_ok=True)
    lines = ["# 03 — Structure extraction report", ""]
    total = sum(dist.values()) or 1
    lines.append("## Distribution")
    for k in sorted(STRUCTURE_LABELS + ['unknown']):
        if k not in dist:
            continue
        pct = 100.0 * dist[k] / total
        lines.append(f"- {k}: {dist[k]} ({pct:.1f}%)")
    lines.append("\n## Examples (truncated)")
    for k, arr in examples.items():
        lines.append(f"\n### {k}")
        for s in arr[:3]:
            s = (s[:280] + "…") if len(s) > 280 else s
            lines.append(f"- {s}")
    with open(os.path.join(reports_dir, "03-report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def process_posts(input_path: str | None,
                  output_path: str | None,
                  run_id: str | None,
                  base_dir: str,
                  seed: int | None,
                  batch_size: int,
                  model: str,
                  max_model_len: int,
                  temperature: float,
                  report: bool = False):
    set_global_seed(seed)

    # Resolve IO based on run_id contract
    resolved_input: str
    resolved_output: str
    manifest = None

    if run_id:
        from utils.io import resolve_io
        from utils.artifacts import ArtifactNames
        resolved_input, resolved_output, run_id = resolve_io(stage="03-structures", run_id=run_id, base_dir=base_dir, explicit_in=input_path, prior_stage="02-label", std_name=ArtifactNames.STAGE03_STRUCTURES)
        manifest = read_manifest(run_id, base_dir)
        signature = compute_hash([resolved_input], {"stage": 3, "model": model, "batch": batch_size, "max_len": max_model_len, "temp": temperature, "stage_version": STAGE_VERSION})
        if should_skip(manifest, "03-structures", signature, [resolved_output]):
            logger.info(f"Skipping 03-structures; up-to-date at {resolved_output}")
            return
    else:
        if not input_path:
            raise ValueError("When --run-id is not provided, you must specify --input (no ad-hoc outputs)")
        resolved_input = input_path
        resolved_output = None
        signature = None

    # Initialize vLLM
    llm = LLM(model=model, max_model_len=max_model_len)
    guided = GuidedDecodingParams(choice=STRUCTURE_LABELS)
    sampling = SamplingParams(n=1, temperature=temperature, max_tokens=16, guided_decoding=guided)

    # Stream processing
    total = 0
    dist_total: Dict[str, int] = {}
    examples: Dict[str, List[str]] = {k: [] for k in STRUCTURE_LABELS}

    with open(resolved_output, 'w', encoding='utf-8') as out:
        for batch in batch_iter(resolved_input, batch_size):
            # Process
            batch_dist = process_batch(batch, llm, sampling)
            # Merge dist
            for k, v in batch_dist.items():
                dist_total[k] = dist_total.get(k, 0) + v
            # Capture examples
            for p in batch:
                lbl = p.get('structure', 'unknown')
                if lbl in examples and len(examples[lbl]) < 3:
                    txt = (p.get('post_text') or '')
                    if txt:
                        examples[lbl].append(txt)
            # Write outputs
            for p in batch:
                out.write(json.dumps(p, ensure_ascii=False) + "\n")
            total += len(batch)
            logger.info(f"Processed {total} posts so far…")

    logger.info("All inference completed. Process finished successfully.")

    # Validate standardized JSONL before updating manifest
    from pipe.schemas import Stage03Record
    from pipe.utils.validation import validate_jsonl_records
    ok_std = validate_jsonl_records(resolved_output, model_cls=Stage03Record, required_keys=["post_text", "structure"], allowed_values={"structure": set(STRUCTURE_LABELS) | {"unknown"}})

    # Update manifest
    if run_id and manifest and ok_std:
        manifest = read_manifest(run_id, base_dir)
        signature = compute_hash([resolved_input], {"stage": 3, "model": model, "batch": batch_size, "max_len": max_model_len, "temp": temperature, "stage_version": STAGE_VERSION})
        update_stage(run_id, base_dir, manifest, "03-structures", resolved_input, [resolved_output], signature, extra={"counts": dist_total, "total": total})
        logger.info(f"Standardized output written to: {resolved_output}")
    elif not ok_std:
        logger.error("Stage03 standardized JSONL failed validation; skipping manifest update")

    # Report
    if report and run_id:
        reports_dir = os.path.join("reports", run_id, "structures")
        write_report(dist_total, examples, reports_dir)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract content structure using vLLM (Phase 2 compat CLI)")
    parser.add_argument("--run-id", dest="run_id", default=None, help="Use 'latest' to pick up the most recent run")
    parser.add_argument("--base-dir", dest="base_dir", default="data/processed")
    parser.add_argument("--input", dest="input_path", default=None, help="Only used when --run-id is not provided")
    parser.add_argument("--output", dest="output_path", default=None, help="Only used when --run-id is not provided")
    parser.add_argument("--seed", dest="seed", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    parser.add_argument("--model", dest="model", default="Qwen/Qwen3-32B")
    parser.add_argument("--max-model-len", dest="max_model_len", type=int, default=8192)
    parser.add_argument("--temperature", dest="temperature", type=float, default=0.0)
    parser.add_argument("--report", dest="report", action="store_true", default=False)
    args = parser.parse_args()

    # Resolve 'latest' run id ergonomics for stage 2+
    if args.run_id == "latest":
        from pipe.utils.run_id import get_last_run_id
        latest = get_last_run_id(args.base_dir)
        if not latest:
            raise ValueError("No .last_run_id found; run stage 1 first or provide --run-id")
        args.run_id = latest
        logger.info(f"Using latest run-id: {args.run_id}")

    process_posts(
        input_path=args.input_path,
        output_path=args.output_path,
        run_id=args.run_id,
        base_dir=args.base_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        model=args.model,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        report=args.report,
    )

