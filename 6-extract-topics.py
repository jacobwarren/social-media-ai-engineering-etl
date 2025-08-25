import json
import os
from typing import List, Dict

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

# Compatibility bootstrap: allow running this file directly or as part of the package
if __package__ is None or __package__ == "":
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from utils.logging_setup import init_pipeline_logging
from utils.manifest import read_manifest, compute_hash, should_skip, update_stage, discover_input
from utils.seed import set_global_seed
from utils.version import STAGE_VERSION

# Configure logging
logger = init_pipeline_logging("phase2.topics", None, "06-extract-topics")

structured_schema = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"}
    },
    "required": ["topic"]
}

def _build_prompt(text: str) -> str:
    task = (
        "# Request\n\n"
        "Given the following social media post, classify the overarching topic of the post.\n\n"
        "Topic should be title cased, such as \"Content Marketing\", \"Deep Learning\", etc.\n\n"
        "## Social Media Post\n\n"
        f"{text}"
    )
    return f"<|im_start|>user\n{task}<|im_end|>\n<|im_start|>assistant\n"


def process_batch(batch: List[Dict], llm: LLM, sampling_params: SamplingParams) -> Dict[str, int]:
    tasks: List[str] = []
    idx_map: List[int] = []
    for i, post in enumerate(batch):
        t = (post.get('post_text') or '').strip()
        if not t:
            post['topic'] = 'Unknown'
            continue
        tasks.append(_build_prompt(t))
        idx_map.append(i)

    if not tasks:
        return {"Unknown": len(batch)}

    outputs = llm.generate(tasks, sampling_params)

    dist: Dict[str, int] = {}
    for map_idx, output in enumerate(outputs):
        post = batch[idx_map[map_idx]]
        text = (output.outputs[0].text.strip() if output and output.outputs else "")
        topic = "Unknown"
        try:
            result_json = json.loads(text)
            topic = result_json.get("topic", "Unknown").strip() or "Unknown"
        except Exception:
            topic = "Unknown"
        post['topic'] = topic
        dist[topic] = dist.get(topic, 0) + 1
    return dist



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

def process_posts(input_path: str | None,
                  output_path: str | None,
                  run_id: str | None,
                  base_dir: str,
                  seed: int | None,
                  batch_size: int,
                  model: str,
                  max_model_len: int,
                  temperature: float):
    set_global_seed(seed)

    # Resolve IO like stage 3
    resolved_input: str
    resolved_output: str
    manifest = None

    if run_id:
        manifest = read_manifest(run_id, base_dir)
        discovered = discover_input(manifest, "05-balance") or discover_input(manifest, "03-structures")
        resolved_input = discovered or (input_path or "")
        if not resolved_input:
            raise ValueError("No input found: provide --input or ensure manifest contains 05-balance or 03-structures output")
        out_dir = os.path.join(base_dir, run_id)
        os.makedirs(out_dir, exist_ok=True)
        resolved_output = os.path.join(out_dir, "06-topics.jsonl")
        signature = compute_hash([resolved_input], {"stage": 6, "model": model, "batch": batch_size, "max_len": max_model_len, "temp": temperature, "stage_version": STAGE_VERSION})
        if should_skip(manifest, "06-extract-topics", signature, [resolved_output]):
            logger.info(f"Skipping 06-extract-topics; up-to-date at {resolved_output}")
            return
    else:
        if not input_path or not output_path:
            raise ValueError("When --run-id is not provided, you must specify both --input and --output")
        resolved_input = input_path
        resolved_output = output_path

    # Initialize vLLM
    llm = LLM(model=model, max_model_len=max_model_len)
    guided = GuidedDecodingParams(json=structured_schema, backend="outlines")
    sampling = SamplingParams(n=1, temperature=temperature, max_tokens=100, guided_decoding=guided)

    total = 0
    with open(resolved_output, 'w', encoding='utf-8') as out:
        for batch in batch_iter(resolved_input, batch_size):
            process_batch(batch, llm, sampling)
            for p in batch:
                out.write(json.dumps(p, ensure_ascii=False) + "\n")
            total += len(batch)
            logger.info(f"Processed {total} posts so far…")

    logger.info("All inference completed. Process finished successfully.")

    # Update manifest
    if run_id and manifest:
        manifest = read_manifest(run_id, base_dir)
        signature = compute_hash([resolved_input], {"stage": 6, "model": model, "batch": batch_size, "max_len": max_model_len, "temp": temperature, "stage_version": STAGE_VERSION})
        update_stage(run_id, base_dir, manifest, "06-extract-topics", resolved_input, [resolved_output], signature, extra={"total": total})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract topics using vLLM (Phase 2 compat CLI)")
    parser.add_argument("--run-id", dest="run_id", default=None, help="Use 'latest' to pick up the most recent run")
    parser.add_argument("--base-dir", dest="base_dir", default="data/processed")
    parser.add_argument("--input", dest="input_path", default=None, help="Only used when --run-id is not provided")
    parser.add_argument("--output", dest="output_path", default=None, help="Only used when --run-id is not provided")
    parser.add_argument("--seed", dest="seed", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    parser.add_argument("--model", dest="model", default="Qwen/Qwen3-32B")
    parser.add_argument("--max-model-len", dest="max_model_len", type=int, default=8192)
    parser.add_argument("--temperature", dest="temperature", type=float, default=0.0)
    args = parser.parse_args()

    # Resolve 'latest' run id
    if args.run_id == "latest":
        from utils.run_id import get_last_run_id
        latest = get_last_run_id(args.base_dir)
        if not latest:
            raise ValueError("No .last_run_id found; run stage 1 first or provide --run-id")
        args.run_id = latest
        logger.info(f"Using latest run-id: {args.run_id}")

    process_posts(
        input_path=args.input_path,
        output_path=args.output_path or 'step-6-posts-with-topics.jsonl',
        run_id=args.run_id,
        base_dir=args.base_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        model=args.model,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
    )
