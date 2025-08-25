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
from utils.run_id import get_last_run_id
from utils.version import STAGE_VERSION

# Configure logging
logger = init_pipeline_logging("phase2.context", None, "14-extract-context")

structured_schema = {
    "type": "object",
    "properties": {
        "context": {
            "type": "string"
        }
    },
    "required": ["context"]
}

def _build_prompt(text: str, opinion: str) -> str:
    task = (
        "Extract the context and details from the following social media post that support the opinion.\n"
        "- Focus on references to outside stats, names, or context.\n"
        "- Do not add prefaces like 'The author thinks/believes' or 'I think/believe.'\n"
        "- Do not use the content from the examples for your answer. These are just stylistic examples.\n"
        "- Simply phrase the context from the user's perspective, as a direct statement.\n"
        "- Reply ONLY with the context. If there is no context reply with 'Unknown'\n\n"
        "**Social Media Post to Extract Context From**\n\n"
        f"{text}\n\n"
        "**Opinion from Social Media Post to Support with Context**\n\n"
        f"{opinion}"
    )
    return f"<|im_start|>user\n{task}<|im_end|>\n<|im_start|>assistant\n"


def process_batch(batch: List[Dict], llm: LLM, sampling_params: SamplingParams) -> Dict[str, int]:
    tasks: List[str] = []
    idx_map: List[int] = []
    for i, post in enumerate(batch):
        t = (post.get('post_text') or '').strip()
        o = (post.get('opinion') or '').strip()
        if not t or not o:
            post['context'] = ''
            continue
        tasks.append(_build_prompt(t, o))
        idx_map.append(i)

    if not tasks:
        return {"": len(batch)}

    outputs = llm.generate(tasks, sampling_params)

    dist: Dict[str, int] = {}
    for map_idx, output in enumerate(outputs):
        post = batch[idx_map[map_idx]]
        text = (output.outputs[0].text.strip() if output and output.outputs else "")
        context = ""
        try:
            result_json = json.loads(text)
            context = result_json.get("context", "").strip()
        except Exception:
            context = text
        post['context'] = context
        dist_key = 'nonempty' if context else 'empty'
        dist[dist_key] = dist.get(dist_key, 0) + 1
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

    # Resolve IO
    resolved_input: str
    resolved_output: str
    manifest = None

    if run_id:
        if run_id == "latest":
            rid = get_last_run_id(base_dir)
            if not rid:
                raise ValueError("No .last_run_id found; run previous stages first or provide --run-id")
            run_id = rid
        manifest = read_manifest(run_id, base_dir)
        discovered = discover_input(manifest, "12-clean-opinions") or discover_input(manifest, "11-extract-opinion")
        resolved_input = discovered or (input_path or "")
        if not resolved_input:
            raise ValueError("No input found: provide --input or ensure manifest contains 12-clean-opinions/11-extract-opinion output")
        out_dir = os.path.join(base_dir, run_id)
        os.makedirs(out_dir, exist_ok=True)
        resolved_output = os.path.join(out_dir, "14-context.jsonl")
        signature = compute_hash([resolved_input], {"stage": 14, "model": model, "batch": batch_size, "max_len": max_model_len, "temp": temperature, "stage_version": STAGE_VERSION})
        if should_skip(manifest, "14-extract-context", signature, [resolved_output]):
            logger.info(f"Skipping 14-extract-context; up-to-date at {resolved_output}")
            return
    else:
        if not input_path or not output_path:
            raise ValueError("When --run-id is not provided, you must specify both --input and --output")
        resolved_input = input_path
        resolved_output = output_path

    # Initialize vLLM
    llm = LLM(model=model, max_model_len=max_model_len)
    guided = GuidedDecodingParams(json=structured_schema)
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

    if run_id and manifest:
        manifest = read_manifest(run_id, base_dir)
        signature = compute_hash([resolved_input], {"stage": 14, "model": model, "batch": batch_size, "max_len": max_model_len, "temp": temperature, "stage_version": STAGE_VERSION})
        update_stage(run_id, base_dir, manifest, "14-extract-context", resolved_input, [resolved_output], signature, extra={"total": total})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract context statements using vLLM (Phase 2 compat CLI)")
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

    process_posts(
        input_path=args.input_path,
        output_path=args.output_path or 'step-14-posts-with-context.jsonl',
        run_id=args.run_id,
        base_dir=args.base_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        model=args.model,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
    )

