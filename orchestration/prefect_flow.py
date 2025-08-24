from __future__ import annotations
import os
import sys
from pathlib import Path
from datetime import datetime

from prefect import flow, task
from pipe.utils.logging_setup import init_pipeline_logging


def run(cmd: list[str], cwd: str | None = None):
    import subprocess
    p = subprocess.run(cmd, cwd=cwd or str(Path(__file__).resolve().parents[1]), text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


logger = init_pipeline_logging("orchestration.prefect", None, "prefect-flow")


@task
def step_17(input_path: str, output_path: str, tmp_output: str, run_id: str, base_dir: str, seed: int | None = None):
    run([sys.executable, "17-writing-style.py", "--input", input_path, "--output", output_path, "--temp-output", tmp_output, "--run-id", run_id, "--base-dir", base_dir, *( ["--seed", str(seed)] if seed is not None else [] )])


@task
def step_18(input_path: str, output_path: str, run_id: str, base_dir: str, seed: int | None = None):
    run([sys.executable, "18-generate-prompts.py", "--input", input_path, "--output", output_path, "--run-id", run_id, "--base-dir", base_dir, *( ["--seed", str(seed)] if seed is not None else [] )])


@task
def step_22(input_path: str, output_path: str, run_id: str, base_dir: str):
    run([sys.executable, "22-generate-dataset.py", "--input", input_path, "--output", output_path, "--run-id", run_id, "--base-dir", base_dir])


@task
def step_23(input_path: str, balanced: str, sft: str, dpo: str, run_id: str, base_dir: str):
    run([sys.executable, "23-split.py", "--input", input_path, "--balanced", balanced, "--sft", sft, "--dpo", dpo, "--run-id", run_id, "--base-dir", base_dir, "--disable-augmentation"])


@task
def eval_rewards(run_id: str, reports_dir: str, weights: str | None = None):
    cmd = [sys.executable, "scripts/evaluate_rewards.py", "--run-id", run_id, "--base-dir", reports_dir]
    if weights:
        cmd += ["--weights", weights]
    run(cmd)


@flow(name="pipe-phase8-flow")
def pipeline(
    run_id: str | None = None,
    base_dir: str = "data/processed",
    reports_dir: str = "reports",
    tmp_dir: str = "tmp",
    weights: str | None = None,
):
    """Prefect flow orchestrating 17 -> 18 -> 22 -> 23 and evaluation.
    End-to-end demo with standardized artifacts; suitable for portfolio/demo.
    """
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp = Path(tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)

    in17 = tmp / "step-15-cleaned.jsonl"
    # For demo: create a tiny synthetic file if not present
    if not in17.exists():
        sample = {
            "post_text": "We launched a tiny AI tool.",
            "structure": "instructional",
            "topic": "AI tooling",
            "opinion": "I think it's useful",
            "context": "for quick tasks",
            "tone": "Serious",
        }
        with in17.open("w", encoding="utf-8") as f:
            import json
            f.write(json.dumps(sample) + "\n")

    out17 = tmp / "step-17-posts-with-writing-style.jsonl"
    tmp17 = tmp / "step-17-partial-results.jsonl"
    out18 = tmp / "step-18-with-prompts.jsonl"
    out22 = tmp / "ready-dataset.csv"
    out_bal = tmp / "balanced-dataset.csv"
    out_sft = tmp / "sft.csv"
    out_dpo = tmp / "dpo.csv"

    logger.info(f"Submitting flow run_id={run_id}")
    step_17.submit(str(in17), str(out17), str(tmp17), run_id, base_dir)
    step_18.submit(str(out17), str(out18), run_id, base_dir)
    step_22.submit(str(out18), str(out22), run_id, base_dir)
    step_23.submit(str(out22), str(out_bal), str(out_sft), str(out_dpo), run_id, base_dir)
    eval_rewards.submit(run_id, reports_dir, weights)
    logger.info("Submitted all tasks")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run Prefect flow for 18→22→23 + eval")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--base-dir", default="data/processed")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--tmp-dir", default="tmp")
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()
    pipeline(run_id=args.run_id, base_dir=args.base_dir, reports_dir=args.reports_dir, tmp_dir=args.tmp_dir, weights=args.weights)

