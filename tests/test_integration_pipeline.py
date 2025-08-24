import json
import os
import sys
import subprocess
from pathlib import Path


def run(cmd, cwd):
    result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0, f"Command failed: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return result


def test_etl_backbone_18_22_23(tmp_path: Path):
    # Arrange: tiny synthetic input for step 18
    input_path = tmp_path / "step-17-posts-with-writing-style.jsonl"
    posts = [
        {
            "post_text": "We launched a tiny AI tool.",
            "structure": "instructional",
            "topic": "AI tooling",
            "opinion": "I think it's useful",
            "context": "for quick tasks",
            "tone": "Serious",
        },
        {
            "post_text": "On building useful ML projects.",
            "structure": "instructional",
            "topic": "ML projects",
            "opinion": "Focus on reproducibility",
            "context": "in small teams",
            "tone": "Serious",
        },
    ]
    with input_path.open("w", encoding="utf-8") as f:
        for p in posts:
            f.write(json.dumps(p) + "\n")

    proc_dir = tmp_path / "proc"
    run_id = "ci"

    # Act: run 18 -> 22 -> 23 via CLIs (standardized outputs under base_dir/run_id)
    run([sys.executable, "pipe/18-generate-prompts.py", "--input", str(input_path), "--run-id", run_id, "--base-dir", str(proc_dir)], cwd=str(Path.cwd()))

    std18 = proc_dir / run_id / "18-with-prompts.jsonl"
    assert std18.exists() and std18.stat().st_size > 0

    run([sys.executable, "pipe/22-generate-dataset.py", "--input", str(std18), "--run-id", run_id, "--base-dir", str(proc_dir)], cwd=str(Path.cwd()))
    std22 = proc_dir / run_id / "22-ready-dataset.csv"
    assert std22.exists() and std22.stat().st_size > 0

    run([sys.executable, "pipe/23-split.py", "--input", str(std22), "--run-id", run_id, "--base-dir", str(proc_dir), "--disable-augmentation"], cwd=str(Path.cwd()))

    std_sft = proc_dir / run_id / "23-sft.csv"
    std_dpo = proc_dir / run_id / "23-dpo.csv"
    assert std_sft.exists() and std_dpo.exists()
    # Basic column checks
    sft_head = std_sft.read_text(encoding="utf-8").splitlines()[:1]
    dpo_head = std_dpo.read_text(encoding="utf-8").splitlines()[:1]
    assert sft_head and ("prompt" in sft_head[0] or ",prompt," in sft_head[0])
    assert dpo_head and ("chosen" in dpo_head[0] and "rejected" in dpo_head[0])

