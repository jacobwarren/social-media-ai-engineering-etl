import json
import sys
import subprocess
from pathlib import Path


def run(cmd, cwd):
    res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0, f"Command failed: {' '.join(cmd)}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    return res


def test_integration_01_find_gradient_then_02_label(tmp_path: Path):
    # Arrange: create a tiny example-dataset.jsonl with a couple of users each having posts
    input_path = tmp_path / "example-dataset.jsonl"
    users = [
        {
            "user_id": 1,
            "posts": [
                {"post_text": "Hello AI world!", "follower_count": 1000, "comments_count": 10, "total_likes_count": 50, "shares_count": 5},
                {"post_text": "No emoji here", "follower_count": 800, "comments_count": 2, "total_likes_count": 5, "shares_count": 0},
            ],
        },
        {
            "user_id": 2,
            "posts": [
                {"post_text": "Great launch 🚀", "follower_count": 1200, "comments_count": 20, "total_likes_count": 120, "shares_count": 10},
            ],
        },
    ]
    with input_path.open("w", encoding="utf-8") as f:
        for rec in users:
            f.write(json.dumps(rec) + "\n")

    out1 = tmp_path / "step-1-best-posts.jsonl"
    out2 = tmp_path / "step-2-labeled-posts.jsonl"
    proc_dir = tmp_path / "proc"
    reports_dir = tmp_path / "reports"
    run_id = "ci"

    # Act: run step 1 (standardized outputs only)
    run([
        sys.executable,
        "pipe/1-find-gradient.py",
        "--input",
        str(input_path),
        "--run-id",
        run_id,
        "--base-dir",
        str(proc_dir),
        "--reports-dir",
        str(reports_dir),
        "--report",
    ], cwd=str(Path.cwd()))

    std1 = proc_dir / run_id / "01-best-posts.jsonl"
    assert std1.exists() and std1.stat().st_size > 0

    # Act: run step 2
    run([
        sys.executable,
        "pipe/2-label.py",
        "--input",
        str(std1),
        "--run-id",
        run_id,
        "--base-dir",
        str(proc_dir),
        "--seed",
        "3407",
    ], cwd=str(Path.cwd()))

    # Assert: labeled standardized output exists and contains required fields
    std2 = proc_dir / run_id / "02-labeled.jsonl"
    assert std2.exists() and std2.stat().st_size > 0

    # Read a couple of lines and inspect keys
    sample_lines = []
    with std2.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            sample_lines.append(json.loads(line))

    assert sample_lines, "No labeled lines read"
    required_keys = {"emoji_usage", "emoji_count", "emoji_frequency", "max_length", "post_text"}
    for rec in sample_lines:
        assert required_keys.issubset(rec.keys())
        assert isinstance(rec["emoji_count"], int)
        assert isinstance(rec["emoji_frequency"], (int, float))
        assert rec["emoji_usage"] in {"none", "very low", "low", "medium", "high", "extreme"}

