import os
import csv
import importlib
import types

def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "prompt", "chosen", "rejected"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def test_split_determinism(tmp_path):
    m = importlib.import_module("23-split")
    rows = [
        {"system": "", "prompt": "Create a LinkedIn post that shares a step-by-step guide", "chosen": "A", "rejected": "B"},
        {"system": "", "prompt": "Create a LinkedIn post that reflects on an experience", "chosen": "C", "rejected": "D"},
        {"system": "", "prompt": "Create a LinkedIn post that inspires and motivates", "chosen": "E", "rejected": "F"},
    ]

    inp = tmp_path / "prompts.csv"
    _write_csv(inp, rows)

    out_bal = tmp_path / "balanced.csv"
    out_sft = tmp_path / "sft.csv"
    out_dpo = tmp_path / "dpo.csv"

    # First run
    m.process_csv(
        input_file=str(inp),
        balanced_file=str(out_bal),
        sft_file=str(out_sft),
        dpo_file=str(out_dpo),
        run_id=None,
        base_dir=str(tmp_path),
        seed=42,
        sft_percentage=0.66,
        dpo_percentage=0.34,
        prefer_downsampling=True,
        disable_augmentation=True,
    )
    contents1 = (_read(out_bal), _read(out_sft), _read(out_dpo))

    # Clean outputs and re-run with same params
    for p in [out_bal, out_sft, out_dpo]:
        os.remove(p)

    m.process_csv(
        input_file=str(inp),
        balanced_file=str(out_bal),
        sft_file=str(out_sft),
        dpo_file=str(out_dpo),
        run_id=None,
        base_dir=str(tmp_path),
        seed=42,
        sft_percentage=0.66,
        dpo_percentage=0.34,
        prefer_downsampling=True,
        disable_augmentation=True,
    )
    contents2 = (_read(out_bal), _read(out_sft), _read(out_dpo))

    assert contents1 == contents2

