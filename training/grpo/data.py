from __future__ import annotations
from typing import Tuple

from datasets import load_dataset, Dataset  # type: ignore

from utils.manifest import read_manifest, discover_input


def prepare_grpo_datasets(
    tokenizer,
    run_id: str | None,
    base_dir: str,
    csv_fallback: str = "dpo.csv",
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Discover the DPO-ready CSV via manifest and prepare GRPO datasets.
    Mirrors the logic in 26-train-grpo.py:get_balanced_sft + split.
    """
    # Discover CSV via manifest if run-id provided, else fallback
    csv_path = csv_fallback
    try:
        if run_id:
            m = read_manifest(run_id, base_dir)
            discovered = discover_input(m, "24-add-negatives") or discover_input(m, "23-split")
            if isinstance(discovered, list):
                preferred = [p for p in discovered if p.endswith("24-dpo-ready.csv")] or [p for p in discovered if p.endswith("23-dpo.csv")] or discovered
                csv_path = preferred[0]
            elif isinstance(discovered, str):
                csv_path = discovered
    except Exception:
        pass

    data = load_dataset("csv", data_files=csv_path)["train"]

    # Filter for rows that have a 'prompt' and 'chosen'
    def _filter_samples(sample):
        return sample.get("prompt") is not None and sample.get("chosen") is not None

    data = data.filter(_filter_samples)

    def _map_to_grpo(sample):
        chat_prompt = [
            {"role": "user", "content": sample["prompt"]},
        ]
        combined_string = tokenizer.apply_chat_template(
            chat_prompt, tokenize=False, add_generation_prompt=True
        )
        return {
            "prompt": combined_string,
            "answer": sample["chosen"],
        }

    data = data.map(_map_to_grpo)

    split = data.train_test_split(test_size=test_size, seed=seed, shuffle=True)
    return split["train"], split["test"]

