from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ArtifactNames:
    # JSONL
    STAGE01_BEST_POSTS: str = "01-best-posts.jsonl"
    STAGE02_LABELED: str = "02-labeled.jsonl"
    STAGE03_STRUCTURES: str = "03-structures.jsonl"
    STAGE17_STYLE: str = "17-posts-with-writing-style.jsonl"
    STAGE17_PARTIAL: str = "17-partial-results.jsonl"
    STAGE18_PROMPTS: str = "18-with-prompts.jsonl"
    # CSV (prefixed with stage for standardized outputs)
    STAGE22_DATASET: str = "22-ready-dataset.csv"
    STAGE23_BALANCED: str = "23-balanced-dataset.csv"
    STAGE23_SFT: str = "23-sft.csv"
    STAGE23_DPO: str = "23-dpo.csv"


ARTIFACTS: Dict[int, Dict[str, str]] = {
    1: {"std": ArtifactNames.STAGE01_BEST_POSTS},
    2: {"std": ArtifactNames.STAGE02_LABELED},
    3: {"std": ArtifactNames.STAGE03_STRUCTURES},
    17: {"std": ArtifactNames.STAGE17_STYLE, "partial": ArtifactNames.STAGE17_PARTIAL},
    18: {"std": ArtifactNames.STAGE18_PROMPTS},
    22: {"std": ArtifactNames.STAGE22_DATASET},
    23: {
        "balanced": ArtifactNames.STAGE23_BALANCED,
        "sft": ArtifactNames.STAGE23_SFT,
        "dpo": ArtifactNames.STAGE23_DPO,
    },
}

