import json
import os
from typing import Dict, Any

from .manifest import compute_hash


def write_contract(artifact_path: str, schema_version: str, counts: Dict[str, int] | None = None, extra: Dict[str, Any] | None = None) -> str:
    """Write a small data contract JSON next to an artifact.

    Includes: schema_version, file hash signature, counts (e.g., rows), and optional extras.
    Returns the contract path.
    """
    contract = {
        "schema_version": schema_version,
        "signature": compute_hash([artifact_path]),
        "counts": counts or {},
        **(extra or {}),
    }
    path = artifact_path + ".contract.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(contract, f, indent=2)
    except Exception:
        pass
    return path

