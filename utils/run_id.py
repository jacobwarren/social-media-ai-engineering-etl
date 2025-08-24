import os
from datetime import datetime
import json
from typing import Optional, Dict, Any


def generate_run_id() -> str:
    """Generate a filesystem-friendly timestamp-based run id."""
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def _last_run_file(base_dir: str) -> str:
    return os.path.join(base_dir, ".last_run_id")


def save_last_run_id(base_dir: str, run_id: str) -> None:
    os.makedirs(base_dir, exist_ok=True)
    with open(_last_run_file(base_dir), "w", encoding="utf-8") as f:
        f.write(run_id.strip() + "\n")


def get_last_run_id(base_dir: str) -> Optional[str]:
    path = _last_run_file(base_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip() or None
    except Exception:
        return None


def write_run_metadata(base_dir: str, run_id: str, data: Dict[str, Any]) -> None:
    """Write or update metadata file under the run directory."""
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    meta_path = os.path.join(run_dir, "_run.json")
    existing: Dict[str, Any] = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                existing = json.load(f) or {}
        except Exception:
            existing = {}
    existing.update(data)
    if "created_at" not in existing:
        existing["created_at"] = datetime.now().isoformat()
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

