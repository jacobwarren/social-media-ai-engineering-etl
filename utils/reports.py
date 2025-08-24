import json
import os
from typing import Dict, Any


def _reports_dir(run_id: str) -> str:
    return os.path.join("reports", run_id)


def write_summary(run_id: str, section: str, metrics: Dict[str, Any]) -> str:
    """Append or create a run summary JSON under reports/{run_id}/summary.json.

    Sections are merged by top-level key.
    Returns the path to the summary file.
    """
    rep_dir = _reports_dir(run_id)
    os.makedirs(rep_dir, exist_ok=True)
    path = os.path.join(rep_dir, "summary.json")
    data: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data[section] = metrics
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass
    return path

