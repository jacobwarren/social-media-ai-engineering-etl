import os
import json
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional


def _manifest_path(base_dir: str, run_id: str) -> str:
    return os.path.join(base_dir, run_id, "manifest.json")


def read_manifest(run_id: str, base_dir: str) -> Dict[str, Any]:
    """Load manifest or return a fresh structure."""
    path = _manifest_path(base_dir, run_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # Corrupt manifest; start fresh
            pass
    return {"run_id": run_id, "stages": {}}


def write_manifest(run_id: str, base_dir: str, manifest: Dict[str, Any]) -> None:
    os.makedirs(os.path.join(base_dir, run_id), exist_ok=True)
    path = _manifest_path(base_dir, run_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def compute_hash(paths: List[str], config: Optional[Dict[str, Any]] = None) -> str:
    """Compute a hash over file contents and a config dict."""
    h = hashlib.sha256()
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            # Stream to handle large files
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    if config is not None:
        try:
            payload = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
            h.update(payload)
        except Exception:
            # Best-effort: string repr
            h.update(str(config).encode("utf-8"))
    return h.hexdigest()


def outputs_exist(outputs: List[str]) -> bool:
    return all(os.path.exists(p) for p in outputs if p)


def should_skip(manifest: Dict[str, Any], stage_name: str, signature: str, outputs: List[str]) -> bool:
    stage = manifest.get("stages", {}).get(stage_name)
    if not stage:
        return False
    if stage.get("signature") != signature:
        return False
    return outputs_exist(outputs)


def update_stage(
    run_id: str,
    base_dir: str,
    manifest: Dict[str, Any],
    stage_name: str,
    input_path: Optional[str],
    outputs: List[str],
    signature: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    stages = manifest.setdefault("stages", {})
    stages[stage_name] = {
        "input": input_path,
        "outputs": [p for p in outputs if p],
        "signature": signature,
        "completed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        **(extra or {}),
    }
    write_manifest(run_id, base_dir, manifest)


def discover_input(manifest: Dict[str, Any], prior_stage: str) -> Optional[str]:
    stage = manifest.get("stages", {}).get(prior_stage)
    if not stage:
        return None
    outs = stage.get("outputs") or []
    return outs[0] if outs else None

