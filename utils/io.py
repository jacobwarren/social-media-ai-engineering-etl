from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple, List, Dict

from .manifest import read_manifest, discover_input
from .run_id import get_last_run_id


class IOResolutionError(RuntimeError):
    pass


def resolve_run_id(run_id: Optional[str], base_dir: str) -> Optional[str]:
    """Resolve a run_id; supports the special value 'latest'.
    Returns the resolved run_id or None if not provided.
    """
    if not run_id:
        return None
    if run_id == "latest":
        rid = get_last_run_id(base_dir)
        if not rid:
            raise IOResolutionError("No .last_run_id found; run previous stages first or provide --run-id")
        return rid
    return run_id


def resolve_input_path(
    *,
    run_id: Optional[str],
    base_dir: str,
    explicit_input: Optional[str],
    prior_stages: Iterable[str] | None,
) -> str:
    """Resolve the input path for a stage.

    Rules:
    - If run_id is provided, prefer manifest discovery from the given prior_stages list.
    - If discovery fails, fall back to explicit_input if provided.
    - If run_id is not provided, require explicit_input.
    """
    rid = resolve_run_id(run_id, base_dir)
    if rid:
        manifest = read_manifest(rid, base_dir)
        if prior_stages:
            for stage in prior_stages:
                path = discover_input(manifest, stage)
                if path and os.path.exists(path):
                    return path
        # Fallback: if explicit input provided, use it
        if explicit_input:
            return explicit_input
        raise IOResolutionError(
            "No input discovered from manifest; provide --input or ensure prior stage outputs exist"
        )
    # No run_id path
    if not explicit_input:
        raise IOResolutionError("When --run-id is not provided, you must specify --input")
    return explicit_input


def std_dir(base_dir: str, run_id: str) -> str:
    out = os.path.join(base_dir, run_id)
    os.makedirs(out, exist_ok=True)
    return out


def std_paths(base_dir: str, run_id: str, filenames: Iterable[str]) -> List[str]:
    d = std_dir(base_dir, run_id)
    return [os.path.join(d, name) for name in filenames]


def ensure_explicit_outputs_when_no_runid(
    *,
    run_id: Optional[str],
    outputs: Dict[str, Optional[str]],
) -> None:
    """Raise if any required output is missing when run_id is not provided.

    Pass outputs as a mapping of label -> path.
    """
    if run_id:
        return
    missing = [k for k, v in outputs.items() if not v]
    if missing:
        raise IOResolutionError(
            f"When --run-id is not provided, you must specify output paths for: {', '.join(missing)}"
        )




def resolve_io(
    *,
    stage: str,
    run_id: Optional[str],
    base_dir: str,
    explicit_in: Optional[str] = None,
    prior_stage: Optional[Iterable[str] | str] = None,
    std_name: Optional[str] = None,
):
    """Resolve input and standardized output for a stage.

    - If run_id is provided, prefer discovering input from manifest using prior_stage.
      prior_stage may be a single stage name or an iterable of stage names to try in order.
    - Ensure standardized output directory exists under base_dir/run_id and return std path.
    - If run_id is None, require explicit_in and return (explicit_in, None, None).
    """
    rid = resolve_run_id(run_id, base_dir)
    if rid:
        manifest = read_manifest(rid, base_dir)
        inp = explicit_in
        # Support multiple prior stages to try in order
        if prior_stage:
            stages_to_try: List[str] = []
            if isinstance(prior_stage, (list, tuple, set)):
                stages_to_try = list(prior_stage)
            else:
                stages_to_try = [prior_stage]  # type: ignore[list-item]
            for stage_name in stages_to_try:
                discovered = discover_input(manifest, stage_name)
                if discovered and os.path.exists(discovered):
                    inp = discovered
                    break
        if not inp:
            raise IOResolutionError("No input found: provide --input or ensure prior stage outputs exist")
        out_dir = Path(base_dir) / rid
        out_dir.mkdir(parents=True, exist_ok=True)
        std_out = out_dir / (std_name or f"{stage}.out")
        return str(inp), str(std_out), rid
    # No run_id scenario
    if not explicit_in:
        raise IOResolutionError("When --run-id is not provided, you must specify --input")
    return explicit_in, None, None
