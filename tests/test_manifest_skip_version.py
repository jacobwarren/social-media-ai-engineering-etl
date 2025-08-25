import os
import json
import tempfile
from utils.manifest import compute_hash, should_skip, read_manifest, update_stage


def test_should_not_skip_when_stage_version_changes(tmp_path):
    base_dir = tmp_path
    run_id = "run"
    os.makedirs(base_dir / run_id, exist_ok=True)

    # Create a dummy input file
    input_file = base_dir / run_id / "input.txt"
    input_file.write_text("hello")

    # First signature with stage_version=1
    sig_v1 = compute_hash([str(input_file)], {"stage": 99, "stage_version": 1})

    # Write manifest stage
    manifest = read_manifest(run_id, str(base_dir))
    update_stage(run_id, str(base_dir), manifest, "99-stage", str(input_file), [str(input_file)], sig_v1)

    # New signature with stage_version=2
    sig_v2 = compute_hash([str(input_file)], {"stage": 99, "stage_version": 2})

    # should_skip must be False because signature changed
    man2 = read_manifest(run_id, str(base_dir))
    assert not should_skip(man2, "99-stage", sig_v2, [str(input_file)])

