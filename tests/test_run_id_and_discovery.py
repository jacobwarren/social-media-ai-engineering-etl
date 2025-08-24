import os
import json
from pipe.utils.run_id import save_last_run_id, get_last_run_id
from pipe.utils.io import resolve_run_id
from pipe.utils.manifest import read_manifest, update_stage, discover_input


def test_latest_resolution(tmp_path):
    base = tmp_path
    save_last_run_id(str(base), "abc123")
    assert get_last_run_id(str(base)) == "abc123"

    # resolve_run_id should map 'latest' to saved id
    rid = resolve_run_id("latest", str(base))
    assert rid == "abc123"


def test_discover_input(tmp_path):
    base_dir = tmp_path
    run_id = "r1"
    os.makedirs(base_dir / run_id, exist_ok=True)
    # create a fake output file from stage A
    out_path = base_dir / run_id / "A.jsonl"
    out_path.write_text("{}\n")

    m = read_manifest(run_id, str(base_dir))
    update_stage(run_id, str(base_dir), m, "A", None, [str(out_path)], signature="sig")

    found = discover_input(read_manifest(run_id, str(base_dir)), "A")
    assert found == str(out_path)

