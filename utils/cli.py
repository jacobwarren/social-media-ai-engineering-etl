from __future__ import annotations
from typing import Optional, Dict

from .io import resolve_run_id, ensure_explicit_outputs_when_no_runid


def add_standard_args(parser, *, include_seed: bool = False, include_reports: bool = False):
    parser.add_argument("--run-id", dest="run_id", default=None, help="Use 'latest' to pick up the most recent run")
    parser.add_argument("--base-dir", dest="base_dir", default="data/processed")
    if include_seed:
        parser.add_argument("--seed", dest="seed", type=int, default=None)
    if include_reports:
        parser.add_argument("--reports-dir", dest="reports_dir", default="reports")
    return parser


def resolve_common_args(args, *, require_input_when_no_run_id: bool = False, required_outputs: Dict[str, Optional[str]] | None = None):
    args.run_id = resolve_run_id(args.run_id, args.base_dir)
    if require_input_when_no_run_id and not args.run_id:
        if not getattr(args, "input_path", None):
            import sys
            print("Error: --input is required when --run-id is not provided", file=sys.stderr)
            sys.exit(1)
    if required_outputs is not None:
        ensure_explicit_outputs_when_no_runid(run_id=args.run_id, outputs=required_outputs)
    return args

