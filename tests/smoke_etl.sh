#!/usr/bin/env bash
set -euo pipefail

# This smoke test assumes it is run from the 'pipe' directory.
# It will:
#  1) Run stage 1 without --run-id (auto-generate run id)
#  2) Run stage 2 with --run-id latest
#  3) Run stage 3 with --run-id latest
#  4) Verify that outputs exist
#  5) Optionally run stage 5 (balance) with --run-id latest and verify output

BASE_DIR="data/processed"
DATASET="${DATASET:-../course-dataset-clean.jsonl}"
RUN_STAGE5=${RUN_STAGE5:-0}  # set to 1 to run stage 5 balance as part of smoke

if [[ ! -f "$DATASET" ]]; then
  echo "[ERROR] Dataset not found at $DATASET. Set DATASET env var or place ../course-dataset-clean.jsonl."
  exit 1
fi

echo "[SMOKE] Running stage 1 (auto run_id)..."
python 1-find-gradient.py --input "$DATASET"

if [[ ! -f "$BASE_DIR/.last_run_id" ]]; then
  echo "[ERROR] .last_run_id was not created under $BASE_DIR"
  exit 1
fi
RUN_ID=$(cat "$BASE_DIR/.last_run_id")
echo "[SMOKE] Latest run id: $RUN_ID"

echo "[SMOKE] Running stage 2 (latest)..."
python 2-label.py --run-id latest

echo "[SMOKE] Running stage 3 (latest)..."
python 3-extract-structures.py --run-id latest --report

OUT03="$BASE_DIR/$RUN_ID/03-structures.jsonl"
if [[ ! -s "$OUT03" ]]; then
  echo "[ERROR] Output not found or empty: $OUT03"
  exit 1
fi

COUNT=$(wc -l < "$OUT03" | tr -d ' ')
echo "[SMOKE] Stage 3 produced $COUNT lines at $OUT03"

REPORT_DIR="reports/$RUN_ID/structures"
if [[ -d "$REPORT_DIR" ]]; then
  echo "[SMOKE] Report directory exists: $REPORT_DIR"
else
  echo "[WARN] Report directory not found (this is OK if --report not set in stage 3)"
fi

if [[ "$RUN_STAGE5" == "1" ]]; then
  echo "[SMOKE] Running stage 5 (balance, latest)..."
  python 5-balance.py --run-id latest --debug
  OUT05="$BASE_DIR/$RUN_ID/05-balanced.jsonl"
  if [[ ! -s "$OUT05" ]]; then
    echo "[ERROR] Stage 5 output not found or empty: $OUT05"
    exit 1
  fi
  C5=$(wc -l < "$OUT05" | tr -d ' ')
  echo "[SMOKE] Stage 5 produced $C5 lines at $OUT05"
fi

echo "[SMOKE] ETL 1->2->3${RUN_STAGE5:+->5} smoke test completed successfully."

