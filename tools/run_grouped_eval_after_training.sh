#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/bywang/miniconda3/envs/doa/bin/python"
ROOT_DIR="/disk2/bywang/DOA-net"

if [[ $# -ne 7 ]]; then
  echo "Usage: $0 <status_file> <resolved_config> <checkpoint> <oldtest_root> <diffuse_root> <old_output_dir> <diffuse_output_dir>"
  exit 1
fi

STATUS_FILE="$1"
RESOLVED_CONFIG="$2"
CHECKPOINT="$3"
OLDTEST_ROOT="$4"
DIFFUSE_ROOT="$5"
OLD_OUTPUT_DIR="$6"
DIFFUSE_OUTPUT_DIR="$7"

cd "$ROOT_DIR"

echo "[auto-eval] waiting for training to finish..."
while true; do
  if [[ -f "$STATUS_FILE" ]] && grep -q '^exit_code=' "$STATUS_FILE"; then
    EXIT_CODE="$(grep '^exit_code=' "$STATUS_FILE" | tail -n 1 | cut -d= -f2)"
    if [[ "$EXIT_CODE" != "0" ]]; then
      echo "[auto-eval] training finished with exit_code=$EXIT_CODE, skip evaluation."
      exit "$EXIT_CODE"
    fi
    break
  fi
  sleep 60
done

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[auto-eval] checkpoint not found: $CHECKPOINT"
  exit 1
fi

echo "[auto-eval] running old-test grouped evaluation..."
"$PYTHON_BIN" tools/evaluate_kemar_grouped.py \
  --config "$RESOLVED_CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$OLD_OUTPUT_DIR" \
  --test_root "$OLDTEST_ROOT" \
  --batch_size 64 \
  --num_workers 8 \
  --device cuda:5 \
  --log_interval 20

echo "[auto-eval] running diffuse-test grouped evaluation..."
"$PYTHON_BIN" tools/evaluate_kemar_grouped.py \
  --config "$RESOLVED_CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$DIFFUSE_OUTPUT_DIR" \
  --test_root "$DIFFUSE_ROOT" \
  --batch_size 64 \
  --num_workers 8 \
  --device cuda:5 \
  --log_interval 20

echo "[auto-eval] done."
