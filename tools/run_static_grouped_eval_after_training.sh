#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "Usage: $0 <status_file> <config> <checkpoint> <test_root> <output_dir> <batch_size> <num_workers>"
  exit 1
fi

STATUS_FILE="$1"
CONFIG="$2"
CHECKPOINT="$3"
TEST_ROOT="$4"
OUTPUT_DIR="$5"
BATCH_SIZE="$6"
NUM_WORKERS="$7"
ROOT_DIR="/disk2/bywang/DOA-net"
PYTHON_BIN="/home/bywang/miniconda3/envs/doa/bin/python"

cd "$ROOT_DIR"
while true; do
  if [[ -f "$STATUS_FILE" ]] && grep -q '^exit_code=' "$STATUS_FILE"; then
    EXIT_CODE="$(grep '^exit_code=' "$STATUS_FILE" | tail -n 1 | cut -d= -f2)"
    if [[ "$EXIT_CODE" != "0" ]]; then
      echo "Training failed with exit_code=$EXIT_CODE; skipping evaluation."
      exit "$EXIT_CODE"
    fi
    break
  fi
  sleep 60
done

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
exec "$PYTHON_BIN" -u tools/evaluate_kemar_grouped.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --test_root "$TEST_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --device cuda:0 \
  --log_interval 20 \
  > "$OUTPUT_DIR/eval.log" 2>&1
