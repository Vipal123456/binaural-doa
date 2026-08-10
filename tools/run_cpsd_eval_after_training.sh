#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "Usage: $0 <status_file> <config> <checkpoint> <standard_output> <compound_output> <log_file>"
  exit 1
fi

ROOT="/disk2/bywang/DOA-net"
PYTHON="/home/bywang/miniconda3/envs/doa/bin/python"
STATUS_FILE="$1"
CONFIG="$2"
CHECKPOINT="$3"
STANDARD_OUTPUT="$4"
COMPOUND_OUTPUT="$5"
LOG_FILE="$6"

cd "$ROOT"
mkdir -p "$(dirname "$LOG_FILE")"
exec >> "$LOG_FILE" 2>&1

echo "[auto-eval] waiting for $STATUS_FILE"
while true; do
  if [[ -f "$STATUS_FILE" ]] && grep -q '^exit_code=' "$STATUS_FILE"; then
    EXIT_CODE="$(grep '^exit_code=' "$STATUS_FILE" | tail -n 1 | cut -d= -f2)"
    if [[ "$EXIT_CODE" != "0" ]]; then
      echo "[auto-eval] training exit_code=$EXIT_CODE; evaluation skipped"
      exit "$EXIT_CODE"
    fi
    break
  fi
  sleep 60
done

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[auto-eval] missing checkpoint: $CHECKPOINT"
  exit 1
fi

echo "[auto-eval] standard directional-DNS test"
"$PYTHON" -u tools/evaluate_kemar_grouped.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --test_root data/librispeech_cipic_roomsim25_directional_dns_v4/test \
  --output_dir "$STANDARD_OUTPUT" \
  --device cuda:0 \
  --batch_size 64 \
  --num_workers 8 \
  --log_interval 100

echo "[auto-eval] compound directional-plus-diffuse test"
"$PYTHON" -u tools/evaluate_cipic_compound_grouped.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --test_root data/librispeech_cipic_roomsim25_compound_demand_v1/test \
  --output_dir "$COMPOUND_OUTPUT" \
  --device cuda:0 \
  --batch_size 64 \
  --num_workers 8 \
  --log_interval 100

echo "[auto-eval] complete"
