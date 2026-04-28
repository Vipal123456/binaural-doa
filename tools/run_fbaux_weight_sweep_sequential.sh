#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/disk2/bywang/DOA-net"
PYTHON_BIN="/home/bywang/miniconda3/envs/doa/bin/python"

CONFIGS=(
  "configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux_w010.yaml"
  "configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux_w015.yaml"
  "configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux_w020.yaml"
)

cd "$ROOT_DIR"

echo "Starting sequential fbaux aux-weight sweep..."
echo "Configs:"
for config in "${CONFIGS[@]}"; do
  echo "  - $config"
done
echo

for config in "${CONFIGS[@]}"; do
  if [[ ! -f "$config" ]]; then
    echo "Config not found: $config"
    exit 1
  fi

  mapfile -t CFG_VALUES < <(
    "$PYTHON_BIN" - <<'PY' "$config"
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print(cfg["output"]["save_dir"])
print(cfg["output"]["log_dir"])
PY
  )

  SAVE_DIR="${CFG_VALUES[0]}"
  LOG_DIR="${CFG_VALUES[1]}"
  mkdir -p "$SAVE_DIR" "$LOG_DIR"

  echo "============================================================"
  echo "Running: $config"
  echo "Checkpoint dir: $SAVE_DIR"
  echo "Log dir: $LOG_DIR"
  echo "Start time: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "============================================================"

  RESUME_ARGS=()
  if [[ -f "$SAVE_DIR/latest.pth" ]]; then
    RESUME_ARGS=(--resume "$SAVE_DIR/latest.pth")
    echo "Resuming from: $SAVE_DIR/latest.pth"
  fi

  "$PYTHON_BIN" -u train.py \
    --config "$config" \
    "${RESUME_ARGS[@]}" \
    --train.num_workers 4

  echo
  echo "Finished: $config"
  echo "End time: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo
done

echo "Sequential fbaux aux-weight sweep completed."
