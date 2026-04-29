#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/disk2/bywang/DOA-net"
PYTHON_BIN="/home/bywang/miniconda3/envs/doa/bin/python"

cd "$ROOT_DIR"

CONFIGS=(
  "configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_nocsl_enhanced_fbaux_only_cohfix.yaml"
  "configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_nols_enhanced_fbaux_only_cohfix.yaml"
  "configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_nocsl_nols_enhanced_fbaux_only_cohfix.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "============================================================"
  echo "Starting training: $cfg"
  echo "Start time: $(date '+%Y-%m-%d %H:%M:%S %z')"
  "$PYTHON_BIN" -u train.py --config "$cfg" --train.num_workers 4
  echo "Finished training: $cfg"
  echo "End time: $(date '+%Y-%m-%d %H:%M:%S %z')"
done
