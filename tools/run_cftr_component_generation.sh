#!/usr/bin/env bash
set -euo pipefail

cd /disk2/bywang/DOA-net
LOG_DIR="outputs/logs_cipic_roomsim25_directional_dns_v4_cftr_components"
mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/generate.log" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] starting/resuming train component generation"
/home/bywang/miniconda3/envs/doa/bin/python -u \
  tools/generate_cipic_dns_component_sidecars.py \
  --dataset_root data/librispeech_cipic_roomsim25_directional_dns_v4 \
  --splits train \
  --workers 16 \
  --verify_limit 32
echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] component generation complete"
