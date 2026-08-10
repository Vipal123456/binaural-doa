#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk2/bywang/DOA-net"
PYTHON="/home/bywang/miniconda3/envs/doa/bin/python"
CONFIG="configs/train_cipic_roomsim25_directional_dns_v4_v7_localtf32_cftr_cpsd5_maskonly_bestmae_seed42.yaml"
CHECKPOINT="outputs/checkpoints_cipic_roomsim25_directional_dns_v4_v7_localtf32_cftr_cpsd5_maskonly_bestmae_seed42/best_mae.pth"
VAL_ROOT="data/librispeech_cipic_roomsim25_directional_dns_v4/val"
OUTPUT_ROOT="outputs/cftr_cpsd_q_ablation_val"

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT"

run_mode() {
  local label="$1"
  local bias_mode="$2"
  echo "[q-ablation] start ${label} (${bias_mode})"
  "$PYTHON" -u tools/evaluate_kemar_grouped.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --test_root "$VAL_ROOT" \
    --output_dir "$OUTPUT_ROOT/$label" \
    --device cuda:0 \
    --batch_size 64 \
    --num_workers 8 \
    --log_interval 50 \
    --component_supervision \
    --cue_target_bias_mode "$bias_mode"
}

run_mode disabled disabled
run_mode predicted shared_unit
run_mode oracle oracle_shared

echo "[q-ablation] complete"
