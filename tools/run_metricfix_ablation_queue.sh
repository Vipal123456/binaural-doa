#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/bywang/miniconda3/envs/doa/bin/python}"
DEVICE="${DEVICE:-cuda:5}"
BATCH_SIZE=64
NUM_WORKERS=8

evaluate_checkpoint() {
  local config="$1"
  local checkpoint="$2"
  local eval_dir="$3"

  "${PYTHON_BIN}" tools/evaluate_kemar_grouped.py \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --output_dir "${eval_dir}" \
    --device "${DEVICE}" \
    --num_workers "${NUM_WORKERS}" \
    --batch_size "${BATCH_SIZE}"
}

run_ablation() {
  local config="$1"
  local run_name="$2"
  local checkpoint_dir="outputs/checkpoints_kemar_${run_name}"
  local log_dir="outputs/logs_kemar_${run_name}"
  local eval_dir="outputs/grouped_eval_runs/${run_name}"

  "${PYTHON_BIN}" train.py \
    --config "${config}" \
    --train.device "${DEVICE}" \
    --output.save_dir "${checkpoint_dir}" \
    --output.log_dir "${log_dir}"

  evaluate_checkpoint \
    "${config}" \
    "${checkpoint_dir}/best.pth" \
    "${eval_dir}"
}

# Finish the requested old-vs-new seed42 comparison before occupying GPU5.
evaluate_checkpoint \
  configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_metricfix_seed42_g5.yaml \
  outputs/checkpoints_kemar_v7_dualcue_liteenc_v1_diffusefg_metricfix_seed42_g5/best.pth \
  outputs/grouped_eval_runs/v7_dualcue_liteenc_v1_diffusefg_metricfix_seed42_g5

# Core paper ablations: each variant uses the same three seeds as the full model.
run_ablation \
  configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_norel_g4.yaml \
  v7_dualcue_liteenc_v1_diffusefg_norel_metricfix_seed42
run_ablation \
  configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_norel_seed43_g5.yaml \
  v7_dualcue_liteenc_v1_diffusefg_norel_metricfix_seed43
run_ablation \
  configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_norel_seed44_g5.yaml \
  v7_dualcue_liteenc_v1_diffusefg_norel_metricfix_seed44

run_ablation \
  configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_nocontent_g5.yaml \
  v7_dualcue_liteenc_v1_diffusefg_nocontent_metricfix_seed42
run_ablation \
  configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_nocontent_seed43_g6.yaml \
  v7_dualcue_liteenc_v1_diffusefg_nocontent_metricfix_seed43
run_ablation \
  configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_nocontent_seed44_g6.yaml \
  v7_dualcue_liteenc_v1_diffusefg_nocontent_metricfix_seed44

run_ablation \
  configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_mergedcue_g6.yaml \
  v7_dualcue_liteenc_v1_diffusefg_mergedcue_metricfix_seed42
run_ablation \
  configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_mergedcue_seed43_g4.yaml \
  v7_dualcue_liteenc_v1_diffusefg_mergedcue_metricfix_seed43
run_ablation \
  configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_mergedcue_seed44_g4.yaml \
  v7_dualcue_liteenc_v1_diffusefg_mergedcue_metricfix_seed44
