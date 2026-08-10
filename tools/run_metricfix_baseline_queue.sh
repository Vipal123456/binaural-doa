#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/bywang/miniconda3/envs/doa/bin/python}"
DEVICE="${DEVICE:-cuda:3}"

run_experiment() {
  local config="$1"
  local checkpoint_dir="$2"
  local log_dir="$3"
  local eval_dir="$4"
  local batch_size="$5"
  local num_workers="$6"

  "${PYTHON_BIN}" train.py \
    --config "${config}" \
    --train.device "${DEVICE}" \
    --output.save_dir "${checkpoint_dir}" \
    --output.log_dir "${log_dir}"

  "${PYTHON_BIN}" tools/evaluate_kemar_grouped.py \
    --config "${config}" \
    --checkpoint "${checkpoint_dir}/best.pth" \
    --output_dir "${eval_dir}" \
    --device "${DEVICE}" \
    --num_workers "${num_workers}" \
    --batch_size "${batch_size}"
}

run_experiment \
  configs/train_kemar_sdel_doa_cls_diffusefg_nofb_seed43_g5.yaml \
  outputs/checkpoints_kemar_sdel_doa_cls_diffusefg_nofb_metricfix_seed43_g5 \
  outputs/logs_kemar_sdel_doa_cls_diffusefg_nofb_metricfix_seed43_g5 \
  outputs/grouped_eval_runs/sdel_diffusefg_nofb_metricfix_seed43_g5 \
  64 \
  4

run_experiment \
  configs/train_kemar_dprtf_doa_cls_diffusefg_g5.yaml \
  outputs/checkpoints_kemar_dprtf_doa_cls_diffusefg_metricfix_g5 \
  outputs/logs_kemar_dprtf_doa_cls_diffusefg_metricfix_g5 \
  outputs/grouped_eval_runs/dprtf_diffusefg_metricfix_g5 \
  32 \
  8

run_experiment \
  configs/train_kemar_bilstyle_gccphat_crn72_diffusefg_g7.yaml \
  outputs/checkpoints_kemar_bilstyle_gccphat_crn72_diffusefg_metricfix_g7 \
  outputs/logs_kemar_bilstyle_gccphat_crn72_diffusefg_metricfix_g7 \
  outputs/grouped_eval_runs/bilstyle_gccphat_crn72_diffusefg_metricfix_g7 \
  64 \
  4

run_experiment \
  configs/train_kemar_fnssl_diffusefg_g5_stable.yaml \
  outputs/checkpoints_kemar_fnssl_diffusefg_metricfix_g5_stable \
  outputs/logs_kemar_fnssl_diffusefg_metricfix_g5_stable \
  outputs/grouped_eval_runs/fnssl_diffusefg_metricfix_g5_stable \
  8 \
  4
