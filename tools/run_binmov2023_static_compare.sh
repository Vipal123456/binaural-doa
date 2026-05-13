#!/usr/bin/env bash
set -euo pipefail

cd /disk2/bywang/DOA-net

PY=/home/bywang/miniconda3/envs/doa/bin/python
GPU_ID="${1:-1}"

run_train_eval () {
  local config="$1"
  local ckpt="$2"
  local test_log_dir="$3"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PY}" -u train.py --config "${config}" --train.num_workers 8
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PY}" -u evaluate.py \
    --checkpoint "${ckpt}" \
    --config "${config}" \
    --train.num_workers 8 \
    --output.log_dir "${test_log_dir}"
}

run_train_eval \
  "configs/train_binmov2023_static_doanet_mainline_nocsl_fbaux.yaml" \
  "outputs/checkpoints_binmov2023_static_doanet_mainline_nocsl_fbaux/best.pth" \
  "outputs/logs_binmov2023_static_doanet_mainline_nocsl_fbaux_test_best_workers8"

run_train_eval \
  "configs/train_binmov2023_static_sdel_doa_cls_fbaux.yaml" \
  "outputs/checkpoints_binmov2023_static_sdel_doa_cls_fbaux/best.pth" \
  "outputs/logs_binmov2023_static_sdel_doa_cls_fbaux_test_best_workers8"
