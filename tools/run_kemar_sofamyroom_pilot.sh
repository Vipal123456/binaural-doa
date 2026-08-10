#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/bywang/miniconda3/envs/doa/bin/python}"
CONDA_LIB="${CONDA_LIB:-/home/bywang/miniconda3/envs/doa/lib}"
OUTPUT_ROOT="${1:-outputs/kemar_sofamyroom_pilot}"

export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"

cd "${ROOT_DIR}"

echo "[pilot] output_root=${OUTPUT_ROOT}"

"${PYTHON_BIN}" tools/generate_kemar_sofamyroom_dataset.py \
  --output_root "${OUTPUT_ROOT}_train" \
  --split train \
  --num_samples 720 \
  --overwrite \
  --log_interval 24

"${PYTHON_BIN}" tools/generate_kemar_sofamyroom_dataset.py \
  --output_root "${OUTPUT_ROOT}_val" \
  --split val \
  --num_samples 144 \
  --overwrite \
  --log_interval 24

"${PYTHON_BIN}" tools/generate_kemar_sofamyroom_dataset.py \
  --output_root "${OUTPUT_ROOT}_testgrid432" \
  --split test \
  --test_grid \
  --num_samples 432 \
  --overwrite \
  --log_interval 24

"${PYTHON_BIN}" tools/check_kemar_sofamyroom_dataset.py \
  --dataset_roots "${OUTPUT_ROOT}_train" "${OUTPUT_ROOT}_val" "${OUTPUT_ROOT}_testgrid432" \
  --check_audio \
  --output_json "${OUTPUT_ROOT}_sanity_summary.json"

echo "[pilot] done"
echo "[pilot] sanity summary: ${OUTPUT_ROOT}_sanity_summary.json"
