#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/bywang/miniconda3/envs/doa/bin/python}"
CONDA_LIB="${CONDA_LIB:-/home/bywang/miniconda3/envs/doa/lib}"
OUTPUT_ROOT="${1:-/disk2/bywang/DOA-net/data/kemar_sofamyroom/train_20h_minimal}"
SUMMARY_JSON="${OUTPUT_ROOT}_sanity_summary.json"

export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"

cd "${ROOT_DIR}"
mkdir -p "$(dirname "${OUTPUT_ROOT}")"

echo "[train20h] output_root=${OUTPUT_ROOT}"
echo "[train20h] started at $(date '+%F %T')"

"${PYTHON_BIN}" tools/generate_kemar_sofamyroom_dataset.py \
  --output_root "${OUTPUT_ROOT}" \
  --split train \
  --num_samples 36000 \
  --save_mode train_minimal \
  --log_interval 100 \
  --overwrite

"${PYTHON_BIN}" tools/check_kemar_sofamyroom_dataset.py \
  --dataset_roots "${OUTPUT_ROOT}" \
  --check_audio \
  --output_json "${SUMMARY_JSON}"

echo "[train20h] finished at $(date '+%F %T')"
echo "[train20h] sanity summary: ${SUMMARY_JSON}"
