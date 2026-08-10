#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/bywang/miniconda3/envs/doa/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/data/librispeech_cipic_roomsim25_anf_nonstationary_v3}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/outputs/logs_cipic_roomsim25_anf_nonstationary_v3_generation}"
WORKERS_PER_SPLIT="${WORKERS_PER_SPLIT:-12}"

mkdir -p "${LOG_DIR}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib-doa

cd "${ROOT_DIR}"
"${PYTHON_BIN}" -u tools/generate_cipic_roomsim25_anf_nonstationary_v3.py \
  --output_root "${OUTPUT_ROOT}" \
  --mode full \
  --workers "${WORKERS_PER_SPLIT}" \
  --parallel_splits \
  "$@" 2>&1 | tee -a "${LOG_DIR}/generate.log"
