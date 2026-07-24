#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/bywang/miniconda3/envs/doa/bin/python}"
CONDA_LIB="${CONDA_LIB:-/home/bywang/miniconda3/envs/doa/lib}"
LIBRISPEECH_ROOT="${LIBRISPEECH_ROOT:-/disk2/bywang/data/LibriSpeech}"
OUT_BASE="${OUT_BASE:-/disk2/bywang/DOA-net/data}"
LOG_DIR="${LOG_DIR:-/disk2/bywang/DOA-net/outputs/logs_dataset_generation_officialsplit}"

TRAIN_OUT="${TRAIN_OUT:-${OUT_BASE}/kemar_sofamyroom/train_20h_minimal_diffusefg_officialsplit}"
VAL_OUT="${VAL_OUT:-${OUT_BASE}/kemar_sofamyroom/val_4h_diffusefg_officialsplit}"
TEST_OUT="${TEST_OUT:-${OUT_BASE}/kemar_sofamyroom_diffusefg_static_v1_test_officialsplit}"

export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"

cd "${ROOT_DIR}"
mkdir -p "${LOG_DIR}"

case "${1:-all}" in
  train)
    "${PYTHON_BIN}" tools/generate_kemar_sofamyroom_dataset.py \
      --output_root "${TRAIN_OUT}" \
      --split train \
      --num_samples 36000 \
      --librispeech_root "${LIBRISPEECH_ROOT}/train-clean-100" \
      --noise_mode diffusefg \
      --save_mode train_minimal \
      --log_interval 100 \
      --overwrite
    ;;
  val)
    "${PYTHON_BIN}" tools/generate_kemar_sofamyroom_dataset.py \
      --output_root "${VAL_OUT}" \
      --split val \
      --num_samples 7200 \
      --librispeech_root "${LIBRISPEECH_ROOT}/LibriSpeech_dev/dev-clean" \
      --noise_mode diffusefg \
      --save_mode train_minimal \
      --log_interval 100 \
      --overwrite
    ;;
  test)
    "${PYTHON_BIN}" tools/generate_kemar_sofamyroom_dataset.py \
      --output_root "${TEST_OUT}" \
      --split test \
      --test_grid \
      --num_samples 0 \
      --librispeech_root "${LIBRISPEECH_ROOT}/LibriSpeech_test/test-clean" \
      --noise_mode diffusefg \
      --save_mode train_minimal \
      --log_interval 100 \
      --overwrite
    ;;
  check)
    "${PYTHON_BIN}" tools/check_kemar_sofamyroom_dataset.py \
      --dataset_roots "${TRAIN_OUT}" "${VAL_OUT}" "${TEST_OUT}" \
      --check_audio \
      --output_json "${LOG_DIR}/officialsplit_sanity_summary.json"
    "${PYTHON_BIN}" tools/check_kemar_speech_overlap.py \
      --metadata \
        "train=${TRAIN_OUT}/train/metadata.csv" \
        "val=${VAL_OUT}/val/metadata.csv" \
        "test=${TEST_OUT}/test/metadata.csv" \
      --output_json "${LOG_DIR}/officialsplit_speech_overlap.json"
    ;;
  all)
    "$0" train 2>&1 | tee "${LOG_DIR}/train20h_officialsplit.log"
    "$0" val 2>&1 | tee "${LOG_DIR}/val4h_officialsplit.log"
    "$0" test 2>&1 | tee "${LOG_DIR}/testgrid_officialsplit.log"
    "$0" check 2>&1 | tee "${LOG_DIR}/check_officialsplit.log"
    ;;
  *)
    echo "Usage: $0 {train|val|test|check|all}" >&2
    exit 2
    ;;
esac
