#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/bywang/miniconda3/envs/doa/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/data/librispeech_cipic_roomsim25_v1}"
MODE="${MODE:-full}"
WORKERS="${WORKERS:-12}"

cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" -u tools/generate_cipic_roomsim25.py \
  --mode "${MODE}" \
  --output_root "${OUTPUT_ROOT}" \
  --workers "${WORKERS}" \
  --seed 42 \
  --resume \
  "$@"
