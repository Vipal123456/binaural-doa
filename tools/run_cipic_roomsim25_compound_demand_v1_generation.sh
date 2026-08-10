#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/bywang/miniconda3/envs/doa/bin/python}"
MODE="${MODE:-full}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/data/librispeech_cipic_roomsim25_compound_demand_v1}"
WORKERS="${WORKERS:-8}"

cd "${ROOT_DIR}"
exec env MPLCONFIGDIR=/tmp/mpl-doa-compound "${PYTHON_BIN}" -u \
  tools/generate_cipic_roomsim25_compound_demand_v1.py \
  --mode "${MODE}" \
  --output_root "${OUTPUT_ROOT}" \
  --workers "${WORKERS}" \
  "$@"
