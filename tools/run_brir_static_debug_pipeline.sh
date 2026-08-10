#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/bywang/miniconda3/envs/doa/bin/python}
ROOT=${ROOT:-/disk2/bywang/DOA-net}
DATA_ROOT="$ROOT/data/librispeech_cipic_multisubject_brir50h_v1_debug"
CONFIG="$ROOT/configs/train_librispeech_multisubject_brir50h_v1_debug_v7_dualcue.yaml"
RUN_LOG="$ROOT/outputs/logs_brir_static_debug_pipeline.log"
BRIR_MAX_ORDER=${BRIR_MAX_ORDER:-2}
BRIR_AUTO_ORDER=${BRIR_AUTO_ORDER:-0}
BRIR_MAX_AUTO_ORDER_CAP=${BRIR_MAX_AUTO_ORDER_CAP:-8}
BRIR_SECONDS=${BRIR_SECONDS:-1.0}
FIXED_SNR_DB=${FIXED_SNR_DB:-30}
PATH_DEBUG_CSV=${PATH_DEBUG_CSV:-}

cd "$ROOT"
mkdir -p "$ROOT/outputs"

{
  echo "[$(date '+%F %T')] static BRIR debug pipeline start"
  "$PYTHON" tools/diagnostics/check_brir_direct_path_sanity.py \
    --output_dir "$ROOT/outputs/brir_direct_path_sanity"

  GEN_ARGS=(
    --output_root "$DATA_ROOT"
    --train_samples 1000
    --val_samples 200
    --test_samples 200
    --brir_max_order "$BRIR_MAX_ORDER"
    --brir_seconds "$BRIR_SECONDS"
    --overwrite
    --log_interval 100
  )
  if [[ "$BRIR_AUTO_ORDER" == "1" ]]; then
    GEN_ARGS+=(--brir_auto_order --brir_max_auto_order_cap "$BRIR_MAX_AUTO_ORDER_CAP")
  fi
  if [[ "$FIXED_SNR_DB" != "random" ]]; then
    GEN_ARGS+=(--fixed_snr_db "$FIXED_SNR_DB")
  fi
  if [[ -n "$PATH_DEBUG_CSV" ]]; then
    GEN_ARGS+=(--path_debug_csv "$PATH_DEBUG_CSV")
  fi

  "$PYTHON" prepare_brir_debug_dataset.py "${GEN_ARGS[@]}"

  "$PYTHON" - <<'PY'
import csv
from pathlib import Path
import numpy as np
root = Path("data/librispeech_cipic_multisubject_brir50h_v1_debug")
for split in ["train_subjects", "val_subjects", "test_subjects_unseen"]:
    rows = list(csv.DictReader((root / split / "mixing_report.csv").open()))
    snr = np.array([float(r["snr_db"]) for r in rows])
    rt60 = np.array([float(r["target_rt60"]) for r in rows])
    est = np.array([float(r["estimated_rt60"]) for r in rows if r["estimated_rt60"]])
    paths = np.array([int(r["num_paths"]) for r in rows if r["num_paths"]])
    labels = np.bincount([int(r["doa_class"]) for r in rows], minlength=72)
    modes = sorted({r["rendering_mode"] for r in rows})
    print(split, "n=", len(rows),
          "rendering_mode=", modes,
          "snr=", (round(float(snr.min()), 2), round(float(snr.max()), 2)),
          "target_rt60=", (round(float(rt60.min()), 3), round(float(rt60.max()), 3)),
          "est_rt60=", (round(float(est.min()), 3), round(float(est.max()), 3)) if len(est) else None,
          "paths=", (int(paths.min()), int(paths.max())) if len(paths) else None,
          "label_nonzero=", int((labels > 0).sum()))
PY

  "$PYTHON" train.py --config "$CONFIG"
  "$PYTHON" evaluate.py --config "$CONFIG" \
    --checkpoint "$ROOT/outputs/checkpoints_multisubject_brir50h_v1_debug_v7_dualcue/best.pth"
  echo "[$(date '+%F %T')] static BRIR debug pipeline done"
} 2>&1 | tee -a "$RUN_LOG"
