#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/bywang/miniconda3/envs/doa/bin/python}
ROOT=${ROOT:-/disk2/bywang/DOA-net}
DATA_ROOT=${DATA_ROOT:-$ROOT/data/librispeech_cipic_multisubject_brir50h_order2_v1}
RUN_LOG=${RUN_LOG:-$ROOT/outputs/logs_brir_static_robust50h_order2_generation.log}

cd "$ROOT"
mkdir -p "$ROOT/outputs"

{
  echo "[$(date '+%F %T')] static BRIR robust50h order2 generation start"
  echo "Dataset: $DATA_ROOT"
  echo "Settings:"
  echo "  speech=LibriSpeech train-clean-100"
  echo "  hrtf=CIPIC SOFA, robust50h subject-disjoint split 24/3/3"
  echo "  duration=10s, sr=16k, classes=72, azimuth=[-180,180), bin=5deg"
  echo "  source_distance=[1.0,1.5]m"
  echo "  room_profiles=small/medium/large with RT60 ranges copied from robust50h"
  echo "  rendering_mode=image_source_pathwise_hrtf_brir"
  echo "  brir_max_order=2, brir_seconds=1.0, auto_order=false"
  echo "  noise=DEMAND, SNR uniform [-10,10] dB"
  echo "  counts=train 14400 / val 1800 / test 1800"

  "$PYTHON" tools/diagnostics/check_brir_direct_path_sanity.py \
    --output_dir "$ROOT/outputs/brir_direct_path_sanity_order2_formal"

  GEN_MODE=(--overwrite)
  if [[ -d "$DATA_ROOT" ]]; then
    GEN_MODE=(--resume)
  fi

  "$PYTHON" prepare_brir_debug_dataset.py \
    --dataset_name "librispeech_cipic_multisubject_brir50h_order2_v1" \
    --output_root "$DATA_ROOT" \
    --train_samples 14400 \
    --val_samples 1800 \
    --test_samples 1800 \
    --brir_max_order 2 \
    --brir_seconds 1.0 \
    "${GEN_MODE[@]}" \
    --log_interval 200

  "$PYTHON" - <<'PY'
import csv
import json
from pathlib import Path

import numpy as np

root = Path("data/librispeech_cipic_multisubject_brir50h_order2_v1")
summary = {}
for split in ["train_subjects", "val_subjects", "test_subjects_unseen"]:
    rows = list(csv.DictReader((root / split / "mixing_report.csv").open()))
    snr = np.array([float(r["snr_db"]) for r in rows])
    rt60 = np.array([float(r["target_rt60"]) for r in rows])
    est = np.array([float(r["estimated_rt60"]) for r in rows if r["estimated_rt60"]])
    paths = np.array([int(r["num_paths"]) for r in rows if r["num_paths"]])
    labels = np.bincount([int(r["doa_class"]) for r in rows], minlength=72)
    subjects = sorted({r["subject_id"] for r in rows})
    summary[split] = {
        "n": len(rows),
        "subjects": subjects,
        "snr_min_max": [float(snr.min()), float(snr.max())],
        "target_rt60_min_max": [float(rt60.min()), float(rt60.max())],
        "estimated_rt60_min_median_max": [float(est.min()), float(np.median(est)), float(est.max())] if len(est) else None,
        "num_paths_min_max": [int(paths.min()), int(paths.max())] if len(paths) else None,
        "label_nonzero": int((labels > 0).sum()),
        "label_min_max_count": [int(labels.min()), int(labels.max())],
    }
    print(split, json.dumps(summary[split], ensure_ascii=False), flush=True)
(root / "generation_quality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
PY

  echo "[$(date '+%F %T')] static BRIR robust50h order2 generation done"
} 2>&1 | tee -a "$RUN_LOG"
