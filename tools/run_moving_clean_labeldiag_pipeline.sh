#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/bywang/miniconda3/envs/doa/bin/python}
ROOT=${ROOT:-/disk2/bywang/DOA-net}
DATA_ROOT="$ROOT/data/librispeech_cipic_moving_clean_labeldiag"
TARGET_CONFIG="$ROOT/configs/train_moving_clean_labeldiag_target_v7_dualcue_seq.yaml"
RENDERED_CONFIG="$ROOT/configs/train_moving_clean_labeldiag_rendered_v7_dualcue_seq.yaml"
RUN_LOG="$ROOT/outputs/logs_moving_clean_labeldiag_pipeline.log"

cd "$ROOT"
mkdir -p "$ROOT/outputs"

{
  echo "[$(date '+%F %T')] moving clean labeldiag pipeline start"
  "$PYTHON" prepare_moving_dataset.py \
    --output_root "$DATA_ROOT" \
    --samples_per_train 5000 \
    --samples_per_eval 500 \
    --overwrite \
    --log_interval 500

  "$PYTHON" - <<'PY'
import json
from pathlib import Path
from collections import Counter
import numpy as np

def wrap(x):
    return ((x + 180.0) % 360.0) - 180.0

root = Path("data/librispeech_cipic_moving_clean_labeldiag")
for split in ["train_subjects", "val_subjects", "test_subjects_unseen"]:
    metas = list((root / split / "metadata_dev").glob("metadata*.json"))
    traj = Counter()
    speed = Counter()
    target_labels = []
    rendered_labels = []
    mismatch = []
    for p in metas:
        m = json.loads(p.read_text())
        traj[m["trajectory_type"]] += 1
        speed[m["speed_bin"]] += 1
        target = np.asarray(m["target_angle_seq"], dtype=np.float64)
        rendered = np.asarray(m["rendered_angle_seq"], dtype=np.float64)
        mismatch.extend(np.abs(wrap(target - rendered)).tolist())
        target_labels.extend(m["target_label_seq"])
        rendered_labels.extend(m["rendered_label_seq"])
    th = np.bincount(target_labels, minlength=72)
    rh = np.bincount(rendered_labels, minlength=72)
    print(split, "n=", len(metas), "traj=", dict(traj), "speed=", dict(speed))
    print("  target_label_nonzero=", int((th > 0).sum()), "target_minmax=", (int(th.min()), int(th.max())))
    print("  rendered_label_nonzero=", int((rh > 0).sum()), "rendered_minmax=", (int(rh.min()), int(rh.max())))
    print("  target_render_mismatch_deg mean/p50/p95/max=",
          round(float(np.mean(mismatch)), 3),
          round(float(np.percentile(mismatch, 50)), 3),
          round(float(np.percentile(mismatch, 95)), 3),
          round(float(np.max(mismatch)), 3))
PY

  echo "[$(date '+%F %T')] train label_source=target"
  "$PYTHON" train_moving.py --config "$TARGET_CONFIG"
  "$PYTHON" evaluate_moving.py --config "$TARGET_CONFIG" \
    --checkpoint "$ROOT/outputs/checkpoints_moving_clean_labeldiag_target_v7_dualcue_seq/best.pth"

  echo "[$(date '+%F %T')] train label_source=rendered"
  "$PYTHON" train_moving.py --config "$RENDERED_CONFIG"
  "$PYTHON" evaluate_moving.py --config "$RENDERED_CONFIG" \
    --checkpoint "$ROOT/outputs/checkpoints_moving_clean_labeldiag_rendered_v7_dualcue_seq/best.pth"

  echo "[$(date '+%F %T')] moving clean labeldiag pipeline done"
} 2>&1 | tee -a "$RUN_LOG"
