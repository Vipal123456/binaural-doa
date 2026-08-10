#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/bywang/miniconda3/envs/doa/bin/python}
ROOT=${ROOT:-/disk2/bywang/DOA-net}
DATA_ROOT="$ROOT/data/librispeech_cipic_moving_robust50h_v1/reverb_moving_v1"
CONFIG="$ROOT/configs/train_moving_reverb_v1_v7_dualcue_seq.yaml"
RUN_LOG="$ROOT/outputs/logs_moving_reverb_v1_pipeline.log"

cd "$ROOT"
mkdir -p "$ROOT/outputs"

{
  echo "[$(date '+%F %T')] moving reverb v1 pipeline start"
  "$PYTHON" prepare_moving_dataset.py \
    --condition reverb_moving \
    --output_root "$DATA_ROOT" \
    --samples_per_train 20000 \
    --samples_per_eval 2000 \
    --rt60_min 0.3 \
    --rt60_max 0.6 \
    --overwrite \
    --log_interval 500

  "$PYTHON" - <<'PY'
import json
from pathlib import Path
from collections import Counter
import numpy as np
root = Path("data/librispeech_cipic_moving_robust50h_v1/reverb_moving_v1")
for split in ["train_subjects", "val_subjects", "test_subjects_unseen"]:
    metas = list((root / split / "metadata_dev").glob("metadata*.json"))
    traj = Counter()
    speed = Counter()
    profiles = Counter()
    rt60 = []
    labels = []
    for p in metas:
        m = json.loads(p.read_text())
        traj[m["trajectory_type"]] += 1
        speed[m["speed_bin"]] += 1
        profiles[m.get("room_profile", "unknown")] += 1
        rt60.append(float(m.get("rt60", 0.0)))
        labels.extend(m["target_label_seq"])
    hist = np.bincount(labels, minlength=72)
    print(split, "n=", len(metas), "traj=", dict(traj), "speed=", dict(speed), "profiles=", dict(profiles))
    print("  rt60 mean/min/max=", round(float(np.mean(rt60)), 3), round(float(np.min(rt60)), 3), round(float(np.max(rt60)), 3))
    print("  label_nonzero=", int((hist > 0).sum()), "label_minmax=", (int(hist.min()), int(hist.max())))
PY

  "$PYTHON" train_moving.py --config "$CONFIG"
  "$PYTHON" evaluate_moving.py --config "$CONFIG" \
    --checkpoint "$ROOT/outputs/checkpoints_moving_reverb_v1_v7_dualcue_seq/best.pth"
  echo "[$(date '+%F %T')] moving reverb v1 pipeline done"
} 2>&1 | tee -a "$RUN_LOG"
