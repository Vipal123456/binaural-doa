#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk2/bywang/DOA-net"
PYTHON="/home/bywang/miniconda3/envs/doa/bin/python"
DATA_ROOT="$ROOT/data/librispeech_cipic_moving_hybridbrir_gate2_20h_v1"
LOG="$ROOT/outputs/logs_moving_hybridbrir_gate2_20h_v1_parallel_generation.log"
WORKER_LOG_DIR="$ROOT/outputs/logs_moving_hybridbrir_gate2_20h_v1_workers"

TRAIN_TOTAL=14400
EVAL_TOTAL=1800
TRAIN_JOBS="${TRAIN_JOBS:-8}"
EVAL_JOBS="${EVAL_JOBS:-2}"

TRAIN_SUBJECTS="003,008,011,012,020,021,027,028,033,048,051,058,059,060,061,065,119,124,126,131,133,134,135,147"
VAL_SUBJECTS="152,154,155"
TEST_SUBJECTS="156,163,165"

cd "$ROOT"
mkdir -p "$ROOT/outputs" "$WORKER_LOG_DIR"
rm -rf "$DATA_ROOT"
mkdir -p "$DATA_ROOT"
rm -f "$LOG"
rm -f "$WORKER_LOG_DIR"/*.log

common_args=(
  --condition moving_hybridbrir_gate2
  --output_root "$DATA_ROOT"
  --train_subjects "$TRAIN_SUBJECTS"
  --val_subjects "$VAL_SUBJECTS"
  --test_subjects "$TEST_SUBJECTS"
  --samples_per_train "$TRAIN_TOTAL"
  --samples_per_eval "$EVAL_TOTAL"
  --duration_sec 4.0
  --chunk_seconds 0.1
  --label_steps 40
  --label_source target
  --distance_min 1.0
  --distance_max 1.5
  --brir_max_order 3
  --brir_seconds 1.8
  --early_cut_ms 80
  --late_start_ms 80
  --snr_min -10
  --snr_max 10
  --require_quality_gate
  --max_attempts_per_sample 10
  --append
  --skip_manifest
  --log_interval 100
)

run_shards() {
  local split="$1"
  local total="$2"
  local jobs="$3"
  local base=$((total / jobs))
  local rem=$((total % jobs))
  local start=1
  local pids=()

  for ((job=0; job<jobs; job++)); do
    local count="$base"
    if (( job < rem )); then
      count=$((count + 1))
    fi
    local end=$((start + count - 1))
    local seed=$((42 + job * 1009))
    local worker_log="$WORKER_LOG_DIR/${split}_shard$(printf '%02d' "$job")_${start}_${end}.log"
    {
      echo "[$(date '+%F %T')] start split=$split shard=$job range=$start-$end seed=$seed"
      "$PYTHON" -u prepare_moving_dataset.py \
        "${common_args[@]}" \
        --split_only "$split" \
        --index_start "$start" \
        --index_end "$end" \
        --seed "$seed"
      echo "[$(date '+%F %T')] done split=$split shard=$job range=$start-$end"
    } >"$worker_log" 2>&1 &
    pids+=("$!")
    start=$((end + 1))
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

{
  echo "[$(date '+%F %T')] moving hybrid BRIR gate2 20h parallel generation start"
  echo "data_root=$DATA_ROOT"
  echo "setting=train $TRAIN_TOTAL, val $EVAL_TOTAL, test $EVAL_TOTAL, duration=4s, label_steps=40"
  echo "hours=train 16h, val 2h, test 2h, total 20h"
  echo "parallel=train_jobs=$TRAIN_JOBS, eval_jobs=$EVAL_JOBS"
  echo "subject_split=static_gate2_24_3_3"
  echo "worker_logs=$WORKER_LOG_DIR"

  run_shards train_subjects "$TRAIN_TOTAL" "$TRAIN_JOBS"
  run_shards val_subjects "$EVAL_TOTAL" "$EVAL_JOBS"
  run_shards test_subjects_unseen "$EVAL_TOTAL" "$EVAL_JOBS"

  "$PYTHON" - <<'PY'
import json
from pathlib import Path

root = Path("data/librispeech_cipic_moving_hybridbrir_gate2_20h_v1")
manifest = {
    "dataset": "librispeech_cipic_moving_hybridbrir_gate2_20h_v1",
    "hours": {"train": 16.0, "val": 2.0, "test": 2.0, "total": 20.0},
    "samples": {"train_subjects": 14400, "val_subjects": 1800, "test_subjects_unseen": 1800},
    "duration_sec": 4.0,
    "label_steps": 40,
    "label_source": "target",
    "condition": "moving_hybridbrir_gate2",
    "rendering": {
        "mode": "moving_hybrid_pathwise_hrtf_brir_gate2_v1",
        "early_max_order": 3,
        "early_cut_ms": 80,
        "late_start_ms": 80,
        "brir_seconds": 1.8,
        "quality_gate_required": True,
        "max_attempts_per_sample": 10,
    },
    "trajectory": {"static": 0.20, "linear": 0.60, "piecewise": 0.20},
    "distance_m": [1.0, 1.5],
    "snr_db": [-10.0, 10.0],
    "split": {
        "train_subjects": "003,008,011,012,020,021,027,028,033,048,051,058,059,060,061,065,119,124,126,131,133,134,135,147",
        "val_subjects": "152,154,155",
        "test_subjects_unseen": "156,163,165",
    },
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY

  "$PYTHON" - <<'PY'
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

root = Path("data/librispeech_cipic_moving_hybridbrir_gate2_20h_v1")
print("[diagnostics] root=", root)
for split in ["train_subjects", "val_subjects", "test_subjects_unseen"]:
    metas = sorted((root / split / "metadata_dev").glob("metadata*.json"))
    wavs = sorted((root / split / "binaural_dev").glob("binaural*.wav"))
    traj = Counter()
    subjects = Counter()
    scenes = Counter()
    profiles = Counter()
    label_values = []
    rendered_label_values = []
    quality = []
    attempts = []
    snrs = []
    target_rt60 = []
    estimated_rt60 = []
    drr = []
    mismatch = []
    wav_lengths = []
    by_room = defaultdict(list)
    file_ids = []
    for mp in metas:
        m = json.loads(mp.read_text())
        file_ids.append(int(m["file_id"]))
        traj[m["trajectory_type"]] += 1
        subjects[m["subject_id"]] += 1
        scenes[m["noise_scene"]] += 1
        profiles[m["room_profile"]] += 1
        label_values.extend(m["doa_labels"])
        rendered_label_values.extend(m["rendered_label_seq"])
        quality.append(bool(m["quality_gate_ok"]))
        attempts.append(int(m.get("accepted_attempt", 1)))
        snrs.append(float(m["snr_db"]))
        target_rt60.append(float(m["target_rt60"]))
        estimated_rt60.append(float(m["estimated_rt60"]))
        drr.append(float(m["estimated_drr_db"]))
        target = np.asarray(m["target_angle_seq"], dtype=np.float64)
        rendered = np.asarray(m["direct_rendered_angle_seq"], dtype=np.float64)
        diff = np.abs(((target - rendered + 180.0) % 360.0) - 180.0)
        mismatch.extend(diff.tolist())
        by_room[m["room_profile"]].append(float(m["estimated_rt60"]))
    for wp in wavs[: min(200, len(wavs))]:
        wav_lengths.append(sf.info(str(wp)).frames)
    def stat(xs):
        if not xs:
            return "n/a"
        a = np.asarray(xs, dtype=np.float64)
        return f"mean={a.mean():.4f}, min={a.min():.4f}, p50={np.percentile(a, 50):.4f}, p95={np.percentile(a, 95):.4f}, max={a.max():.4f}"
    missing = []
    if file_ids:
        expected = set(range(1, max(file_ids) + 1))
        missing = sorted(expected - set(file_ids))[:10]
    print(f"[{split}] files metadata={len(metas)} wav={len(wavs)} sampled_wav_lengths={sorted(set(wav_lengths))}")
    print(f"[{split}] file_id_range={min(file_ids) if file_ids else 'n/a'}-{max(file_ids) if file_ids else 'n/a'} missing_first10={missing}")
    print(f"[{split}] subjects={dict(sorted(subjects.items()))}")
    print(f"[{split}] trajectory={dict(traj)} profiles={dict(profiles)} scenes={dict(scenes)}")
    print(f"[{split}] label_cov={len(set(label_values))} rendered_label_cov={len(set(rendered_label_values))} quality_rate={np.mean(quality):.4f}")
    print(f"[{split}] attempts {stat(attempts)}")
    print(f"[{split}] snr_db {stat(snrs)}")
    print(f"[{split}] target_rt60 {stat(target_rt60)}")
    print(f"[{split}] estimated_rt60 {stat(estimated_rt60)}")
    print(f"[{split}] estimated_drr_db {stat(drr)}")
    print(f"[{split}] target_render_mismatch_deg {stat(mismatch)}")
    print(f"[{split}] rt60_by_room=" + ", ".join(f"{k}:{stat(v)}" for k, v in sorted(by_room.items())))
print("[diagnostics] done")
PY
  echo "[$(date '+%F %T')] moving hybrid BRIR gate2 20h parallel generation done"
} 2>&1 | tee "$LOG"
