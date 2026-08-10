#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk2/bywang/DOA-net"
PYTHON="/home/bywang/miniconda3/envs/doa/bin/python"
DATA_ROOT="$ROOT/data/librispeech_cipic_moving_hybridbrir_gate2_debug_v2"
LOG="$ROOT/outputs/logs_moving_hybridbrir_gate2_debug_pipeline.log"

cd "$ROOT"
mkdir -p "$ROOT/outputs"

{
  echo "[$(date '+%F %T')] moving hybrid BRIR gate2 debug generation start"
  echo "data_root=$DATA_ROOT"
  "$PYTHON" -u prepare_moving_dataset.py \
    --condition moving_hybridbrir_gate2 \
    --output_root "$DATA_ROOT" \
    --train_subjects 003,008,011,012,020,021,027,028,033,048,051,058,059,060,061,065,119,124,126,131,133,134,135,147 \
    --val_subjects 152,154,155 \
    --test_subjects 156,163,165 \
    --samples_per_train 1000 \
    --samples_per_eval 200 \
    --duration_sec 4.0 \
    --chunk_seconds 0.1 \
    --label_steps 40 \
    --label_source target \
    --distance_min 1.0 \
    --distance_max 1.5 \
    --brir_max_order 3 \
    --brir_seconds 1.8 \
    --early_cut_ms 80 \
    --late_start_ms 80 \
    --snr_min -10 \
    --snr_max 10 \
    --log_interval 20 \
    --overwrite

  "$PYTHON" - <<'PY'
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

root = Path("data/librispeech_cipic_moving_hybridbrir_gate2_debug_v2")
print("[diagnostics] root=", root)
for split in ["train_subjects", "val_subjects", "test_subjects_unseen"]:
    metas = sorted((root / split / "metadata_dev").glob("*.json"))
    wavs = sorted((root / split / "binaural_dev").glob("*.wav"))
    traj = Counter()
    snrs = []
    target_rt60 = []
    estimated_rt60 = []
    drr = []
    quality = []
    rt60_close = []
    rt60_profile = []
    late_join = []
    label_lens = []
    wav_lengths = []
    target_render_mismatch = []
    low_corr = []
    mid_corr = []
    high_corr = []
    doa_label_values = []
    label_sources = Counter()
    by_room = defaultdict(list)
    by_traj = defaultdict(list)
    for mp in metas:
        m = json.loads(mp.read_text())
        traj[m["trajectory_type"]] += 1
        snrs.append(float(m["snr_db"]))
        target_rt60.append(float(m["target_rt60"]))
        estimated_rt60.append(float(m["estimated_rt60"]))
        if m.get("estimated_drr_db") is not None:
            drr.append(float(m["estimated_drr_db"]))
        quality.append(bool(m.get("quality_gate_ok")))
        rt60_close.append(bool(m.get("rt60_close_to_target_ok")))
        rt60_profile.append(bool(m.get("rt60_within_profile_ok")))
        late_join.append(bool(m.get("late_join_ok")))
        label_lens.append(len(m["rendered_label_seq"]))
        doa_label_values.extend(m["doa_labels"])
        label_sources[m.get("label_source", "unknown")] += 1
        target = np.asarray(m["target_angle_seq"], dtype=np.float64)
        rendered = np.asarray(m["direct_rendered_angle_seq"], dtype=np.float64)
        diff = np.abs(((target - rendered + 180.0) % 360.0) - 180.0)
        target_render_mismatch.extend(diff.tolist())
        if m.get("low_band_corr") is not None:
            low_corr.append(float(m["low_band_corr"]))
            mid_corr.append(float(m["mid_band_corr"]))
            high_corr.append(float(m["high_band_corr"]))
        by_room[m["room_profile"]].append(float(m["estimated_rt60"]))
        by_traj[m["trajectory_type"]].append(float(np.mean(diff)))
    for wp in wavs:
        info = sf.info(str(wp))
        wav_lengths.append(info.frames)
    def stat(xs):
        if not xs:
            return "n/a"
        a = np.asarray(xs, dtype=np.float64)
        return f"mean={a.mean():.4f}, min={a.min():.4f}, max={a.max():.4f}"
    print(f"[{split}] files: metadata={len(metas)} wav={len(wavs)}")
    print(f"[{split}] wav_lengths={sorted(set(wav_lengths))} label_lengths={sorted(set(label_lens))}")
    print(f"[{split}] trajectory={dict(traj)}")
    print(f"[{split}] snr_db {stat(snrs)}")
    print(f"[{split}] target_rt60 {stat(target_rt60)}")
    print(f"[{split}] estimated_rt60 {stat(estimated_rt60)}")
    print(f"[{split}] estimated_drr_db {stat(drr)}")
    print(f"[{split}] target_render_mismatch_deg {stat(target_render_mismatch)}")
    print(f"[{split}] doa_label_coverage={len(set(doa_label_values))} label_sources={dict(label_sources)}")
    print(f"[{split}] low/mid/high_corr {stat(low_corr)} | {stat(mid_corr)} | {stat(high_corr)}")
    print(f"[{split}] ok_rates quality={np.mean(quality):.3f} rt60_close={np.mean(rt60_close):.3f} rt60_profile={np.mean(rt60_profile):.3f} late_join={np.mean(late_join):.3f}")
    print(f"[{split}] rt60_by_room=" + ", ".join(f"{k}:{stat(v)}" for k, v in sorted(by_room.items())))
    print(f"[{split}] mismatch_by_traj=" + ", ".join(f"{k}:{stat(v)}" for k, v in sorted(by_traj.items())))
print("[diagnostics] done")
PY
  echo "[$(date '+%F %T')] moving hybrid BRIR gate2 debug generation done"
} 2>&1 | tee "$LOG"
