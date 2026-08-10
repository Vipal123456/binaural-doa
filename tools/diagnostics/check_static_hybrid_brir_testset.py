#!/usr/bin/env python3
"""Check hybrid static BRIR testset generation sanity."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt


def estimate_rt60_from_ir(ir: np.ndarray, sample_rate: int) -> float:
    if ir.size == 0 or np.max(np.abs(ir)) < 1e-8:
        return float("nan")
    edc = np.cumsum(np.square(ir[::-1].astype(np.float64)))[::-1]
    edc_db = 10.0 * np.log10(np.maximum(edc / np.max(edc), 1e-12))
    times = np.arange(len(ir), dtype=np.float64) / float(sample_rate)
    mask = (edc_db <= -5.0) & (edc_db >= -35.0)
    if mask.sum() < 8:
        return float("nan")
    slope, _ = np.polyfit(times[mask], edc_db[mask], deg=1)
    if slope >= -1e-6:
        return float("nan")
    return float(-60.0 / slope)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64))) + 1e-12))


def join_metric(clean_reverb: np.ndarray, late_start: int, sample_rate: int) -> float:
    pre = clean_reverb[:, max(0, late_start - int(0.01 * sample_rate)):late_start]
    post = clean_reverb[:, late_start: late_start + int(0.01 * sample_rate)]
    pre_rms = rms(pre) if pre.size else 0.0
    post_rms = rms(post) if post.size else 0.0
    return float(abs(pre_rms - post_rms) / max(pre_rms, 1e-8))


def corrcoef_lr(x: np.ndarray) -> float:
    if x.ndim != 2:
        return float("nan")
    if x.shape[0] == 2:
        left, right = x[0], x[1]
    elif x.shape[1] == 2:
        left, right = x[:, 0], x[:, 1]
    else:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def bandpass_pair(x: np.ndarray, sample_rate: int, kind: str) -> np.ndarray:
    nyq = sample_rate * 0.5
    if kind == "low":
        sos = butter(4, 500.0 / nyq, btype="lowpass", output="sos")
    elif kind == "mid":
        sos = butter(4, [500.0 / nyq, 2000.0 / nyq], btype="bandpass", output="sos")
    elif kind == "high":
        sos = butter(4, 2000.0 / nyq, btype="highpass", output="sos")
    else:
        raise ValueError(f"Unsupported band kind: {kind}")
    return sosfilt(sos, x, axis=-1).astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=Path, required=True)
    p.add_argument("--sample_rate", type=int, default=16000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    split_root = args.dataset_root / "test_subjects_unseen"
    meta_dir = split_root / "metadata_dev"
    brir_dir = split_root / "brir_dev"
    clean_reverb_dir = split_root / "clean_reverb_dev"
    path_debug_path = split_root / "path_debug.csv"

    metas = sorted(meta_dir.glob("metadata*.json"))
    path_rows = list(csv.DictReader(path_debug_path.open(newline="", encoding="utf-8")))
    path_map = {}
    for row in path_rows:
        path_map.setdefault(row["file_id"], []).append(row)

    direct_delay_ok = []
    early_count = []
    early_delay_ok = []
    late_smooth = []
    waveform_late_smooth = []
    rt60_close = []
    rt60_in_range = []
    drr_close = []
    lr_not_same = []
    lr_not_indep = []
    late_corr_low = []
    late_corr_mid = []
    late_corr_high = []
    noisy_after_reverb = []
    per_profile = {"small": [], "medium": [], "large": []}
    early_last_delay_ms = []
    early_energy_db = []
    late_energy_db = []
    direct_energy_db = []
    late_join_metric_values = []
    gate_like_ok = []

    for mp in metas:
        meta = json.loads(mp.read_text())
        file_id = meta["file_id"]
        brir = np.load(brir_dir / f"brir{file_id}.npy")
        clean_reverb = np.load(clean_reverb_dir / f"cleanreverb{file_id}.npy")
        direct_delay = int(meta["direct_delay_samples"])
        late_start = int(meta["late_start_sample"])
        early_cut_ms = float(meta["early_cut_ms"])
        target_rt60 = float(meta["target_rt60"])
        est_rt60 = estimate_rt60_from_ir(0.5 * (brir[0] + brir[1]), args.sample_rate)
        rows = path_map.get(file_id, [])
        rows_delay = [int(r["delay_samples"]) for r in rows]

        direct_delay_ok.append(direct_delay > 0)
        early_count.append(len(rows))
        early_delay_ok.append(all(d <= late_start for d in rows_delay))

        if clean_reverb.ndim == 2 and clean_reverb.shape[0] != 2 and clean_reverb.shape[1] == 2:
            clean_reverb = clean_reverb.T
        metric = join_metric(clean_reverb, late_start, args.sample_rate)
        late_join_metric_values.append(metric)
        late_smooth.append(metric < 0.5)
        waveform_late_smooth.append(metric < 0.35)

        rt60_close.append(abs(est_rt60 - target_rt60) <= max(0.10, 0.25 * target_rt60))
        lo, hi = {
            "small": (0.20, 0.45),
            "medium": (0.35, 0.65),
            "large": (0.50, 0.80),
        }[meta["room_profile"]]
        rt60_in_range.append(lo <= est_rt60 <= hi)
        close_ok = abs(est_rt60 - target_rt60) <= max(0.08, 0.20 * target_rt60)
        profile_ok = lo <= est_rt60 <= hi
        per_profile[meta["room_profile"]].append((target_rt60, est_rt60, profile_ok, close_ok))
        gate_like_ok.append(bool(profile_ok and close_ok and metric < 0.35))

        target_drr = meta.get("target_drr_db")
        est_drr = meta.get("estimated_drr_db")
        if target_drr is None or est_drr is None:
            drr_close.append(False)
        else:
            drr_close.append(abs(float(target_drr) - float(est_drr)) <= 3.0)

        cc = corrcoef_lr(brir)
        lr_not_same.append(abs(cc) < 0.999)
        lr_not_indep.append(abs(cc) > 0.05)
        late = brir[:, late_start:]
        if late.shape[-1] > int(0.05 * args.sample_rate):
            late_corr_low.append(corrcoef_lr(bandpass_pair(late, args.sample_rate, "low")))
            late_corr_mid.append(corrcoef_lr(bandpass_pair(late, args.sample_rate, "mid")))
            late_corr_high.append(corrcoef_lr(bandpass_pair(late, args.sample_rate, "high")))
        early_last_delay_ms.append(float(meta.get("early_last_delay_ms", np.nan)))
        early_energy_db.append(float(meta.get("early_energy_db", np.nan)))
        late_energy_db.append(float(meta.get("late_energy_db", np.nan)))
        direct_energy_db.append(float(meta.get("direct_energy_db", np.nan)))
        noisy_after_reverb.append(True)

    profile_summary = {}
    for room, rows in per_profile.items():
        if not rows:
            continue
        target = np.array([r[0] for r in rows], dtype=np.float64)
        estimated = np.array([r[1] for r in rows], dtype=np.float64)
        hit = np.array([r[2] for r in rows], dtype=np.float64)
        close = np.array([r[3] for r in rows], dtype=np.float64)
        profile_summary[room] = {
            "count": int(len(rows)),
            "target_rt60_mean": float(target.mean()),
            "target_rt60_std": float(target.std()),
            "estimated_rt60_mean": float(estimated.mean()),
            "estimated_rt60_std": float(estimated.std()),
            "profile_range_hit_rate": float(hit.mean()),
            "close_to_target_rate": float(close.mean()),
        }

    summary = {
        "num_samples": len(metas),
        "direct_path_delay_reasonable_rate": float(np.mean(direct_delay_ok)),
        "early_path_count_min_mean_max": [int(np.min(early_count)), float(np.mean(early_count)), int(np.max(early_count))],
        "early_path_count_median": float(np.median(early_count)),
        "early_path_count_p10_p90": [float(np.percentile(early_count, 10)), float(np.percentile(early_count, 90))],
        "early_path_delay_le_80ms_rate": float(np.mean(early_delay_ok)),
        "early_last_delay_ms_mean_max": [float(np.nanmean(early_last_delay_ms)), float(np.nanmax(early_last_delay_ms))],
        "late_tail_smooth_join_rate": float(np.mean(late_smooth)),
        "waveform_late_join_ok_rate": float(np.mean(waveform_late_smooth)),
        "late_join_metric_mean_p90_max": [
            float(np.nanmean(late_join_metric_values)),
            float(np.nanpercentile(late_join_metric_values, 90)),
            float(np.nanmax(late_join_metric_values)),
        ],
        "quality_gate_like_pass_rate": float(np.mean(gate_like_ok)),
        "estimated_rt60_close_to_target_rate": float(np.mean(rt60_close)),
        "estimated_rt60_within_profile_range_rate": float(np.mean(rt60_in_range)),
        "estimated_drr_close_rate": float(np.mean(drr_close)),
        "left_right_not_identical_rate": float(np.mean(lr_not_same)),
        "left_right_not_independent_rate": float(np.mean(lr_not_indep)),
        "energy_db_mean": {
            "direct": float(np.nanmean(direct_energy_db)),
            "early": float(np.nanmean(early_energy_db)),
            "late": float(np.nanmean(late_energy_db)),
        },
        "late_bandwise_corr_mean": {
            "low": None if not late_corr_low else float(np.nanmean(late_corr_low)),
            "mid": None if not late_corr_mid else float(np.nanmean(late_corr_mid)),
            "high": None if not late_corr_high else float(np.nanmean(late_corr_high)),
        },
        "late_bandwise_corr_std": {
            "low": None if not late_corr_low else float(np.nanstd(late_corr_low)),
            "mid": None if not late_corr_mid else float(np.nanstd(late_corr_mid)),
            "high": None if not late_corr_high else float(np.nanstd(late_corr_high)),
        },
        "late_bandwise_corr_p10_p50_p90": {
            "low": None if not late_corr_low else [float(np.nanpercentile(late_corr_low, q)) for q in (10, 50, 90)],
            "mid": None if not late_corr_mid else [float(np.nanpercentile(late_corr_mid, q)) for q in (10, 50, 90)],
            "high": None if not late_corr_high else [float(np.nanpercentile(late_corr_high, q)) for q in (10, 50, 90)],
        },
        "profile_rt60_summary": profile_summary,
        "noisy_added_after_reverb_placeholder_rate": float(np.mean(noisy_after_reverb)),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
