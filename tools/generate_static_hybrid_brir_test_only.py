#!/usr/bin/env python3
"""Generate a static hybrid-BRIR test-only dataset using the existing unseen-subject split."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prepare_robust_multisubject_dataset import (
    DEMAND_SCENES,
    HRTFSubject,
    angular_error_deg,
    fit_to_length,
    list_librispeech_files,
    list_noise_files,
    load_mono_resampled,
    make_balanced_shuffled_bins,
    read_metadata_azimuth,
    render_sample,
    write_metadata_csv,
)


def quality_gate_ok(report: dict) -> bool:
    close_ok = rt60_close_ok(report)
    profile_ok = rt60_profile_ok(report)
    join_ok = bool(report.get("waveform_late_join_ok", report.get("late_join_ok", False)))
    return bool(profile_ok and close_ok and join_ok)


def rt60_profile_ok(report: dict) -> bool:
    estimated_rt60 = float(report["estimated_rt60"])
    room_profile = report["room_profile"]
    lo, hi = {
        "small": (0.20, 0.45),
        "medium": (0.35, 0.65),
        "large": (0.50, 0.80),
    }[room_profile]
    return bool(lo <= estimated_rt60 <= hi)


def rt60_close_ok(report: dict) -> bool:
    target_rt60 = float(report["target_rt60"])
    estimated_rt60 = float(report["estimated_rt60"])
    return bool(abs(estimated_rt60 - target_rt60) <= max(0.08, 0.20 * target_rt60))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate static hybrid BRIR test-only dataset")
    p.add_argument("--split_manifest", type=Path, required=True)
    p.add_argument("--librispeech_root", type=Path, default=Path("/disk2/bywang/data/LibriSpeech/train-clean-100"))
    p.add_argument("--hrtf_root", type=Path, default=Path("/disk2/bywang/data/HRTF"))
    p.add_argument("--demand_root", type=Path, default=Path("/disk2/bywang/data/demand"))
    p.add_argument("--output_root", type=Path, required=True)
    p.add_argument("--test_samples", type=int, default=1800)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--duration_sec", type=float, default=10.0)
    p.add_argument("--source_distance_min", type=float, default=1.0)
    p.add_argument("--source_distance_max", type=float, default=1.5)
    p.add_argument("--num_classes", type=int, default=72)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--brir_max_order", type=int, default=3)
    p.add_argument("--brir_seconds", type=float, default=1.8)
    p.add_argument("--max_attempts_per_sample", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log_interval", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    payload = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    test_subjects = payload["split"]["test_subjects_unseen"]

    speech_files = list_librispeech_files(args.librispeech_root)
    noise_files = list_noise_files(args.demand_root, DEMAND_SCENES)
    hrtf_cache = {
        sid: HRTFSubject(args.hrtf_root / f"subject_{sid}.sofa", args.sample_rate)
        for sid in test_subjects
    }

    wav_dir = args.output_root / "test_subjects_unseen" / "binaural_dev"
    meta_dir = args.output_root / "test_subjects_unseen" / "metadata_dev"
    brir_dir = args.output_root / "test_subjects_unseen" / "brir_dev"
    clean_reverb_dir = args.output_root / "test_subjects_unseen" / "clean_reverb_dev"
    wav_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    brir_dir.mkdir(parents=True, exist_ok=True)
    clean_reverb_dir.mkdir(parents=True, exist_ok=True)

    report_path = args.output_root / "test_subjects_unseen" / "mixing_report.csv"
    path_debug_path = args.output_root / "test_subjects_unseen" / "path_debug.csv"
    fieldnames = [
        "file_id", "split", "rendering_mode", "subject_id", "speech_path", "sofa_path",
        "target_azimuth_deg", "rendered_azimuth_deg", "doa_class", "target_label", "rendered_label", "azimuth_deg", "azimuth_bin",
        "elevation_deg", "target_elevation_deg", "rendered_elevation_deg", "room_profile", "room_dims_m",
        "rt60_s", "target_rt60", "estimated_rt60", "head_center_xyz", "source_xyz", "source_distance_m",
        "room_source_azimuth_deg", "brir_method", "max_order", "sabine_max_order", "num_paths",
        "early_path_count", "brir_seconds", "absorption", "reflection_beta", "direct_delay_samples",
        "late_start_sample", "early_cut_ms", "late_start_ms", "late_tail_type", "late_join_ok", "late_join_metric",
        "waveform_late_join_ok", "waveform_late_join_metric",
        "late_anchor_window_ms", "late_anchor_energy", "rt60_close_to_target_ok", "rt60_within_profile_ok", "quality_gate_ok", "target_drr_db",
        "estimated_drr_db", "estimated_early_late_ratio_db", "left_right_corrcoef",
        "direct_energy_db", "early_energy_db", "late_energy_db", "early_last_delay_ms",
        "low_band_corr", "mid_band_corr", "high_band_corr", "demand_scene",
        "noise_ch_left", "noise_ch_right", "noise_id", "snr_db", "sample_rate", "duration_sec", "num_samples",
    ]

    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    num_samples = int(round(args.duration_sec * args.sample_rate))
    bin_schedule = make_balanced_shuffled_bins(args.test_samples, args.num_classes, np_rng)

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        with path_debug_path.open("w", newline="", encoding="utf-8") as pf:
            path_writer = csv.DictWriter(pf, fieldnames=[
                "file_id", "split", "subject_id", "target_azimuth_deg", "rendered_azimuth_deg",
                "path_id", "order", "image_source_x", "image_source_y", "image_source_z",
                "distance_m", "delay_samples", "gain", "arrival_azimuth_deg",
                "arrival_elevation_deg", "selected_hrir_index",
                "selected_hrir_azimuth_deg", "selected_hrir_elevation_deg",
            ])
            path_writer.writeheader()
            for idx in range(1, args.test_samples + 1):
                subject_id = test_subjects[(idx - 1) % len(test_subjects)]
                subject = hrtf_cache[subject_id]
                target_bin = int(bin_schedule[idx - 1])
                speech_path = speech_files[int(np_rng.integers(0, len(speech_files)))]
                speech = fit_to_length(load_mono_resampled(speech_path, args.sample_rate), num_samples, np_rng)
                mixed = None
                report = None
                for _attempt in range(args.max_attempts_per_sample):
                    mixed, report = render_sample(
                        speech=speech,
                        subject=subject,
                        target_bin=target_bin,
                        noise_files=noise_files,
                        scenes=DEMAND_SCENES,
                        sample_rate=args.sample_rate,
                        num_samples=num_samples,
                        source_distance_min=args.source_distance_min,
                        source_distance_max=args.source_distance_max,
                        rendering_mode="hybrid_pathwise_hrtf_brir_v3",
                        brir_max_order=args.brir_max_order,
                        brir_seconds=args.brir_seconds,
                        rng=py_rng,
                        np_rng=np_rng,
                        no_noise=False,
                        fixed_snr_db=None,
                    )
                    if quality_gate_ok(report):
                        break
                if report is None or mixed is None:
                    raise RuntimeError("Failed to render sample")

                file_id = f"{subject_id}_{idx:06d}"
                wav_path = wav_dir / f"binaural{file_id}.wav"
                meta_path = meta_dir / f"metadata{file_id}.csv"
                sf.write(str(wav_path), mixed, args.sample_rate, subtype="PCM_16")
                np.save(brir_dir / f"brir{file_id}.npy", report["clean_brir"].astype(np.float32))
                np.save(clean_reverb_dir / f"cleanreverb{file_id}.npy", report["clean_reverb_waveform"].astype(np.float32))
                write_metadata_csv(meta_path, float(report["azimuth_deg"]), float(report["elevation_deg"]), float(report["radius"]))
                metadata_az = read_metadata_azimuth(meta_path)
                if angular_error_deg(metadata_az, float(report["azimuth_deg"])) > 1e-4:
                    raise RuntimeError(f"Metadata azimuth mismatch for {file_id}")

                row = {
                    "file_id": file_id,
                    "split": "test",
                    "rendering_mode": report["rendering_mode"],
                    "subject_id": subject_id,
                    "speech_path": str(speech_path),
                    "sofa_path": str(subject.sofa_path),
                    "target_azimuth_deg": f"{float(report['target_azimuth_deg']):.6f}",
                    "rendered_azimuth_deg": f"{float(report['rendered_azimuth_deg']):.6f}",
                    "doa_class": int(report["doa_class"]),
                    "target_label": int(report["target_label"]),
                    "rendered_label": int(report["rendered_label"]),
                    "azimuth_deg": f"{float(report['azimuth_deg']):.6f}",
                    "azimuth_bin": int(report["azimuth_bin"]),
                    "elevation_deg": f"{float(report['elevation_deg']):.6f}",
                    "target_elevation_deg": f"{float(report['target_elevation_deg']):.6f}",
                    "rendered_elevation_deg": f"{float(report['rendered_elevation_deg']):.6f}",
                    "room_profile": report["room_profile"],
                    "room_dims_m": report["room_dims_m"],
                    "rt60_s": f"{float(report['rt60_s']):.6f}",
                    "target_rt60": f"{float(report['target_rt60']):.6f}",
                    "estimated_rt60": f"{float(report['estimated_rt60']):.6f}",
                    "head_center_xyz": report["head_center_xyz"],
                    "source_xyz": report["source_xyz"],
                    "source_distance_m": f"{float(report['source_distance_m']):.6f}",
                    "room_source_azimuth_deg": f"{float(report['room_source_azimuth_deg']):.6f}",
                    "brir_method": report["brir_method"],
                    "max_order": int(report["max_order"]),
                    "sabine_max_order": int(report["sabine_max_order"]),
                    "num_paths": int(report["num_paths"]),
                    "early_path_count": int(report["early_path_count"]),
                    "brir_seconds": f"{float(report['brir_seconds']):.3f}",
                    "absorption": f"{float(report['absorption']):.6f}",
                    "reflection_beta": f"{float(report['reflection_beta']):.6f}",
                    "direct_delay_samples": int(report["direct_delay_samples"]),
                    "late_start_sample": int(report["late_start_sample"]),
                    "early_cut_ms": f"{float(report['early_cut_ms']):.3f}",
                    "late_start_ms": f"{float(report['late_start_ms']):.3f}",
                    "late_tail_type": report["late_tail_type"],
                    "late_join_ok": int(bool(report["late_join_ok"])),
                    "late_join_metric": f"{float(report['late_join_metric']):.6f}",
                    "waveform_late_join_ok": int(bool(report.get("waveform_late_join_ok", report["late_join_ok"]))),
                    "waveform_late_join_metric": f"{float(report.get('waveform_late_join_metric', report['late_join_metric'])):.6f}",
                    "late_anchor_window_ms": report["late_anchor_window_ms"],
                    "late_anchor_energy": f"{float(report['late_anchor_energy']):.6f}",
                    "rt60_close_to_target_ok": int(rt60_close_ok(report)),
                    "rt60_within_profile_ok": int(rt60_profile_ok(report)),
                    "quality_gate_ok": int(quality_gate_ok(report)),
                    "target_drr_db": f"{float(report['target_drr_db']):.6f}" if report.get("target_drr_db") == report.get("target_drr_db") else "",
                    "estimated_drr_db": f"{float(report['estimated_drr_db']):.6f}" if report.get("estimated_drr_db") == report.get("estimated_drr_db") else "",
                    "estimated_early_late_ratio_db": f"{float(report['estimated_early_late_ratio_db']):.6f}" if report.get("estimated_early_late_ratio_db") == report.get("estimated_early_late_ratio_db") else "",
                    "left_right_corrcoef": f"{float(report['left_right_corrcoef']):.6f}" if report.get("left_right_corrcoef") == report.get("left_right_corrcoef") else "",
                    "direct_energy_db": f"{float(report['direct_energy_db']):.6f}",
                    "early_energy_db": f"{float(report['early_energy_db']):.6f}",
                    "late_energy_db": f"{float(report['late_energy_db']):.6f}",
                    "early_last_delay_ms": f"{float(report['early_last_delay_ms']):.6f}",
                    "low_band_corr": f"{float(report['low_band_corr']):.6f}",
                    "mid_band_corr": f"{float(report['mid_band_corr']):.6f}",
                    "high_band_corr": f"{float(report['high_band_corr']):.6f}",
                    "demand_scene": report["demand_scene"],
                    "noise_ch_left": report["noise_ch_left"],
                    "noise_ch_right": report["noise_ch_right"],
                    "noise_id": report["noise_id"],
                    "snr_db": f"{float(report['snr_db']):.6f}",
                    "sample_rate": args.sample_rate,
                    "duration_sec": f"{args.duration_sec:.3f}",
                    "num_samples": num_samples,
                }
                writer.writerow(row)

                json_meta = {
                    "file_id": file_id,
                    "rendering_mode": report["rendering_mode"],
                    "target_azimuth": float(report["target_azimuth_deg"]),
                    "rendered_azimuth": float(report["rendered_azimuth_deg"]),
                    "cipic_selected_azimuth": float(report["cipic_selected_azimuth"]),
                    "doa_class": int(report["doa_class"]),
                    "target_label": int(report["target_label"]),
                    "rendered_label": int(report["rendered_label"]),
                    "subject_id": subject_id,
                    "source_distance": float(report["source_distance_m"]),
                    "source_position": report["source_xyz"],
                    "listener_position": report["head_center_xyz"],
                    "room_profile": report["room_profile"],
                    "room_dimensions": report["room_dims_m"],
                    "target_rt60": float(report["target_rt60"]),
                    "estimated_rt60": float(report["estimated_rt60"]),
                    "max_order": int(report["max_order"]),
                    "num_paths": int(report["num_paths"]),
                    "early_path_count": int(report["early_path_count"]),
                    "direct_delay_samples": int(report["direct_delay_samples"]),
                    "late_start_sample": int(report["late_start_sample"]),
                    "early_cut_ms": float(report["early_cut_ms"]),
                    "late_start_ms": float(report["late_start_ms"]),
                    "target_drr_db": report["target_drr_db"],
                    "estimated_drr_db": report["estimated_drr_db"],
                    "estimated_early_late_ratio_db": report["estimated_early_late_ratio_db"],
                    "left_right_corrcoef": report["left_right_corrcoef"],
                    "late_tail_type": report["late_tail_type"],
                    "rt60_close_to_target_ok": rt60_close_ok(report),
                    "rt60_within_profile_ok": rt60_profile_ok(report),
                    "drr_ok": True,
                    "late_join_ok": bool(report["late_join_ok"]),
                    "late_join_metric": float(report["late_join_metric"]),
                    "waveform_late_join_ok": bool(report.get("waveform_late_join_ok", report["late_join_ok"])),
                    "waveform_late_join_metric": float(report.get("waveform_late_join_metric", report["late_join_metric"])),
                    "late_anchor_window_ms": report["late_anchor_window_ms"],
                    "late_anchor_energy": float(report["late_anchor_energy"]),
                    "early_last_delay_ms": float(report["early_last_delay_ms"]),
                    "direct_energy_db": float(report["direct_energy_db"]),
                    "early_energy_db": float(report["early_energy_db"]),
                    "late_energy_db": float(report["late_energy_db"]),
                    "low_band_corr": float(report["low_band_corr"]),
                    "mid_band_corr": float(report["mid_band_corr"]),
                    "high_band_corr": float(report["high_band_corr"]),
                    "noise_id": report["noise_id"],
                    "snr_db": float(report["snr_db"]),
                }
                (meta_dir / f"metadata{file_id}.json").write_text(json.dumps(json_meta, indent=2), encoding="utf-8")
                for path_row in report.get("path_debug", []):
                    path_writer.writerow({
                        "file_id": file_id,
                        "split": "test",
                        "subject_id": subject_id,
                        "target_azimuth_deg": f"{float(report['target_azimuth_deg']):.6f}",
                        "rendered_azimuth_deg": f"{float(report['rendered_azimuth_deg']):.6f}",
                        **path_row,
                    })

                if idx % args.log_interval == 0 or idx == args.test_samples:
                    print(f"[test] {idx}/{args.test_samples} generated", flush=True)

    manifest = {
        "dataset": args.output_root.name,
        "split_manifest": str(args.split_manifest),
        "rendering_mode": "hybrid_pathwise_hrtf_brir_v3",
        "counts": {"test_recordings": args.test_samples},
        "test_subjects": test_subjects,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
