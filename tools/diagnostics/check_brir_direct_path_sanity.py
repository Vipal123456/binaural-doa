#!/usr/bin/env python3
"""Sanity-check direct-path path-wise HRIR rendering for static BRIR data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prepare_robust_multisubject_dataset import (  # noqa: E402
    HRTFSubject,
    peak_normalize,
    spherical_to_cartesian,
    synthesize_pathwise_hrtf_brir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check BRIR direct-path left/right direction sanity")
    parser.add_argument("--hrtf_root", type=Path, default=Path("/disk2/bywang/data/HRTF"))
    parser.add_argument("--subject_id", default="003")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--distance", type=float, default=1.2)
    parser.add_argument("--rt60", type=float, default=0.3)
    parser.add_argument("--brir_seconds", type=float, default=0.25)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/brir_direct_path_sanity"))
    parser.add_argument("--write_wav", action="store_true")
    return parser.parse_args()


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64))) + 1e-12))


def estimate_itd_samples(left: np.ndarray, right: np.ndarray) -> int:
    """Return positive when right channel leads left."""
    corr = np.correlate(left, right, mode="full")
    lag = int(np.argmax(corr) - (len(right) - 1))
    return lag


def spectral_distance(a: np.ndarray, b: np.ndarray) -> float:
    spec_a = np.abs(np.fft.rfft(a))
    spec_b = np.abs(np.fft.rfft(b))
    log_a = np.log(np.maximum(spec_a, 1e-8))
    log_b = np.log(np.maximum(spec_b, 1e-8))
    return float(np.mean(np.abs(log_a - log_b)))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    subject = HRTFSubject(args.hrtf_root / f"subject_{args.subject_id}.sofa", args.sample_rate)
    dims = (6.0, 5.0, 3.0)
    head = np.array([3.0, 2.5, 1.5], dtype=np.float64)
    angles = [0.0, 90.0, -90.0, -180.0]
    rng = random.Random(1234)
    impulse = np.zeros(int(round(0.1 * args.sample_rate)), dtype=np.float32)
    impulse[0] = 1.0

    rows = []
    brirs = {}
    for az in angles:
        dx, dy, dz = spherical_to_cartesian(az, 0.0, args.distance)
        source = head + np.array([dx, dy, dz], dtype=np.float64)
        brir, report = synthesize_pathwise_hrtf_brir(
            subject=subject,
            sample_rate=args.sample_rate,
            dims=dims,
            rt60=args.rt60,
            head_center=head,
            source_xyz=source,
            max_order=0,
            brir_seconds=args.brir_seconds,
        )
        stereo = np.stack([
            fftconvolve(impulse, brir[0], mode="full")[: len(impulse)],
            fftconvolve(impulse, brir[1], mode="full")[: len(impulse)],
        ], axis=1).astype(np.float32)
        stereo = peak_normalize(stereo)
        brirs[az] = brir

        left = brir[0]
        right = brir[1]
        left_peak = int(np.argmax(np.abs(left)))
        right_peak = int(np.argmax(np.abs(right)))
        left_rms = rms(left)
        right_rms = rms(right)
        ild_db = 20.0 * math.log10(max(right_rms, 1e-12) / max(left_rms, 1e-12))
        itd_samples = estimate_itd_samples(left, right)
        path = report["path_debug"][0] if report.get("path_debug") else {}
        row = {
            "target_azimuth_deg": az,
            "direct_rendered_azimuth_deg": report.get("direct_rendered_azimuth_deg"),
            "direct_rendered_elevation_deg": report.get("direct_rendered_elevation_deg"),
            "left_peak_index": left_peak,
            "right_peak_index": right_peak,
            "peak_delta_right_minus_left": right_peak - left_peak,
            "left_rms": left_rms,
            "right_rms": right_rms,
            "ild_db_right_over_left": ild_db,
            "xcorr_itd_samples_positive_right_leads": itd_samples,
            "num_paths": report.get("num_paths"),
            "selected_hrir_index": path.get("selected_hrir_index"),
            "selected_hrir_azimuth_deg": path.get("selected_hrir_azimuth_deg"),
            "selected_hrir_elevation_deg": path.get("selected_hrir_elevation_deg"),
            "arrival_azimuth_deg": path.get("arrival_azimuth_deg"),
            "arrival_elevation_deg": path.get("arrival_elevation_deg"),
            "delay_samples": path.get("delay_samples"),
            "gain": path.get("gain"),
        }
        rows.append(row)
        if args.write_wav:
            sf.write(args.output_dir / f"direct_path_{int(az):+04d}.wav", stereo, args.sample_rate)

    front = brirs[0.0]
    back = brirs[-180.0]
    for row in rows:
        if float(row["target_azimuth_deg"]) == -180.0:
            row["spectral_distance_vs_front_left"] = spectral_distance(front[0], back[0])
            row["spectral_distance_vs_front_right"] = spectral_distance(front[1], back[1])
        else:
            row["spectral_distance_vs_front_left"] = ""
            row["spectral_distance_vs_front_right"] = ""

    csv_path = args.output_dir / "direct_path_sanity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "direct_path_sanity.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(json.dumps(rows, indent=2), flush=True)
    print(f"Wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
