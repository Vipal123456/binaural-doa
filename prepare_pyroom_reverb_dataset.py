#!/usr/bin/env python3
"""Synthesize ~50h binaural-like room-reverb dataset with pyroomacoustics.

Output format:
- binaural_dev/binauralXXXX.wav
- metadata_dev/metadataXXXX.csv

This script uses LibriSpeech mono utterances as dry source, then simulates a
shoebox room with a 2-mic array (left/right ears) and random source azimuth.
No additive noise is included.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyroomacoustics as pra
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly


def wrap_deg(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


def spherical_to_cartesian(az_deg: float, el_deg: float, r: float) -> Tuple[float, float, float]:
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    x = r * math.cos(el) * math.sin(az)
    y = r * math.cos(el) * math.cos(az)
    z = r * math.sin(el)
    return x, y, z


def list_librispeech_files(root: Path) -> List[Path]:
    files = sorted(root.rglob("*.flac"))
    if not files:
        raise FileNotFoundError(f"No .flac files found under {root}")
    return files


def load_mono_resampled(path: Path, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32, copy=False)

    if sr != target_sr:
        g = math.gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        audio = resample_poly(audio, up=up, down=down).astype(np.float32, copy=False)

    return audio


def fit_to_length(audio: np.ndarray, num_samples: int, rng: np.random.Generator) -> np.ndarray:
    if len(audio) == 0:
        return np.zeros(num_samples, dtype=np.float32)
    if len(audio) >= num_samples:
        max_start = len(audio) - num_samples
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        return audio[start : start + num_samples]
    reps = int(np.ceil(num_samples / len(audio)))
    tiled = np.tile(audio, reps)
    return tiled[:num_samples].astype(np.float32, copy=False)


def room_dims(profile: str, rng: np.random.Generator) -> Tuple[float, float, float]:
    if profile == "small":
        return (
            float(rng.uniform(3.2, 4.2)),
            float(rng.uniform(3.5, 4.8)),
            float(rng.uniform(2.6, 3.0)),
        )
    if profile == "medium":
        return (
            float(rng.uniform(5.0, 6.8)),
            float(rng.uniform(4.5, 6.2)),
            float(rng.uniform(2.8, 3.2)),
        )
    return (
        float(rng.uniform(7.5, 10.0)),
        float(rng.uniform(6.0, 8.0)),
        float(rng.uniform(2.9, 3.5)),
    )


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def write_metadata_csv(path: Path, x: float, y: float, z: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([0, x, y, z, 0, 0, 0, 0])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare pyroomacoustics reverb dataset")
    p.add_argument("--librispeech_root", type=Path, required=True)
    p.add_argument("--output_root", type=Path, required=True)
    p.add_argument("--num_recordings", type=int, default=18000)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--duration_sec", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rt60_min", type=float, default=0.2)
    p.add_argument("--rt60_max", type=float, default=0.8)
    p.add_argument("--room_profiles", nargs="+", default=["small", "medium", "large"])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    binaural_dir = args.output_root / "binaural_dev"
    metadata_dir = args.output_root / "metadata_dev"
    report_path = args.output_root / "reverb_report.csv"

    if args.output_root.exists() and args.overwrite:
        shutil.rmtree(args.output_root)
    binaural_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    flac_files = list_librispeech_files(args.librispeech_root)
    num_samples = int(round(args.duration_sec * args.sample_rate))

    with report_path.open("w", newline="", encoding="utf-8") as rf:
        w = csv.writer(rf)
        w.writerow([
            "file_id",
            "room_profile",
            "room_dims_m",
            "rt60_s",
            "src_distance_m",
            "src_azimuth_deg",
            "num_samples",
        ])

        for i in range(1, args.num_recordings + 1):
            speech_path = flac_files[int(rng.integers(0, len(flac_files)))]
            speech = load_mono_resampled(speech_path, args.sample_rate)
            speech = fit_to_length(speech, num_samples, rng)

            profile = args.room_profiles[int(rng.integers(0, len(args.room_profiles)))]
            lx, ly, lz = room_dims(profile, rng)
            rt60 = float(rng.uniform(args.rt60_min, args.rt60_max))

            e_abs, max_order = pra.inverse_sabine(rt60, [lx, ly, lz])
            max_order = int(clamp(max_order, 1, 8))
            room = pra.ShoeBox(
                [lx, ly, lz],
                fs=args.sample_rate,
                materials=pra.Material(e_abs),
                max_order=max_order,
                ray_tracing=False,
                air_absorption=True,
            )

            cx = float(rng.uniform(0.8, lx - 0.8))
            cy = float(rng.uniform(0.8, ly - 0.8))
            cz = float(rng.uniform(1.2, min(1.8, lz - 0.2)))

            head_w = 0.18
            mic_l = [cx - head_w / 2.0, cy, cz]
            mic_r = [cx + head_w / 2.0, cy, cz]
            mic_array = pra.MicrophoneArray(np.c_[mic_l, mic_r], fs=args.sample_rate)
            room.add_microphone_array(mic_array)

            az_deg = float(rng.uniform(-180.0, 180.0))
            src_dist = float(rng.uniform(1.0, min(4.0, 0.45 * math.sqrt(lx * lx + ly * ly + lz * lz))))
            az_rad = math.radians(az_deg)
            sx = cx + src_dist * math.sin(az_rad)
            sy = cy + src_dist * math.cos(az_rad)
            sz = cz + float(rng.uniform(-0.2, 0.2))
            sx = clamp(sx, 0.5, lx - 0.5)
            sy = clamp(sy, 0.5, ly - 0.5)
            sz = clamp(sz, 1.0, lz - 0.3)

            room.add_source([sx, sy, sz])
            room.compute_rir()

            h_l = np.asarray(room.rir[0][0], dtype=np.float32)
            h_r = np.asarray(room.rir[1][0], dtype=np.float32)
            y_l = fftconvolve(speech, h_l, mode="full").astype(np.float32)
            y_r = fftconvolve(speech, h_r, mode="full").astype(np.float32)
            rec = np.stack([y_l[:num_samples], y_r[:num_samples]], axis=1)
            if rec.shape[0] < num_samples:
                pad = num_samples - rec.shape[0]
                rec = np.pad(rec, ((0, pad), (0, 0)), mode="constant")
            rec = rec[:num_samples]

            peak = float(np.max(np.abs(rec)))
            if peak > 1e-7:
                rec = rec * (0.95 / peak)

            out_id = f"{i:05d}"
            sf.write(binaural_dir / f"binaural{out_id}.wav", rec, args.sample_rate, subtype="PCM_16")

            x, y, z = spherical_to_cartesian(wrap_deg(az_deg), 0.0, src_dist)
            write_metadata_csv(metadata_dir / f"metadata{out_id}.csv", x, y, z)

            w.writerow([
                out_id,
                profile,
                f"{lx:.2f}x{ly:.2f}x{lz:.2f}",
                f"{rt60:.4f}",
                f"{src_dist:.4f}",
                f"{az_deg:.4f}",
                num_samples,
            ])

            if i % 200 == 0 or i == args.num_recordings:
                print(f"{i}/{args.num_recordings} done")

    print(f"Done. Output: {args.output_root}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
