#!/usr/bin/env python3
"""Generate Neidhardt measured BRIR external test set.

Usage:
  python tools/generate_neidhardt_brir_testset.py
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import sofa
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

# Project root
ROOT = Path("/disk2/bywang/DOA-net")
BRIR_DIR = Path("/disk2/bywang/data/neidhardt_brir")
SPEECH_DIR = Path("/disk2/bywang/data/LibriSpeech/LibriSpeech_test/test-clean")
DEMAND_DIR = Path("/disk2/bywang/data/demand")
OUTPUT_DIR = ROOT / "data" / "librispeech_neidhardt_measured_brir_test_v1"

DEMAND_SCENES = ["OOFFICE", "PCAFETER", "TMETRO", "TBUS", "SPSQUARE", "NPARK"]
SNR_LEVELS = ["clean", -10, -5, 0, 5, 10]
NUM_CLASSES = 72
SAMPLE_RATE = 16000
SEGMENT_SEC = 2.0
CROPS_PER_BRIR = 2

# ── helpers ──────────────────────────────────────────────────────────

def wrap_deg(a: float) -> float:
    return ((float(a) + 180) % 360) - 180

def angle_to_label(az_deg: float) -> int:
    w = wrap_deg(az_deg)
    return int(np.clip(np.floor((w + 180.0) / (360.0 / NUM_CLASSES)), 0, NUM_CLASSES - 1))

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))

def peak_normalize(stereo: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_abs = float(np.max(np.abs(stereo)))
    if max_abs > peak and max_abs > 1e-8:
        stereo = stereo * (peak / max_abs)
    return stereo.astype(np.float32, copy=False)

def load_mono(path: Path, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        g = math.gcd(int(sr), int(target_sr))
        audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
    return np.asarray(audio, dtype=np.float32)

def resample_brir(brir: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1D BRIR channel."""
    g = math.gcd(int(orig_sr), int(target_sr))
    return resample_poly(brir, target_sr // g, orig_sr // g).astype(np.float32)

def list_noise_files(demand_root: Path, scenes: Sequence[str]) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for scene in scenes:
        scene_dir = demand_root / scene
        wavs = sorted(scene_dir.glob("ch*.wav"))
        if not wavs:
            raise FileNotFoundError(f"No ch*.wav noise files found in {scene_dir}")
        out[scene] = wavs
    return out

def choose_binaural_noise(
    noise_files: Dict[str, List[Path]],
    length: int,
    sample_rate: int,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    scene = rng.choice(sorted(noise_files))
    files = noise_files[scene]
    left_path = rng.choice(files)
    left = read_noise_segment(left_path, length, sample_rate, rng)
    if len(files) > 1:
        candidates = [p for p in files if p != left_path]
        right_path = rng.choice(candidates)
        right = read_noise_segment(right_path, length, sample_rate, rng)
    else:
        right_path = left_path
        right = decorrelate_noise(left, sample_rate, rng)
    return left, right, {"scene": scene, "ch_left": left_path.name, "ch_right": right_path.name}

def read_noise_segment(path: Path, length: int, target_sr: int, rng: random.Random) -> np.ndarray:
    info = sf.info(str(path))
    if info.frames >= length and info.samplerate == target_sr:
        start = rng.randint(0, max(0, info.frames - length))
        noise, _ = sf.read(str(path), start=start, frames=length, dtype="float32", always_2d=False)
    else:
        noise, sr = sf.read(str(path), dtype="float32", always_2d=False)
        noise = np.asarray(noise, dtype=np.float32)
        if noise.ndim > 1:
            noise = noise[:, 0]
        if sr != target_sr:
            g = math.gcd(int(sr), int(target_sr))
            noise = resample_poly(noise, target_sr // g, sr // g).astype(np.float32)
    noise = np.asarray(noise, dtype=np.float32)
    if noise.ndim > 1:
        noise = noise[:, 0]
    if len(noise) < length:
        reps = int(np.ceil(length / max(1, len(noise))))
        noise = np.tile(noise, reps)
    if len(noise) > length:
        if start is None or info.samplerate != target_sr:
            start = rng.randint(0, max(0, len(noise) - length))
        noise = noise[start:start + length]
    return noise.astype(np.float32, copy=False)

def decorrelate_noise(noise: np.ndarray, sample_rate: int, rng: random.Random) -> np.ndarray:
    shift = rng.randint(max(1, sample_rate // 1000), max(2, sample_rate // 200))
    if rng.random() < 0.5:
        shift = -shift
    shifted = np.roll(noise, shift).astype(np.float32, copy=False)
    return (0.92 * shifted + 0.08 * np.roll(shifted, 1)).astype(np.float32, copy=False)

def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    signal = signal.astype(np.float32, copy=False)
    noise = noise.astype(np.float32, copy=False)
    noise = noise - float(np.mean(noise))
    sig_power = float(np.mean(signal.astype(np.float64) ** 2))
    noise_power = float(np.mean(noise.astype(np.float64) ** 2))
    if sig_power < 1e-12 or noise_power < 1e-12:
        return signal.copy()
    scale = math.sqrt(sig_power / ((10.0 ** (snr_db / 10.0)) * noise_power))
    return (signal + scale * noise).astype(np.float32, copy=False)

def extract_crops(waveform: np.ndarray, num_crops: int, crop_samples: int,
                  rng: np.random.Generator) -> List[np.ndarray]:
    total = waveform.shape[0]
    if total < crop_samples:
        padded = np.pad(waveform, ((0, crop_samples - total), (0, 0)), mode="constant")
        return [padded[:crop_samples].astype(np.float32)]
    max_start = total - crop_samples
    crops = []
    for _ in range(min(num_crops, max_start // crop_samples + 1)):
        start = int(rng.integers(0, max_start + 1))
        crops.append(waveform[start:start + crop_samples].astype(np.float32))
    if len(crops) == 0:
        crops.append(waveform[:crop_samples].astype(np.float32))
    return crops

# ── main ─────────────────────────────────────────────────────────────

def main():
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)

    # 1. Collect speech files and noise files
    speech_files = sorted(SPEECH_DIR.rglob("*.flac"))
    if not speech_files:
        raise FileNotFoundError(f"No FLAC files in {SPEECH_DIR}")
    print(f"Speech files: {len(speech_files)}")

    rng.shuffle(speech_files)
    noise_files = list_noise_files(DEMAND_DIR, DEMAND_SCENES)
    print(f"Noise scenes: {list(noise_files.keys())}")

    # 2. Collect SOFA files
    sofa_files = sorted(BRIR_DIR.glob("Pos*_LS_*.sofa"))
    print(f"SOFA files: {len(sofa_files)}")

    # 3. Prepare output
    wav_dir = OUTPUT_DIR / "test_all" / "binaural_dev"
    meta_dir = OUTPUT_DIR / "test_all" / "metadata_dev"
    wav_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    mixing_rows: List[Dict] = []
    segment_idx = 0
    speech_idx = 0
    crop_samples = int(SEGMENT_SEC * SAMPLE_RATE)

    for sofa_path in sofa_files:
        pos_str = sofa_path.stem  # e.g., "Pos3_LS_0"
        parts = pos_str.split("_")
        pos_num = parts[0].replace("Pos", "")
        ls_angle_str = parts[2]  # "0" or "180"

        db = sofa.Database.open(str(sofa_path))
        ir = np.asarray(db.Data.IR.get_values())  # [72, 2, 1, 44100]
        sofa_sr = int(round(float(db.Data.SamplingRate.get_values()[0])))
        db.close()

        for M in range(72):
            # Extract & downsample BRIR: ch0=RIGHT, ch1=LEFT
            brir_r = resample_brir(ir[M, 0, 0], sofa_sr, SAMPLE_RATE)
            brir_l = resample_brir(ir[M, 1, 0], sofa_sr, SAMPLE_RATE)

            # Label: same formula for both LS_0 and LS_180
            # M=0: head faces the LS → source at 0° relative (front)
            az_deg = wrap_deg(float(-5 * M))
            label = angle_to_label(az_deg)

            # Pick speech (ensure at least 2 crops worth = 4s)
            speech = load_mono(speech_files[speech_idx % len(speech_files)], SAMPLE_RATE)
            speech_idx += 1
            if len(speech) < SAMPLE_RATE * 4:
                reps = int(np.ceil(SAMPLE_RATE * 4 / len(speech)))
                speech = np.tile(speech, reps)

            # Convolve
            left = fftconvolve(speech, brir_l, mode="full")[:len(speech) + len(brir_l) - 1].astype(np.float32)
            right = fftconvolve(speech, brir_r, mode="full")[:len(speech) + len(brir_r) - 1].astype(np.float32)
            stereo_clean = np.stack([left, right], axis=1)

            # Extract crops
            crops = extract_crops(stereo_clean, CROPS_PER_BRIR, crop_samples, np_rng)

            for crop_i, clean_crop in enumerate(crops):
                clean_crop = peak_normalize(clean_crop)
                # Generate all SNR variants using SAME clean crop
                for snr_val in SNR_LEVELS:
                    if snr_val == "clean":
                        mixed = clean_crop
                        snr_db = 999.0
                        noise_info = {"scene": "none", "ch_left": "none", "ch_right": "none"}
                    else:
                        noise_l, noise_r, noise_info = choose_binaural_noise(
                            noise_files, crop_samples, SAMPLE_RATE, rng)
                        noise_stereo = np.stack([noise_l, noise_r], axis=1).astype(np.float32)
                        mixed = mix_at_snr(clean_crop, noise_stereo, float(snr_val))
                        mixed = peak_normalize(mixed)
                        snr_db = float(snr_val)

                    segment_idx += 1
                    file_id = f"binaural{segment_idx:06d}"

                    sf.write(str(wav_dir / f"{file_id}.wav"), mixed, SAMPLE_RATE, subtype="PCM_16")

                    meta = {
                        "file_id": segment_idx,
                        "azimuth_deg": float(az_deg),
                        "doa_class": int(label),
                        "azimuth_bin": int(label),
                        "measurement_index": int(M),
                        "sofa_file": sofa_path.name,
                        "listener_position": int(pos_num),
                        "ls_angle": int(ls_angle_str),
                        "snr_db": snr_db,
                        "snr_label": str(snr_val),
                        "speech_path": str(speech_files[(speech_idx - 1) % len(speech_files)]),
                        "brir_source": "neidhardt_2019_zenodo_2593714",
                        "dummy_head": "KEMAR_45BA",
                        "rt60_approx": "0.3-0.5s",
                        "sample_rate": SAMPLE_RATE,
                        "segment_seconds": SEGMENT_SEC,
                        "crop_index": crop_i,
                        "rendering_mode": "measured_brir_neidhardt",
                    }
                    (meta_dir / f"metadata{segment_idx:06d}.json").write_text(
                        json.dumps(meta, indent=2), encoding="utf-8")

                    mixing_rows.append({
                        "file_id": segment_idx,
                        "azimuth_deg": float(az_deg),
                        "doa_class": int(label),
                        "sofa_file": sofa_path.name,
                        "listener_position": int(pos_num),
                        "ls_angle": int(ls_angle_str),
                        "measurement_index": int(M),
                        "snr_db": snr_db,
                        "snr_label": str(snr_val),
                        "dummy_head": "KEMAR_45BA",
                        "rt60_approx": "0.3-0.5s",
                    })

            if M % 20 == 0:
                print(f"  {pos_str} M={M:2d}/72 (segments so far: {segment_idx})", flush=True)

        print(f"Done {pos_str}: {segment_idx} segments total", flush=True)

    # Write mixing_report.csv
    import csv
    csv_path = OUTPUT_DIR / "test_all" / "mixing_report.csv"
    if mixing_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(mixing_rows[0].keys()))
            writer.writeheader()
            writer.writerows(mixing_rows)

    # Write manifest
    manifest = {
        "dataset": "librispeech_neidhardt_measured_brir_test_v1",
        "speech_source": str(SPEECH_DIR),
        "brir_source": "Neidhardt et al. DAGA 2019, Zenodo 2593714",
        "dummy_head": "KEMAR 45BA",
        "room": "small conference room, RT60 approx 0.3-0.5s",
        "sample_rate": SAMPLE_RATE,
        "segment_seconds": SEGMENT_SEC,
        "num_classes": NUM_CLASSES,
        "snr_levels": [str(s) for s in SNR_LEVELS],
        "total_segments": segment_idx,
        "sofa_files": [sf.name for sf in sofa_files],
        "label_mapping": "az = wrap_deg(-5 * M), same formula for both LS_0 and LS_180",
        "channel_mapping": "SOFA ch0 = RIGHT ear, SOFA ch1 = LEFT ear",
    }
    (OUTPUT_DIR / "test_all" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nDone. Total segments: {segment_idx}")
    print(f"WAV: {wav_dir}")
    print(f"Metadata: {meta_dir}")

if __name__ == "__main__":
    main()
