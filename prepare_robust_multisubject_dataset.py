#!/usr/bin/env python3
"""Prepare a robust subject-disjoint binaural DOA dataset.

The dataset is generated from mono LibriSpeech utterances, CIPIC SOFA HRTFs,
pyroomacoustics room responses, and DEMAND noise.  The key invariant is:

    metadata azimuth == selected HRTF azimuth == room source azimuth

This avoids the label/room-direction mismatch that can corrupt DOA training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pyroomacoustics as pra
import sofa
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly


DEMAND_SCENES = ["OOFFICE", "PCAFETER", "TMETRO", "TBUS", "SPSQUARE", "NPARK"]


@dataclass(frozen=True)
class SubjectSplit:
    train: List[str]
    val: List[str]
    test: List[str]


def wrap_deg(angle: float) -> float:
    return ((angle + 180.0) % 360.0) - 180.0


def angular_error_deg(a: float, b: float) -> float:
    return abs(wrap_deg(a - b))


def bin_center(bin_idx: int, num_bins: int = 72) -> float:
    return -180.0 + (bin_idx + 0.5) * (360.0 / num_bins)


def spherical_to_cartesian(az_deg: float, el_deg: float, radius: float) -> Tuple[float, float, float]:
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    x = radius * math.cos(el) * math.sin(az)
    y = radius * math.cos(el) * math.cos(az)
    z = radius * math.sin(el)
    return x, y, z


def read_metadata_azimuth(path: Path) -> float:
    arr = np.loadtxt(path, delimiter=",")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    mid = len(arr) // 2
    return float(np.degrees(np.arctan2(arr[mid, 1], arr[mid, 2])))


def write_metadata_csv(path: Path, az_deg: float, el_deg: float, radius: float) -> None:
    x, y, z = spherical_to_cartesian(az_deg, el_deg, radius)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([0, x, y, z, 0, 0, 0, 0])


def list_subjects(hrtf_root: Path) -> List[str]:
    subjects = sorted(p.stem.replace("subject_", "") for p in hrtf_root.glob("subject_*.sofa"))
    if not subjects:
        raise FileNotFoundError(f"No subject_*.sofa found in {hrtf_root}")
    return subjects


def choose_subjects(
    all_subjects: Sequence[str],
    total_subjects: int,
    seed: int,
    force_include: Sequence[str],
) -> SubjectSplit:
    if len(all_subjects) < total_subjects:
        raise ValueError(f"Need {total_subjects} subjects, found {len(all_subjects)}")

    forced = [s for s in force_include if s in all_subjects]
    remaining = [s for s in all_subjects if s not in forced]
    rng = random.Random(seed)
    rng.shuffle(remaining)
    selected = forced + remaining[: total_subjects - len(forced)]

    # Keep the historical deterministic/readable order after selection.
    selected = sorted(selected)
    n_train = int(round(total_subjects * 0.8))
    n_val = max(1, int(round(total_subjects * 0.1)))
    n_test = total_subjects - n_train - n_val
    if n_test < 1:
        raise ValueError("Need at least one test subject")
    return SubjectSplit(
        train=selected[:n_train],
        val=selected[n_train : n_train + n_val],
        test=selected[n_train + n_val :],
    )


def list_librispeech_files(root: Path) -> List[Path]:
    files = sorted(root.rglob("*.flac"))
    if not files:
        raise FileNotFoundError(f"No .flac files found under {root}")
    return files


def list_noise_files(demand_root: Path, scenes: Sequence[str]) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for scene in scenes:
        scene_dir = demand_root / scene
        wavs = sorted(scene_dir.glob("ch*.wav"))
        if not wavs:
            raise FileNotFoundError(f"No ch*.wav noise files found in {scene_dir}")
        out[scene] = wavs
    return out


def resample_1d(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return x.astype(np.float32, copy=False)
    g = math.gcd(int(orig_sr), int(target_sr))
    return resample_poly(x, target_sr // g, orig_sr // g).astype(np.float32, copy=False)


def load_mono_resampled(path: Path, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(path, always_2d=False, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return resample_1d(np.asarray(audio, dtype=np.float32), sr, target_sr)


def fit_to_length(audio: np.ndarray, num_samples: int, rng: np.random.Generator) -> np.ndarray:
    if len(audio) == 0:
        return np.zeros(num_samples, dtype=np.float32)
    if len(audio) >= num_samples:
        max_start = len(audio) - num_samples
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        return audio[start : start + num_samples].astype(np.float32, copy=False)
    reps = int(np.ceil(num_samples / len(audio)))
    return np.tile(audio, reps)[:num_samples].astype(np.float32, copy=False)


def read_noise_segment(path: Path, length: int, target_sr: int, rng: random.Random) -> np.ndarray:
    info = sf.info(str(path))
    if info.frames >= length and info.samplerate == target_sr:
        start = rng.randint(0, max(0, info.frames - length))
        noise, sr = sf.read(str(path), start=start, frames=length, dtype="float32", always_2d=False)
    else:
        noise, sr = sf.read(str(path), dtype="float32", always_2d=False)

    noise = np.asarray(noise, dtype=np.float32)
    if noise.ndim > 1:
        noise = noise[:, 0]
    if sr != target_sr:
        noise = resample_1d(noise, sr, target_sr)
    if len(noise) < length:
        reps = int(np.ceil(length / max(1, len(noise))))
        noise = np.tile(noise, reps)
    if len(noise) > length:
        start = rng.randint(0, max(0, len(noise) - length))
        noise = noise[start : start + length]
    return noise.astype(np.float32, copy=False)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    signal = signal.astype(np.float32, copy=False)
    noise = noise.astype(np.float32, copy=False)
    noise = noise - float(np.mean(noise))

    sig_power = float(np.mean(signal.astype(np.float64) ** 2))
    noise_power = float(np.mean(noise.astype(np.float64) ** 2))
    if sig_power < 1e-12 or noise_power < 1e-12:
        return signal.copy()

    snr_linear = 10.0 ** (snr_db / 10.0)
    scale = math.sqrt(sig_power / (snr_linear * noise_power))
    return (signal + scale * noise).astype(np.float32, copy=False)


def peak_normalize(stereo: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_abs = float(np.max(np.abs(stereo)))
    if max_abs > peak and max_abs > 1e-8:
        stereo = stereo * (peak / max_abs)
    return stereo.astype(np.float32, copy=False)


def room_profile_params(profile: str, rng: random.Random) -> Tuple[Tuple[float, float, float], float]:
    if profile == "small":
        dims = (rng.uniform(3.5, 5.0), rng.uniform(3.0, 4.5), rng.uniform(2.5, 3.0))
        rt60 = rng.uniform(0.20, 0.45)
    elif profile == "medium":
        dims = (rng.uniform(5.0, 7.5), rng.uniform(4.0, 6.5), rng.uniform(2.7, 3.2))
        rt60 = rng.uniform(0.35, 0.65)
    elif profile == "large":
        dims = (rng.uniform(7.5, 10.0), rng.uniform(6.0, 8.5), rng.uniform(3.0, 3.8))
        rt60 = rng.uniform(0.50, 0.80)
    else:
        raise ValueError(f"Unknown room profile: {profile}")
    return dims, rt60


def choose_room_geometry(
    dims: Tuple[float, float, float],
    az_deg: float,
    min_distance: float,
    max_distance: float,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, float]:
    lx, ly, lz = dims
    height = min(1.5, lz - 0.7)
    margin = 0.55
    az = math.radians(az_deg)

    for _ in range(200):
        distance_hi = min(max_distance, 0.45 * min(lx, ly))
        distance_lo = min(min_distance, distance_hi)
        distance = rng.uniform(distance_lo, distance_hi)
        head = np.array([
            rng.uniform(1.0, lx - 1.0),
            rng.uniform(1.0, ly - 1.0),
            height,
        ], dtype=np.float64)
        source = head + np.array([
            distance * math.sin(az),
            distance * math.cos(az),
            0.0,
        ], dtype=np.float64)
        if (
            margin <= source[0] <= lx - margin
            and margin <= source[1] <= ly - margin
            and margin <= source[2] <= lz - margin
        ):
            return head, source, distance

    # Conservative fallback for very small rooms/awkward angles.
    head = np.array([lx / 2.0, ly / 2.0, height], dtype=np.float64)
    max_dx = (lx / 2.0 - margin) / max(abs(math.sin(az)), 1e-6)
    max_dy = (ly / 2.0 - margin) / max(abs(math.cos(az)), 1e-6)
    distance = max(0.6, min(max_distance, max_dx, max_dy))
    source = head + np.array([distance * math.sin(az), distance * math.cos(az), 0.0])
    source[0] = np.clip(source[0], margin, lx - margin)
    source[1] = np.clip(source[1], margin, ly - margin)
    return head, source, float(distance)


def synthesize_room_rir(
    sample_rate: int,
    dims: Tuple[float, float, float],
    rt60: float,
    head_center: np.ndarray,
    source_xyz: np.ndarray,
) -> np.ndarray:
    absorption, max_order = pra.inverse_sabine(rt60, dims)
    room = pra.ShoeBox(
        dims,
        fs=sample_rate,
        materials=pra.Material(absorption),
        max_order=max_order,
    )
    room.add_source(source_xyz)
    room.add_microphone_array(head_center.reshape(3, 1))
    room.compute_rir()
    rir = np.asarray(room.rir[0][0], dtype=np.float32)
    if len(rir) == 0:
        rir = np.array([1.0], dtype=np.float32)
    rir = rir / max(float(np.max(np.abs(rir))), 1e-8)
    return rir


class HRTFSubject:
    def __init__(self, sofa_path: Path, target_sr: int):
        self.sofa_path = sofa_path
        self.subject_id = sofa_path.stem.replace("subject_", "")
        db = sofa.Database.open(str(sofa_path))
        self.positions = np.asarray(db.Source.Position.get_values(), dtype=np.float64)
        ir = np.asarray(db.Data.IR.get_values(), dtype=np.float32)
        sofa_sr = int(round(float(db.Data.SamplingRate.get_values()[0])))
        if sofa_sr != target_sr:
            g = math.gcd(sofa_sr, target_sr)
            ir = resample_poly(ir, target_sr // g, sofa_sr // g, axis=-1).astype(np.float32)
        self.ir = ir
        self.azimuths = np.array([wrap_deg(float(a)) for a in self.positions[:, 0]], dtype=np.float64)
        self.elevations = np.asarray(self.positions[:, 1], dtype=np.float64)
        self.radii = np.asarray(self.positions[:, 2], dtype=np.float64)

    def pick_measurement(self, target_az: float) -> Tuple[int, float, float, float]:
        az_err = np.array([angular_error_deg(float(a), target_az) for a in self.azimuths])
        score = az_err + 2.0 * np.abs(self.elevations)
        idx = int(np.argmin(score))
        return idx, float(self.azimuths[idx]), float(self.elevations[idx]), float(self.radii[idx])

    def apply(self, mono: np.ndarray, measurement_idx: int, num_samples: int) -> np.ndarray:
        h_l = self.ir[measurement_idx, 0]
        h_r = self.ir[measurement_idx, 1]
        left = fftconvolve(mono, h_l, mode="full")[:num_samples]
        right = fftconvolve(mono, h_r, mode="full")[:num_samples]
        stereo = np.stack([left, right], axis=1)
        return peak_normalize(stereo)


def render_sample(
    speech: np.ndarray,
    subject: HRTFSubject,
    target_bin: int,
    noise_files: Dict[str, List[Path]],
    scenes: Sequence[str],
    sample_rate: int,
    num_samples: int,
    source_distance_min: float,
    source_distance_max: float,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, object]]:
    target_az = bin_center(target_bin)
    measurement_idx, az_deg, el_deg, radius = subject.pick_measurement(target_az)

    profile = rng.choice(["small", "medium", "large"])
    dims, rt60 = room_profile_params(profile, rng)
    head_center, source_xyz, source_distance = choose_room_geometry(
        dims,
        az_deg,
        source_distance_min,
        source_distance_max,
        rng,
    )
    room_rir = synthesize_room_rir(sample_rate, dims, rt60, head_center, source_xyz)

    roomed_mono = fftconvolve(speech, room_rir, mode="full")[:num_samples].astype(np.float32)
    in_rms = rms(speech)
    out_rms = rms(roomed_mono)
    if out_rms > 1e-9:
        roomed_mono *= in_rms / out_rms

    stereo = subject.apply(roomed_mono, measurement_idx, num_samples)

    scene = rng.choice(list(scenes))
    left_noise_path = rng.choice(noise_files[scene])
    right_noise_path = rng.choice(noise_files[scene])
    snr_db = rng.uniform(-10.0, 10.0)
    noise_l = read_noise_segment(left_noise_path, num_samples, sample_rate, rng)
    noise_r = read_noise_segment(right_noise_path, num_samples, sample_rate, rng)
    mixed_l = mix_at_snr(stereo[:, 0], noise_l, snr_db)
    mixed_r = mix_at_snr(stereo[:, 1], noise_r, snr_db)
    mixed = peak_normalize(np.stack([mixed_l, mixed_r], axis=1))

    report = {
        "azimuth_deg": az_deg,
        "azimuth_bin": target_bin,
        "elevation_deg": el_deg,
        "radius": radius,
        "room_profile": profile,
        "room_dims_m": f"{dims[0]:.2f}x{dims[1]:.2f}x{dims[2]:.2f}",
        "rt60_s": rt60,
        "head_center_xyz": f"{head_center[0]:.3f},{head_center[1]:.3f},{head_center[2]:.3f}",
        "source_xyz": f"{source_xyz[0]:.3f},{source_xyz[1]:.3f},{source_xyz[2]:.3f}",
        "source_distance_m": source_distance,
        "room_source_azimuth_deg": az_deg,
        "demand_scene": scene,
        "noise_ch_left": left_noise_path.name,
        "noise_ch_right": right_noise_path.name,
        "snr_db": snr_db,
    }
    return mixed, report


def ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_output_dir(path: Path, overwrite: bool, resume: bool) -> None:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if resume:
        path.mkdir(parents=True, exist_ok=True)
        return
    ensure_empty_dir(path, overwrite)


def write_manifest(output_root: Path, args: argparse.Namespace, split: SubjectSplit) -> None:
    manifest = {
        "dataset": "librispeech_cipic_multisubject_robust50h_v1",
        "speech_root": str(args.librispeech_root),
        "hrtf_root": str(args.hrtf_root),
        "demand_root": str(args.demand_root),
        "sample_rate": args.sample_rate,
        "duration_sec": args.duration_sec,
        "num_classes": args.num_classes,
        "snr_db": [-10.0, 10.0],
        "total_subjects": args.total_subjects,
        "recordings_per_subject": args.recordings_per_subject,
        "split": {
            "train_subjects": split.train,
            "val_subjects": split.val,
            "test_subjects_unseen": split.test,
        },
        "counts": {
            "train_recordings": len(split.train) * args.recordings_per_subject,
            "val_recordings": len(split.val) * args.recordings_per_subject,
            "test_recordings": len(split.test) * args.recordings_per_subject,
        },
        "invariant": "metadata_azimuth == HRTF_azimuth == room_source_azimuth",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def generate_split(
    split_name: str,
    split_dir_name: str,
    subject_ids: Sequence[str],
    args: argparse.Namespace,
    speech_files: Sequence[Path],
    noise_files: Dict[str, List[Path]],
    hrtf_cache: Dict[str, HRTFSubject],
    py_rng: random.Random,
    np_rng: np.random.Generator,
) -> None:
    split_root = args.output_root / split_dir_name
    wav_dir = split_root / "binaural_dev"
    meta_dir = split_root / "metadata_dev"
    wav_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    report_path = split_root / "mixing_report.csv"
    fieldnames = [
        "file_id",
        "split",
        "subject_id",
        "speech_path",
        "sofa_path",
        "azimuth_deg",
        "azimuth_bin",
        "elevation_deg",
        "room_profile",
        "room_dims_m",
        "rt60_s",
        "head_center_xyz",
        "source_xyz",
        "source_distance_m",
        "room_source_azimuth_deg",
        "demand_scene",
        "noise_ch_left",
        "noise_ch_right",
        "snr_db",
        "sample_rate",
        "duration_sec",
        "num_samples",
    ]

    existing_report_ids = set()
    if args.resume and report_path.is_file():
        with report_path.open(newline="", encoding="utf-8") as f:
            existing_report_ids = {row["file_id"] for row in csv.DictReader(f)}

    report_mode = "a" if args.resume and report_path.is_file() else "w"
    num_samples = int(round(args.duration_sec * args.sample_rate))
    with report_path.open(report_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if report_mode == "w":
            writer.writeheader()

        for subject_id in subject_ids:
            subject = hrtf_cache[subject_id]
            for local_idx in range(1, args.recordings_per_subject + 1):
                global_idx = local_idx
                file_id = f"{subject_id}_{global_idx:06d}"
                wav_path = wav_dir / f"binaural{file_id}.wav"
                meta_path = meta_dir / f"metadata{file_id}.csv"
                if args.resume and wav_path.is_file() and meta_path.is_file() and file_id in existing_report_ids:
                    continue

                target_bin = (local_idx - 1) % args.num_classes
                speech_path = speech_files[int(np_rng.integers(0, len(speech_files)))]
                speech = load_mono_resampled(speech_path, args.sample_rate)
                speech = fit_to_length(speech, num_samples, np_rng)

                mixed, report = render_sample(
                    speech=speech,
                    subject=subject,
                    target_bin=target_bin,
                    noise_files=noise_files,
                    scenes=args.scenes,
                    sample_rate=args.sample_rate,
                    num_samples=num_samples,
                    source_distance_min=args.source_distance_min,
                    source_distance_max=args.source_distance_max,
                    rng=py_rng,
                    np_rng=np_rng,
                )

                sf.write(str(wav_path), mixed, args.sample_rate, subtype="PCM_16")
                write_metadata_csv(
                    meta_path,
                    az_deg=float(report["azimuth_deg"]),
                    el_deg=float(report["elevation_deg"]),
                    radius=float(report["radius"]),
                )

                metadata_az = read_metadata_azimuth(meta_path)
                if angular_error_deg(metadata_az, float(report["azimuth_deg"])) > 1e-4:
                    raise RuntimeError(f"Metadata azimuth mismatch for {file_id}")

                row = {
                    "file_id": file_id,
                    "split": split_name,
                    "subject_id": subject_id,
                    "speech_path": str(speech_path),
                    "sofa_path": str(subject.sofa_path),
                    "azimuth_deg": f"{float(report['azimuth_deg']):.6f}",
                    "azimuth_bin": int(report["azimuth_bin"]),
                    "elevation_deg": f"{float(report['elevation_deg']):.6f}",
                    "room_profile": report["room_profile"],
                    "room_dims_m": report["room_dims_m"],
                    "rt60_s": f"{float(report['rt60_s']):.6f}",
                    "head_center_xyz": report["head_center_xyz"],
                    "source_xyz": report["source_xyz"],
                    "source_distance_m": f"{float(report['source_distance_m']):.6f}",
                    "room_source_azimuth_deg": f"{float(report['room_source_azimuth_deg']):.6f}",
                    "demand_scene": report["demand_scene"],
                    "noise_ch_left": report["noise_ch_left"],
                    "noise_ch_right": report["noise_ch_right"],
                    "snr_db": f"{float(report['snr_db']):.6f}",
                    "sample_rate": args.sample_rate,
                    "duration_sec": f"{args.duration_sec:.3f}",
                    "num_samples": num_samples,
                }
                writer.writerow(row)

                done = (subject_ids.index(subject_id) * args.recordings_per_subject) + local_idx
                if done % args.log_interval == 0:
                    total = len(subject_ids) * args.recordings_per_subject
                    print(f"[{split_name}] {done}/{total} generated", flush=True)


def quality_check_root(root: Path) -> Dict[str, object]:
    report_path = root / "mixing_report.csv"
    wav_dir = root / "binaural_dev"
    meta_dir = root / "metadata_dev"
    rows = list(csv.DictReader(report_path.open(newline="", encoding="utf-8")))
    az_diffs = []
    snrs = []
    rt60s = []
    subjects = {}
    for row in rows:
        file_id = row["file_id"]
        meta_az = read_metadata_azimuth(meta_dir / f"metadata{file_id}.csv")
        az = float(row["azimuth_deg"])
        room_az = float(row["room_source_azimuth_deg"])
        az_diffs.append(max(angular_error_deg(meta_az, az), angular_error_deg(room_az, az)))
        snrs.append(float(row["snr_db"]))
        rt60s.append(float(row["rt60_s"]))
        subjects[row["subject_id"]] = subjects.get(row["subject_id"], 0) + 1

    wav_count = len(list(wav_dir.glob("binaural*.wav")))
    meta_count = len(list(meta_dir.glob("metadata*.csv")))
    return {
        "root": str(root),
        "rows": len(rows),
        "wav_count": wav_count,
        "metadata_count": meta_count,
        "subjects": subjects,
        "max_azimuth_mismatch_deg": max(az_diffs) if az_diffs else None,
        "snr_min": min(snrs) if snrs else None,
        "snr_max": max(snrs) if snrs else None,
        "rt60_min": min(rt60s) if rt60s else None,
        "rt60_max": max(rt60s) if rt60s else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare robust multisubject binaural DOA dataset")
    parser.add_argument("--librispeech_root", type=Path, default=Path("/disk2/bywang/data/LibriSpeech/train-clean-100"))
    parser.add_argument("--hrtf_root", type=Path, default=Path("/disk2/bywang/data/HRTF"))
    parser.add_argument("--demand_root", type=Path, default=Path("/disk2/bywang/data/demand"))
    parser.add_argument("--output_root", type=Path, default=Path("/disk2/bywang/DOA-net/data/librispeech_cipic_multisubject_robust50h_v1"))
    parser.add_argument("--total_subjects", type=int, default=30)
    parser.add_argument("--recordings_per_subject", type=int, default=600)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration_sec", type=float, default=10.0)
    parser.add_argument("--source_distance_min", type=float, default=1.0)
    parser.add_argument("--source_distance_max", type=float, default=1.5)
    parser.add_argument("--num_classes", type=int, default=72)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_include", nargs="*", default=["003"])
    parser.add_argument("--scenes", nargs="+", default=DEMAND_SCENES)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted generation run")
    parser.add_argument("--smoke", action="store_true", help="Generate a tiny dataset for validation")
    parser.add_argument("--log_interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.output_root = args.output_root.with_name(args.output_root.name + "_smoke")
        args.total_subjects = min(args.total_subjects, 10)
        args.recordings_per_subject = min(args.recordings_per_subject, 12)
        args.log_interval = 10

    ensure_output_dir(args.output_root, args.overwrite, args.resume)
    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    all_subjects = list_subjects(args.hrtf_root)
    split = choose_subjects(all_subjects, args.total_subjects, args.seed, args.force_include)
    if not args.resume or not (args.output_root / "manifest.json").is_file():
        write_manifest(args.output_root, args, split)

    print("Selected subjects:", json.dumps({
        "train": split.train,
        "val": split.val,
        "test": split.test,
    }, indent=2), flush=True)

    speech_files = list_librispeech_files(args.librispeech_root)
    noise_files = list_noise_files(args.demand_root, args.scenes)
    needed_subjects = sorted(set(split.train + split.val + split.test))
    hrtf_cache = {
        sid: HRTFSubject(args.hrtf_root / f"subject_{sid}.sofa", args.sample_rate)
        for sid in needed_subjects
    }

    generate_split("train", "train_subjects", split.train, args, speech_files, noise_files, hrtf_cache, py_rng, np_rng)
    generate_split("val", "val_subjects", split.val, args, speech_files, noise_files, hrtf_cache, py_rng, np_rng)
    generate_split("test", "test_subjects_unseen", split.test, args, speech_files, noise_files, hrtf_cache, py_rng, np_rng)

    qc = {
        "train": quality_check_root(args.output_root / "train_subjects"),
        "val": quality_check_root(args.output_root / "val_subjects"),
        "test": quality_check_root(args.output_root / "test_subjects_unseen"),
    }
    (args.output_root / "quality_report.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print("Quality report:", json.dumps(qc, indent=2), flush=True)
    print(f"Done: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
