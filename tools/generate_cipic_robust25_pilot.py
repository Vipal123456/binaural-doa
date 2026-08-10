#!/usr/bin/env python3
"""Generate a small reverberant/noisy CIPIC 25-direction pilot dataset.

The renderer intentionally uses a mono shoebox-room RIR followed by the exact
measured CIPIC HRIR at the target direction.  It avoids direct-path HRTF
interpolation, but it is an approximation: all reflections inherit the target
direction's HRIR instead of receiving path-dependent HRTFs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyroomacoustics as pra
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_cipic_direct25_pilot import (
    CIPICSubject,
    CLASS_ANGLES_DEG,
    SUBJECT_SPLITS,
    check_disjoint,
    choose_source_files,
    list_audio_files,
    load_active_segment,
    speaker_id,
)
from prepare_robust_multisubject_dataset import estimate_rt60_from_ir


ROOM_RANGES = {
    "small": {
        "x_m": (3.5, 5.0),
        "y_m": (3.0, 4.5),
        "z_m": (2.5, 3.0),
        "rt60_s": (0.20, 0.40),
        "distance_m": (0.8, 1.5),
    },
    "medium": {
        "x_m": (5.0, 7.5),
        "y_m": (4.0, 6.5),
        "z_m": (2.7, 3.2),
        "rt60_s": (0.35, 0.60),
        "distance_m": (1.0, 2.5),
    },
    "large": {
        "x_m": (7.5, 10.0),
        "y_m": (6.0, 8.5),
        "z_m": (3.0, 3.8),
        "rt60_s": (0.55, 0.80),
        "distance_m": (1.5, 4.0),
    },
}
ROOM_PROFILES = tuple(ROOM_RANGES)
SNR_CONDITIONS_DB: Tuple[Optional[float], ...] = (None, 20.0, 10.0, 5.0, 0.0, -5.0)
DEMAND_SCENES = ("OOFFICE", "PCAFETER", "TMETRO", "TBUS", "SPSQUARE", "NPARK")
DEMAND_CHANNEL_PAIRS = tuple((index, index + 1) for index in range(1, 16, 2))


@dataclass
class RoomPath:
    split: str
    room_id: str
    profile: str
    dims_m: Tuple[float, float, float]
    target_rt60_s: float
    estimated_rt60_s: float
    absorption: float
    max_order: int
    angle_deg: float
    head_xyz_m: np.ndarray
    source_xyz_m: np.ndarray
    source_distance_m: float
    rir: np.ndarray


def balanced_schedule(values: Sequence, count: int, rng: random.Random) -> List:
    repeats = int(math.ceil(count / len(values)))
    schedule = list(values) * repeats
    schedule = schedule[:count]
    rng.shuffle(schedule)
    return schedule


def sample_room_spec(profile: str, rng: random.Random) -> Tuple[Tuple[float, float, float], float]:
    ranges = ROOM_RANGES[profile]
    dims = (
        rng.uniform(*ranges["x_m"]),
        rng.uniform(*ranges["y_m"]),
        rng.uniform(*ranges["z_m"]),
    )
    return dims, rng.uniform(*ranges["rt60_s"])


def choose_geometry(
    dims: Tuple[float, float, float],
    angle_deg: float,
    distance_range: Tuple[float, float],
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, float]:
    lx, ly, lz = dims
    height = min(1.5, lz - 0.7)
    margin = 0.55
    azimuth = math.radians(angle_deg)
    for _ in range(300):
        head = np.asarray([
            rng.uniform(1.0, lx - 1.0),
            rng.uniform(1.0, ly - 1.0),
            height,
        ])
        distance = rng.uniform(*distance_range)
        source = head + np.asarray([
            distance * math.sin(azimuth),
            distance * math.cos(azimuth),
            0.0,
        ])
        if (
            margin <= source[0] <= lx - margin
            and margin <= source[1] <= ly - margin
            and margin <= source[2] <= lz - margin
        ):
            return head, source, float(distance)
    raise RuntimeError(
        f"Unable to place source for room={dims}, angle={angle_deg}, distance={distance_range}"
    )


def synthesize_rir(
    sample_rate: int,
    dims: Tuple[float, float, float],
    rt60_s: float,
    head_xyz: np.ndarray,
    source_xyz: np.ndarray,
) -> Tuple[np.ndarray, float, int]:
    absorption, max_order = pra.inverse_sabine(rt60_s, dims)
    room = pra.ShoeBox(
        dims,
        fs=sample_rate,
        materials=pra.Material(float(absorption)),
        max_order=int(max_order),
    )
    room.add_source(source_xyz)
    room.add_microphone_array(head_xyz.reshape(3, 1))
    room.compute_rir()
    rir = np.asarray(room.rir[0][0], dtype=np.float32)
    if rir.size == 0 or float(np.max(np.abs(rir))) < 1e-8:
        raise RuntimeError("pyroomacoustics produced an empty RIR")
    rir /= float(np.max(np.abs(rir)))
    return rir, float(absorption), int(max_order)


def build_room_bank(split: str, sample_rate: int, seed: int) -> Dict[Tuple[str, float], RoomPath]:
    rng = random.Random(seed)
    bank: Dict[Tuple[str, float], RoomPath] = {}
    for profile in ROOM_PROFILES:
        dims, target_rt60 = sample_room_spec(profile, rng)
        room_id = f"{split}_{profile}_r0"
        for angle_deg in CLASS_ANGLES_DEG:
            head_xyz, source_xyz, distance = choose_geometry(
                dims,
                angle_deg,
                ROOM_RANGES[profile]["distance_m"],
                rng,
            )
            rir, absorption, max_order = synthesize_rir(
                sample_rate,
                dims,
                target_rt60,
                head_xyz,
                source_xyz,
            )
            bank[(profile, float(angle_deg))] = RoomPath(
                split=split,
                room_id=room_id,
                profile=profile,
                dims_m=dims,
                target_rt60_s=target_rt60,
                estimated_rt60_s=estimate_rt60_from_ir(rir, sample_rate),
                absorption=absorption,
                max_order=max_order,
                angle_deg=float(angle_deg),
                head_xyz_m=head_xyz,
                source_xyz_m=source_xyz,
                source_distance_m=distance,
                rir=rir,
            )
        print(f"[{split}] room bank ready: {room_id}", flush=True)
    return bank


def load_aligned_demand_pair(
    demand_root: Path,
    scene: str,
    channel_pair: Tuple[int, int],
    target_sr: int,
    length: int,
    rng: random.Random,
) -> Tuple[np.ndarray, str, str, float]:
    left_path = demand_root / scene / f"ch{channel_pair[0]:02d}.wav"
    right_path = demand_root / scene / f"ch{channel_pair[1]:02d}.wav"
    left_info = sf.info(left_path)
    right_info = sf.info(right_path)
    if left_info.samplerate != right_info.samplerate:
        raise ValueError(f"DEMAND channel sample-rate mismatch: {left_path}, {right_path}")
    native_sr = int(left_info.samplerate)
    native_length = int(math.ceil(length * native_sr / target_sr)) + 8
    available = min(left_info.frames, right_info.frames)
    start = rng.randint(0, max(0, available - native_length))
    left, _ = sf.read(left_path, start=start, frames=native_length, dtype="float32")
    right, _ = sf.read(right_path, start=start, frames=native_length, dtype="float32")
    noise = np.stack([left, right], axis=1).astype(np.float32, copy=False)
    if native_sr != target_sr:
        divisor = math.gcd(native_sr, target_sr)
        noise = resample_poly(
            noise,
            target_sr // divisor,
            native_sr // divisor,
            axis=0,
        ).astype(np.float32, copy=False)
    if len(noise) < length:
        noise = np.pad(noise, ((0, length - len(noise)), (0, 0)))
    noise = noise[:length]
    return noise, left_path.name, right_path.name, start / float(native_sr)


def stereo_power(x: np.ndarray) -> float:
    return float(np.mean(np.asarray(x, dtype=np.float64) ** 2))


def normalize_stereo(signal: np.ndarray, target_rms: float = 0.08) -> np.ndarray:
    power = stereo_power(signal)
    if power > 1e-12:
        signal = signal * (target_rms / math.sqrt(power))
    peak = float(np.max(np.abs(signal)))
    if peak > 0.98:
        signal = signal * (0.98 / peak)
    return signal.astype(np.float32, copy=False)


def mix_stereo_at_snr(
    signal: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> Tuple[np.ndarray, float]:
    noise = noise - noise.mean(axis=0, keepdims=True)
    signal_power = stereo_power(signal)
    noise_power = stereo_power(noise)
    if signal_power <= 1e-12 or noise_power <= 1e-12:
        raise RuntimeError("Cannot mix silent signal/noise")
    scale = math.sqrt(signal_power / (noise_power * (10.0 ** (snr_db / 10.0))))
    scaled_noise = noise * scale
    mixed = signal + scaled_noise
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.98:
        mixed = mixed * (0.98 / peak)
    achieved = 10.0 * math.log10(signal_power / stereo_power(scaled_noise))
    return mixed.astype(np.float32, copy=False), float(achieved)


def vector_text(values: np.ndarray) -> str:
    return ",".join(f"{float(value):.4f}" for value in values)


def render_split(
    split: str,
    subject_ids: Sequence[str],
    speech_root: Path,
    hrtf_root: Path,
    demand_root: Path,
    output_root: Path,
    sample_rate: int,
    duration_sec: float,
    seed: int,
) -> Dict[str, object]:
    split_dir_name = {"train": "train_subjects", "val": "val_subjects", "test": "test_subjects_unseen"}[split]
    split_root = output_root / split_dir_name
    wav_root = split_root / "binaural"
    wav_root.mkdir(parents=True, exist_ok=False)
    num_samples = int(round(sample_rate * duration_sec))
    room_bank = build_room_bank(split, sample_rate, seed + 101)

    sources = choose_source_files(list_audio_files(speech_root), len(subject_ids), seed + 202)
    total_samples = len(subject_ids) * len(CLASS_ANGLES_DEG)
    schedule_rng = random.Random(seed + 303)
    profiles = balanced_schedule(ROOM_PROFILES, total_samples, schedule_rng)
    snr_conditions = balanced_schedule(SNR_CONDITIONS_DB, total_samples, schedule_rng)
    noise_scenes = balanced_schedule(DEMAND_SCENES, total_samples, schedule_rng)
    channel_pairs = balanced_schedule(DEMAND_CHANNEL_PAIRS, total_samples, schedule_rng)

    rows: List[Dict[str, object]] = []
    profile_counts: Counter = Counter()
    snr_counts: Counter = Counter()
    scene_counts: Counter = Counter()
    sample_index = 0

    for subject_position, (subject_id, source_path) in enumerate(zip(subject_ids, sources), start=1):
        subject = CIPICSubject(hrtf_root / f"subject_{subject_id}.sofa", sample_rate)
        crop_rng = np.random.default_rng(seed * 1_000_003 + subject_position)
        speech = load_active_segment(source_path, sample_rate, num_samples, crop_rng)

        for class_index, angle_deg in enumerate(CLASS_ANGLES_DEG):
            profile = profiles[sample_index]
            snr_db = snr_conditions[sample_index]
            scene = noise_scenes[sample_index]
            channel_pair = channel_pairs[sample_index]
            path = room_bank[(profile, float(angle_deg))]

            roomed = fftconvolve(speech, path.rir, mode="full")[:num_samples]
            roomed_power = float(np.mean(roomed.astype(np.float64) ** 2))
            speech_power = float(np.mean(speech.astype(np.float64) ** 2))
            if roomed_power > 1e-12 and speech_power > 1e-12:
                roomed *= math.sqrt(speech_power / roomed_power)

            measurement_index, hrir, sofa_az, sofa_el = subject.measured_frontal_hrir(angle_deg)
            left = fftconvolve(roomed, hrir[0], mode="full")[:num_samples]
            right = fftconvolve(roomed, hrir[1], mode="full")[:num_samples]
            clean_reverb = normalize_stereo(np.stack([left, right], axis=1))

            noise_left = "none"
            noise_right = "none"
            noise_start_s: Optional[float] = None
            achieved_snr: Optional[float] = None
            if snr_db is None:
                mixed = clean_reverb
                scene_name = "clean"
                snr_name = "clean"
            else:
                noise, noise_left, noise_right, noise_start_s = load_aligned_demand_pair(
                    demand_root,
                    scene,
                    channel_pair,
                    sample_rate,
                    num_samples,
                    schedule_rng,
                )
                mixed, achieved_snr = mix_stereo_at_snr(clean_reverb, noise, float(snr_db))
                scene_name = scene
                snr_name = f"{float(snr_db):g}"

            file_id = f"{split}_s{subject_id}_c{class_index:02d}"
            relative_wav = Path("binaural") / f"{file_id}.wav"
            sf.write(split_root / relative_wav, mixed, sample_rate, subtype="PCM_16")
            rows.append({
                "file_id": file_id,
                "wav_path": str(relative_wav),
                "azimuth_deg": angle_deg,
                "class_index": class_index,
                "subject_id": subject_id,
                "source_path": str(source_path),
                "source_speaker_id": speaker_id(source_path),
                "sofa_measurement_index": measurement_index,
                "sofa_azimuth_deg": f"{sofa_az:.6f}",
                "sofa_elevation_deg": f"{sofa_el:.6f}",
                "rendering_mode": "mono_room_rir_then_exact_hrir",
                "room_id": path.room_id,
                "room_profile": profile,
                "room_x_m": f"{path.dims_m[0]:.4f}",
                "room_y_m": f"{path.dims_m[1]:.4f}",
                "room_z_m": f"{path.dims_m[2]:.4f}",
                "target_rt60_s": f"{path.target_rt60_s:.6f}",
                "estimated_rt60_s": f"{path.estimated_rt60_s:.6f}",
                "sabine_absorption": f"{path.absorption:.6f}",
                "image_source_max_order": path.max_order,
                "head_xyz_m": vector_text(path.head_xyz_m),
                "source_xyz_m": vector_text(path.source_xyz_m),
                "source_distance_m": f"{path.source_distance_m:.6f}",
                "snr_condition_db": "clean" if snr_db is None else f"{float(snr_db):.6f}",
                "achieved_snr_db": "" if achieved_snr is None else f"{achieved_snr:.6f}",
                "demand_scene": scene_name,
                "noise_channel_left": noise_left,
                "noise_channel_right": noise_right,
                "noise_start_s": "" if noise_start_s is None else f"{noise_start_s:.6f}",
            })
            profile_counts[profile] += 1
            snr_counts[snr_name] += 1
            scene_counts[scene_name] += 1
            sample_index += 1

        print(f"[{split}] subject {subject_position}/{len(subject_ids)}: {subject_id}", flush=True)

    metadata_path = split_root / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    return {
        "num_clips": len(rows),
        "subjects": list(subject_ids),
        "source_speakers": sorted({str(row["source_speaker_id"]) for row in rows}),
        "profile_counts": dict(sorted(profile_counts.items())),
        "snr_counts": dict(sorted(snr_counts.items())),
        "scene_counts": dict(sorted(scene_counts.items())),
        "rooms": [
            {
                "room_id": room_bank[(profile, float(CLASS_ANGLES_DEG[0]))].room_id,
                "profile": profile,
                "dims_m": room_bank[(profile, float(CLASS_ANGLES_DEG[0]))].dims_m,
                "target_rt60_s": room_bank[(profile, float(CLASS_ANGLES_DEG[0]))].target_rt60_s,
            }
            for profile in ROOM_PROFILES
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hrtf_root", type=Path, default=Path("/disk2/bywang/data/HRTF"))
    parser.add_argument("--demand_root", type=Path, default=Path("/disk2/bywang/data/demand"))
    parser.add_argument("--train_speech_root", type=Path, default=Path("/disk2/bywang/data/LibriSpeech/train-clean-100"))
    parser.add_argument("--val_speech_root", type=Path, default=Path("/disk2/bywang/data/LibriSpeech/LibriSpeech_dev/dev-clean"))
    parser.add_argument("--test_speech_root", type=Path, default=Path("/disk2/bywang/data/LibriSpeech/LibriSpeech_test/test-clean"))
    parser.add_argument("--output_root", type=Path, default=Path("data/librispeech_cipic_robust25_pilot_v1"))
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration_sec", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Output already exists: {args.output_root}")
    for scene in DEMAND_SCENES:
        for left, right in DEMAND_CHANNEL_PAIRS:
            for channel in (left, right):
                path = args.demand_root / scene / f"ch{channel:02d}.wav"
                if not path.is_file():
                    raise FileNotFoundError(path)
    check_disjoint(SUBJECT_SPLITS.values(), "CIPIC subject")

    args.output_root.mkdir(parents=True, exist_ok=False)
    speech_roots = {
        "train": args.train_speech_root,
        "val": args.val_speech_root,
        "test": args.test_speech_root,
    }
    reports = {}
    for split_index, split in enumerate(("train", "val", "test")):
        reports[split] = render_split(
            split=split,
            subject_ids=SUBJECT_SPLITS[split],
            speech_root=speech_roots[split],
            hrtf_root=args.hrtf_root,
            demand_root=args.demand_root,
            output_root=args.output_root,
            sample_rate=args.sample_rate,
            duration_sec=args.duration_sec,
            seed=args.seed + split_index * 10_000,
        )
    check_disjoint(
        [reports[split]["source_speakers"] for split in ("train", "val", "test")],
        "speech speaker",
    )

    manifest = {
        "name": "librispeech_cipic_robust25_pilot_v1",
        "purpose": "small complete-condition CIPIC unseen-subject pilot",
        "rendering_mode": "mono_room_rir_then_exact_hrir",
        "rendering_limitation": (
            "Direct HRIR is an exact CIPIC measurement, but all reflections share "
            "the target-direction HRIR; this is not a path-dependent physical BRIR."
        ),
        "hrtf_interpolation": False,
        "room_simulator": "pyroomacoustics image-source shoebox",
        "room_ranges": ROOM_RANGES,
        "snr_conditions_db": ["clean" if value is None else value for value in SNR_CONDITIONS_DB],
        "snr_definition": "joint two-channel reverberant-speech power / aligned two-channel noise power",
        "demand_scenes": list(DEMAND_SCENES),
        "demand_channel_pairs": [list(pair) for pair in DEMAND_CHANNEL_PAIRS],
        "noise_note": "Synchronous DEMAND channel pairs; real multichannel environmental noise, not dummy-head binaural noise.",
        "sample_rate": args.sample_rate,
        "duration_sec": args.duration_sec,
        "class_angles_deg": CLASS_ANGLES_DEG,
        "normalization": "one joint scalar across both ears; peak limited to 0.98",
        "split_policy": "DP-RTF CIPIC 30/6/9 subject split; LibriSpeech train/dev/test speaker-disjoint",
        "splits": reports,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({split: reports[split]["num_clips"] for split in reports}, indent=2))
    print(f"Done: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
