#!/usr/bin/env python3
"""Generate a KEMAR + SofaMyRoom moving-source test-only dataset.

This script is intentionally independent from the existing static and old
moving/CIPIC pipelines so it does not affect previous datasets or training.
It renders 2 s moving binaural utterances and stores a center-time DOA label
for evaluating static models under source motion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prepare_robust_multisubject_dataset import (  # noqa: E402
    fit_to_length,
    list_librispeech_files,
    list_noise_files,
    load_mono_resampled,
    mix_at_snr,
    peak_normalize,
    read_noise_segment,
    wrap_deg,
)
from tools.generate_kemar_sofamyroom_dataset import (  # noqa: E402
    DEFAULT_CONDA_LIB,
    DEFAULT_KEMAR_SOFA,
    DEFAULT_SOFAMYROOM_BIN,
    RoomSpec,
    choose_geometry,
    default_absorption_scale,
    estimate_rt60_from_ir,
    eyring_absorption_for_rt60,
    horizontal_wall_clearance,
    make_absorption_matrix,
    peak_normalize_with_gain,
    render_reverberant_speech,
    resample_nd,
    write_sofamyroom_setup,
)
from prepare_moving_dataset import choose_moving_room_geometry  # noqa: E402


TEST_ROOMS = [
    RoomSpec("S1", "small", (4.2, 3.8, 2.6), 0.30),
    RoomSpec("L1", "large", (8.8, 6.8, 3.4), 0.70),
]
CENTER_ANGLES = list(range(0, 360, 15))
NOISE_SCENES = ["TBUS", "NPARK"]
SNR_VALUES = [0.0, -5.0, -10.0]
TRAJECTORY_TYPES = ["linear", "piecewise"]
SPEED_VALUES = [20.0, 40.0]


@dataclass(frozen=True)
class MotionCase:
    room: RoomSpec
    center_angle_deg: float
    trajectory_type: str
    speed_deg_per_sec: float
    noise_scene: str
    snr_db: float


def angle_to_class(angle_deg: float) -> int:
    return int(round(wrap_deg(angle_deg) % 360.0 / 5.0)) % 72


def resample_ir(ir: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return ir.astype(np.float32, copy=False)
    g = math.gcd(int(orig_sr), int(target_sr))
    return resample_poly(ir, target_sr // g, orig_sr // g, axis=0).astype(np.float32, copy=False)


def load_brir_from_wav(path: Path, target_sr: int) -> np.ndarray:
    brir, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if brir.shape[1] != 2:
        raise RuntimeError(f"Unexpected BRIR shape at {path}: {brir.shape}")
    return resample_ir(brir, sr, target_sr)


def build_motion_cases() -> List[MotionCase]:
    cases: List[MotionCase] = []
    for room in TEST_ROOMS:
        for center_angle_deg in CENTER_ANGLES:
            for trajectory_type in TRAJECTORY_TYPES:
                for speed_deg_per_sec in SPEED_VALUES:
                    for noise_scene in NOISE_SCENES:
                        for snr_db in SNR_VALUES:
                            cases.append(
                                MotionCase(
                                    room=room,
                                    center_angle_deg=float(center_angle_deg),
                                    trajectory_type=trajectory_type,
                                    speed_deg_per_sec=float(speed_deg_per_sec),
                                    noise_scene=noise_scene,
                                    snr_db=float(snr_db),
                                )
                            )
    return cases


def build_angle_sequence(
    center_angle_deg: float,
    trajectory_type: str,
    speed_deg_per_sec: float,
    duration_sec: float,
    chunk_seconds: float,
) -> np.ndarray:
    steps = int(round(duration_sec / chunk_seconds))
    times = np.arange(steps, dtype=np.float64) * float(chunk_seconds)
    center_time = duration_sec / 2.0
    if trajectory_type == "linear":
        direction = 1.0 if (int(center_angle_deg) // 15) % 2 == 0 else -1.0
        seq = center_angle_deg + direction * speed_deg_per_sec * (times - center_time)
    elif trajectory_type == "piecewise":
        direction = 1.0 if (int(center_angle_deg) // 15) % 2 == 0 else -1.0
        seq = np.empty_like(times)
        for i, t in enumerate(times):
            if t <= center_time:
                seq[i] = center_angle_deg + direction * speed_deg_per_sec * (t - center_time)
            else:
                seq[i] = center_angle_deg - direction * speed_deg_per_sec * (t - center_time)
    else:
        raise ValueError(f"Unsupported trajectory type: {trajectory_type}")
    return np.asarray([wrap_deg(v) for v in seq], dtype=np.float32)


def render_moving_from_brirs(
    speech_48k: np.ndarray,
    brirs_48k: Sequence[np.ndarray],
    chunk_samples_48k: int,
    crossfade_samples_48k: int,
    out_len_48k: int,
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for idx, brir in enumerate(brirs_48k):
        start = idx * chunk_samples_48k
        dry = speech_48k[start:start + chunk_samples_48k]
        if len(dry) < chunk_samples_48k:
            dry = np.pad(dry, (0, chunk_samples_48k - len(dry)), mode="constant")
        chunk = render_reverberant_speech(dry, brir)
        chunks.append(chunk)

    out = chunks[0].copy()
    for chunk in chunks[1:]:
        if crossfade_samples_48k > 0:
            fade = np.linspace(0.0, 1.0, crossfade_samples_48k, endpoint=False, dtype=np.float32)[:, None]
            out[-crossfade_samples_48k:] = out[-crossfade_samples_48k:] * (1.0 - fade) + chunk[:crossfade_samples_48k] * fade
            out = np.concatenate([out, chunk[crossfade_samples_48k:]], axis=0)
        else:
            out = np.concatenate([out, chunk], axis=0)
    if len(out) < out_len_48k:
        out = np.pad(out, ((0, out_len_48k - len(out)), (0, 0)), mode="constant")
    return out[:out_len_48k].astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_root", type=Path, required=True)
    p.add_argument("--librispeech_root", type=Path, default=Path("/disk2/bywang/data/LibriSpeech/train-clean-100"))
    p.add_argument("--demand_root", type=Path, default=Path("/disk2/bywang/data/demand"))
    p.add_argument("--sofamyroom_bin", type=Path, default=DEFAULT_SOFAMYROOM_BIN)
    p.add_argument("--sofa_path", type=Path, default=DEFAULT_KEMAR_SOFA)
    p.add_argument("--conda_lib", type=Path, default=DEFAULT_CONDA_LIB)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--brir_fs", type=int, default=48000)
    p.add_argument("--duration_sec", type=float, default=2.0)
    p.add_argument("--chunk_seconds", type=float, default=0.25)
    p.add_argument("--crossfade_ms", type=float, default=20.0)
    p.add_argument("--distance_m", type=float, default=1.2)
    p.add_argument("--brir_duration_sec", type=float, default=1.2)
    p.add_argument("--reflection_order", type=int, default=10)
    p.add_argument("--number_of_rays", type=int, default=2000)
    p.add_argument("--save_mode", choices=["full", "minimal"], default="minimal")
    p.add_argument("--limit", type=int, default=0, help="Debug: generate only the first N cases. 0 means all.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log_interval", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)

    split_dir = args.output_root / "test"
    wav_dir = split_dir / "binaural"
    setup_dir = split_dir / "sofamyroom_setup"
    brir_dir = split_dir / "brir"
    for d in (wav_dir, setup_dir, brir_dir):
        d.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{args.conda_lib}:{env.get('LD_LIBRARY_PATH', '')}"

    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    speech_files = list_librispeech_files(args.librispeech_root)
    noise_files = list_noise_files(args.demand_root, NOISE_SCENES)
    motion_cases = build_motion_cases()
    if args.limit > 0:
        motion_cases = motion_cases[: args.limit]

    num_model_samples = int(round(args.duration_sec * args.sample_rate))
    num_brir_samples = int(round(args.duration_sec * args.brir_fs))
    chunk_samples_48k = int(round(args.chunk_seconds * args.brir_fs))
    crossfade_samples_48k = int(round(args.crossfade_ms * args.brir_fs / 1000.0))
    metadata_path = split_dir / "metadata.csv"

    fieldnames = [
        "file_id", "split", "wav_path", "speech_path",
        "center_azimuth_deg", "center_doa_class", "label_time_sec",
        "trajectory_type", "speed_deg_per_sec", "motion_condition_id", "angle_seq_deg",
        "room_size", "room_id", "room_dims_m", "target_rt60", "estimated_rt60",
        "receiver_xyz", "source_xyz_seq", "source_distance_m",
        "receiver_wall_clearance_m", "min_source_wall_clearance_m",
        "noise_scene", "noise_path_l", "noise_path_r", "snr_db",
        "sample_rate", "duration_sec", "chunk_sec", "crossfade_ms",
        "brir_fs", "brir_duration_sec", "reflection_order", "simulate_diffuse",
        "sofamyroom_setup_dir", "brir_dir",
    ]

    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, case in enumerate(motion_cases, start=1):
            file_id = f"kemar_movtest_{idx:06d}"
            angle_seq = build_angle_sequence(
                center_angle_deg=case.center_angle_deg,
                trajectory_type=case.trajectory_type,
                speed_deg_per_sec=case.speed_deg_per_sec,
                duration_sec=args.duration_sec,
                chunk_seconds=args.chunk_seconds,
            )
            receiver_xyz, source_positions = choose_moving_room_geometry(
                case.room.dims,
                angle_seq,
                float(args.distance_m),
                py_rng,
            )
            source_xyz_seq = [np.asarray(pos, dtype=np.float64) for pos in source_positions]
            distance = float(args.distance_m)
            min_source_clearance = min(
                horizontal_wall_clearance(source_xyz, case.room.dims) for source_xyz in source_xyz_seq
            )

            absorption_scale = default_absorption_scale(case.room.profile)
            alpha = eyring_absorption_for_rt60(case.room.dims, case.room.target_rt60, absorption_scale)
            absorption = make_absorption_matrix(alpha)

            setup_case_dir = setup_dir / file_id
            brir_case_dir = brir_dir / file_id
            setup_case_dir.mkdir(parents=True, exist_ok=True)
            brir_case_dir.mkdir(parents=True, exist_ok=True)

            brirs_48k: List[np.ndarray] = []
            estimated_rt60_values: List[float] = []
            for step_idx, source_xyz in enumerate(source_xyz_seq):
                setup_path = setup_case_dir / f"step_{step_idx:02d}.txt"
                output_prefix = brir_case_dir / f"step_{step_idx:02d}"
                write_sofamyroom_setup(
                    setup_path,
                    output_prefix,
                    args.sofa_path,
                    case.room,
                    absorption,
                    source_xyz,
                    receiver_xyz,
                    args.brir_fs,
                    args.brir_duration_sec,
                    args.reflection_order,
                    True,
                    args.number_of_rays,
                )
                subprocess.run([str(args.sofamyroom_bin), str(setup_path)], check=True, env=env)
                brir_wav_path = brir_case_dir / f"step_{step_idx:02d}_receiver_0.wav"
                brir_48k = load_brir_from_wav(brir_wav_path, args.brir_fs)
                brirs_48k.append(brir_48k)
                mono_brir = 0.5 * (brir_48k[:, 0] + brir_48k[:, 1])
                estimated_rt60_values.append(estimate_rt60_from_ir(mono_brir, args.brir_fs))

            speech_path = speech_files[int(np_rng.integers(0, len(speech_files)))]
            speech_48k = fit_to_length(load_mono_resampled(speech_path, args.brir_fs), num_brir_samples, np_rng)
            moving_48k = render_moving_from_brirs(
                speech_48k=speech_48k,
                brirs_48k=brirs_48k,
                chunk_samples_48k=chunk_samples_48k,
                crossfade_samples_48k=crossfade_samples_48k,
                out_len_48k=num_brir_samples,
            )
            moving = resample_nd(moving_48k, args.brir_fs, args.sample_rate, axis=0)[:num_model_samples]
            moving = peak_normalize(moving, peak=0.90)

            scene_files = noise_files[case.noise_scene]
            noise_l_path = scene_files[int(np_rng.integers(0, len(scene_files)))]
            noise_r_path = scene_files[int(np_rng.integers(0, len(scene_files)))]
            noise_l = read_noise_segment(noise_l_path, num_model_samples, args.sample_rate, py_rng)
            noise_r = read_noise_segment(noise_r_path, num_model_samples, args.sample_rate, py_rng)
            noise = np.stack([noise_l, noise_r], axis=1)
            mixed = mix_at_snr(moving, noise, float(case.snr_db))
            mixed, gain = peak_normalize_with_gain(mixed, peak=0.95)
            moving = (moving * gain).astype(np.float32, copy=False)

            wav_path = wav_dir / f"{file_id}.wav"
            sf.write(str(wav_path), mixed, args.sample_rate, subtype="PCM_16")

            if args.save_mode == "minimal":
                for p in brir_case_dir.glob("*_receiver_0.wav"):
                    p.unlink(missing_ok=True)
                for p in brir_case_dir.glob("*.wav"):
                    p.unlink(missing_ok=True)
                for p in setup_case_dir.glob("*.txt"):
                    p.unlink(missing_ok=True)

            row = {
                "file_id": file_id,
                "split": "test",
                "wav_path": str(wav_path),
                "speech_path": str(speech_path),
                "center_azimuth_deg": f"{case.center_angle_deg:.6f}",
                "center_doa_class": angle_to_class(case.center_angle_deg),
                "label_time_sec": f"{args.duration_sec / 2.0:.6f}",
                "trajectory_type": case.trajectory_type,
                "speed_deg_per_sec": f"{case.speed_deg_per_sec:.6f}",
                "motion_condition_id": f"{case.trajectory_type}_{int(case.speed_deg_per_sec)}",
                "angle_seq_deg": ";".join(f"{float(v):.6f}" for v in angle_seq),
                "room_size": case.room.profile,
                "room_id": case.room.room_id,
                "room_dims_m": f"{case.room.dims[0]:.3f}x{case.room.dims[1]:.3f}x{case.room.dims[2]:.3f}",
                "target_rt60": f"{case.room.target_rt60:.6f}",
                "estimated_rt60": f"{float(np.nanmean(estimated_rt60_values)):.6f}",
                "receiver_xyz": f"{receiver_xyz[0]:.6f},{receiver_xyz[1]:.6f},{receiver_xyz[2]:.6f}",
                "source_xyz_seq": ";".join(
                    f"{xyz[0]:.6f},{xyz[1]:.6f},{xyz[2]:.6f}" for xyz in source_xyz_seq
                ),
                "source_distance_m": f"{distance:.6f}",
                "receiver_wall_clearance_m": f"{horizontal_wall_clearance(receiver_xyz, case.room.dims):.6f}",
                "min_source_wall_clearance_m": f"{min_source_clearance:.6f}",
                "noise_scene": case.noise_scene,
                "noise_path_l": str(noise_l_path),
                "noise_path_r": str(noise_r_path),
                "snr_db": f"{case.snr_db:.6f}",
                "sample_rate": args.sample_rate,
                "duration_sec": f"{args.duration_sec:.6f}",
                "chunk_sec": f"{args.chunk_seconds:.6f}",
                "crossfade_ms": f"{args.crossfade_ms:.6f}",
                "brir_fs": args.brir_fs,
                "brir_duration_sec": f"{args.brir_duration_sec:.6f}",
                "reflection_order": args.reflection_order,
                "simulate_diffuse": 1,
                "sofamyroom_setup_dir": "" if args.save_mode == "minimal" else str(setup_case_dir),
                "brir_dir": "" if args.save_mode == "minimal" else str(brir_case_dir),
            }
            writer.writerow(row)
            f.flush()
            if idx == 1 or idx % args.log_interval == 0 or idx == len(motion_cases):
                print(
                    f"[{idx}/{len(motion_cases)}] wrote {file_id} "
                    f"center={case.center_angle_deg:.1f} traj={case.trajectory_type} "
                    f"speed={case.speed_deg_per_sec:.1f} room={case.room.room_id} "
                    f"scene={case.noise_scene} snr={case.snr_db:.1f}",
                    flush=True,
                )

    manifest = {
        "dataset": "kemar_sofamyroom_moving_test_centerlabel_v1",
        "split": "test",
        "cases": len(motion_cases),
        "center_angles_deg": CENTER_ANGLES,
        "trajectory_types": TRAJECTORY_TYPES,
        "speed_deg_per_sec": SPEED_VALUES,
        "rooms": [
            {
                "room_id": room.room_id,
                "room_size": room.profile,
                "dims_m": room.dims,
                "rt60": room.target_rt60,
            }
            for room in TEST_ROOMS
        ],
        "noise_scenes": NOISE_SCENES,
        "snr_db": SNR_VALUES,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done: {metadata_path}")


if __name__ == "__main__":
    main()
