#!/usr/bin/env python3
"""Generate a small CIPIC robust pilot with full SofaMyRoom BRIR rendering."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

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
from tools.generate_cipic_robust25_pilot import (
    DEMAND_CHANNEL_PAIRS,
    DEMAND_SCENES,
    ROOM_PROFILES,
    ROOM_RANGES,
    SNR_CONDITIONS_DB,
    balanced_schedule,
    load_aligned_demand_pair,
    mix_stereo_at_snr,
    normalize_stereo,
)
from tools.generate_kemar_sofamyroom_dataset import (
    DEFAULT_CONDA_LIB,
    DEFAULT_SOFAMYROOM_BIN,
    RoomSpec,
    choose_geometry,
    default_absorption_scale,
    eyring_absorption_for_rt60,
    estimate_rt60_from_ir,
    make_absorption_matrix,
    render_reverberant_speech,
    resample_nd,
    write_sofamyroom_setup,
)


SMALL_SUBJECT_SPLITS = {
    "train": SUBJECT_SPLITS["train"][:10],
    "val": SUBJECT_SPLITS["val"][:3],
    "test": SUBJECT_SPLITS["test"][:3],
}


def sample_rooms(split: str, seed: int) -> Dict[str, RoomSpec]:
    rng = random.Random(seed)
    rooms = {}
    for profile in ROOM_PROFILES:
        ranges = ROOM_RANGES[profile]
        dims = (
            rng.uniform(*ranges["x_m"]),
            rng.uniform(*ranges["y_m"]),
            rng.uniform(*ranges["z_m"]),
        )
        rt60 = rng.uniform(*ranges["rt60_s"])
        rooms[profile] = RoomSpec(
            room_id=f"{split}_{profile}_r0",
            profile=profile,
            dims=dims,
            target_rt60=rt60,
        )
    return rooms


def render_split(
    split: str,
    subject_ids: Sequence[str],
    speech_root: Path,
    args: argparse.Namespace,
    seed: int,
) -> Dict[str, object]:
    split_dir = {
        "train": "train_subjects",
        "val": "val_subjects",
        "test": "test_subjects_unseen",
    }[split]
    split_root = args.output_root / split_dir
    wav_root = split_root / "binaural"
    brir_root = split_root / "brir"
    setup_root = split_root / "sofamyroom_setup"
    wav_root.mkdir(parents=True, exist_ok=False)
    brir_root.mkdir(parents=True, exist_ok=False)
    setup_root.mkdir(parents=True, exist_ok=False)

    rooms = sample_rooms(split, seed + 100)
    total_samples = len(subject_ids) * len(CLASS_ANGLES_DEG)
    schedule_rng = random.Random(seed + 200)
    profiles = balanced_schedule(ROOM_PROFILES, total_samples, schedule_rng)
    snr_conditions = balanced_schedule(SNR_CONDITIONS_DB, total_samples, schedule_rng)
    scenes = balanced_schedule(DEMAND_SCENES, total_samples, schedule_rng)
    channel_pairs = balanced_schedule(DEMAND_CHANNEL_PAIRS, total_samples, schedule_rng)
    sources = choose_source_files(list_audio_files(speech_root), len(subject_ids), seed + 300)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{args.conda_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    brir_samples = int(round(args.duration_sec * args.brir_fs))
    model_samples = int(round(args.duration_sec * args.sample_rate))
    rows: List[Dict[str, object]] = []
    profile_counts: Counter = Counter()
    snr_counts: Counter = Counter()
    scene_counts: Counter = Counter()
    sample_index = 0

    for subject_position, (subject_id, source_path) in enumerate(zip(subject_ids, sources), start=1):
        sofa_path = args.hrtf_root / f"subject_{subject_id}.sofa"
        subject = CIPICSubject(sofa_path, args.sample_rate)
        crop_rng = np.random.default_rng(seed * 1_000_003 + subject_position)
        speech_48k = load_active_segment(source_path, args.brir_fs, brir_samples, crop_rng)

        for class_index, angle_deg in enumerate(CLASS_ANGLES_DEG):
            profile = profiles[sample_index]
            room = rooms[profile]
            distance = schedule_rng.uniform(*ROOM_RANGES[profile]["distance_m"])
            receiver_xyz, source_xyz, distance = choose_geometry(
                room.dims,
                profile,
                angle_deg,
                schedule_rng,
                split,
                sample_index + 1,
                forced_distance=distance,
            )
            absorption_scale = default_absorption_scale(profile)
            alpha = eyring_absorption_for_rt60(room.dims, room.target_rt60, absorption_scale)
            absorption = make_absorption_matrix(alpha)

            file_id = f"{split}_s{subject_id}_c{class_index:02d}"
            setup_path = setup_root / f"{file_id}.txt"
            output_prefix = brir_root / file_id
            write_sofamyroom_setup(
                setup_path,
                output_prefix,
                sofa_path,
                room,
                absorption,
                source_xyz,
                receiver_xyz,
                args.brir_fs,
                args.brir_duration_sec,
                args.reflection_order,
                True,
                args.number_of_rays,
            )
            subprocess.run(
                [str(args.sofamyroom_bin), str(setup_path)],
                check=True,
                env=env,
                stdout=subprocess.DEVNULL,
            )
            brir_path = brir_root / f"{file_id}_receiver_0.wav"
            brir, brir_sr = sf.read(brir_path, dtype="float32", always_2d=True)
            if brir_sr != args.brir_fs or brir.shape[1] != 2:
                raise RuntimeError(f"Invalid SofaMyRoom BRIR: {brir_path}, sr={brir_sr}, shape={brir.shape}")

            clean_reverb_48k = render_reverberant_speech(speech_48k, brir)
            clean_reverb = resample_nd(
                clean_reverb_48k,
                args.brir_fs,
                args.sample_rate,
                axis=0,
            )[:model_samples]
            clean_reverb = normalize_stereo(clean_reverb)

            snr_db = snr_conditions[sample_index]
            scene = scenes[sample_index]
            channel_pair = channel_pairs[sample_index]
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
                    args.demand_root,
                    scene,
                    channel_pair,
                    args.sample_rate,
                    model_samples,
                    schedule_rng,
                )
                mixed, achieved_snr = mix_stereo_at_snr(clean_reverb, noise, float(snr_db))
                scene_name = scene
                snr_name = f"{float(snr_db):g}"

            relative_wav = Path("binaural") / f"{file_id}.wav"
            sf.write(split_root / relative_wav, mixed, args.sample_rate, subtype="PCM_16")
            measurement_index, _, sofa_az, sofa_el = subject.measured_frontal_hrir(angle_deg)
            estimated_rt60 = estimate_rt60_from_ir(brir.mean(axis=1), args.brir_fs)
            rows.append({
                "file_id": file_id,
                "wav_path": str(relative_wav),
                "azimuth_deg": angle_deg,
                "class_index": class_index,
                "subject_id": subject_id,
                "source_path": str(source_path),
                "source_speaker_id": speaker_id(source_path),
                "sofa_path": str(sofa_path),
                "sofa_measurement_index": measurement_index,
                "sofa_azimuth_deg": f"{sofa_az:.6f}",
                "sofa_elevation_deg": f"{sofa_el:.6f}",
                "hrtf_interpolation": 1,
                "rendering_mode": "sofamyroom_path_dependent_brir",
                "room_id": room.room_id,
                "room_profile": profile,
                "room_x_m": f"{room.dims[0]:.6f}",
                "room_y_m": f"{room.dims[1]:.6f}",
                "room_z_m": f"{room.dims[2]:.6f}",
                "target_rt60_s": f"{room.target_rt60:.6f}",
                "estimated_brir_rt60_s": f"{estimated_rt60:.6f}",
                "absorption_scale": f"{absorption_scale:.6f}",
                "mean_absorption": f"{float(absorption.mean()):.6f}",
                "receiver_xyz_m": ",".join(f"{v:.6f}" for v in receiver_xyz),
                "source_xyz_m": ",".join(f"{v:.6f}" for v in source_xyz),
                "source_distance_m": f"{distance:.6f}",
                "snr_condition_db": "clean" if snr_db is None else f"{float(snr_db):.6f}",
                "achieved_snr_db": "" if achieved_snr is None else f"{achieved_snr:.6f}",
                "demand_scene": scene_name,
                "noise_channel_left": noise_left,
                "noise_channel_right": noise_right,
                "noise_start_s": "" if noise_start_s is None else f"{noise_start_s:.6f}",
                "brir_path": str(brir_path),
                "sofamyroom_setup_path": str(setup_path),
            })
            profile_counts[profile] += 1
            snr_counts[snr_name] += 1
            scene_counts[scene_name] += 1
            sample_index += 1

        print(f"[{split}] subject {subject_position}/{len(subject_ids)}: {subject_id}", flush=True)

    with (split_root / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
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
                "room_id": room.room_id,
                "profile": room.profile,
                "dims_m": room.dims,
                "target_rt60_s": room.target_rt60,
            }
            for room in rooms.values()
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, default=Path("data/librispeech_cipic_sofamyroom25_pilot_v1"))
    parser.add_argument("--hrtf_root", type=Path, default=Path("/disk2/bywang/data/HRTF"))
    parser.add_argument("--demand_root", type=Path, default=Path("/disk2/bywang/data/demand"))
    parser.add_argument("--train_speech_root", type=Path, default=Path("/disk2/bywang/data/LibriSpeech/train-clean-100"))
    parser.add_argument("--val_speech_root", type=Path, default=Path("/disk2/bywang/data/LibriSpeech/LibriSpeech_dev/dev-clean"))
    parser.add_argument("--test_speech_root", type=Path, default=Path("/disk2/bywang/data/LibriSpeech/LibriSpeech_test/test-clean"))
    parser.add_argument("--sofamyroom_bin", type=Path, default=DEFAULT_SOFAMYROOM_BIN)
    parser.add_argument("--conda_lib", type=Path, default=DEFAULT_CONDA_LIB)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--brir_fs", type=int, default=48000)
    parser.add_argument("--duration_sec", type=float, default=2.0)
    parser.add_argument("--brir_duration_sec", type=float, default=1.2)
    parser.add_argument("--reflection_order", type=int, default=10)
    parser.add_argument("--number_of_rays", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Output already exists: {args.output_root}")
    if not args.sofamyroom_bin.is_file():
        raise FileNotFoundError(args.sofamyroom_bin)
    check_disjoint(SMALL_SUBJECT_SPLITS.values(), "CIPIC subject")
    args.output_root.mkdir(parents=True, exist_ok=False)

    speech_roots = {
        "train": args.train_speech_root,
        "val": args.val_speech_root,
        "test": args.test_speech_root,
    }
    reports = {}
    for split_index, split in enumerate(("train", "val", "test")):
        reports[split] = render_split(
            split,
            SMALL_SUBJECT_SPLITS[split],
            speech_roots[split],
            args,
            args.seed + split_index * 10_000,
        )
    check_disjoint(
        [reports[split]["source_speakers"] for split in ("train", "val", "test")],
        "speech speaker",
    )

    manifest = {
        "name": "librispeech_cipic_sofamyroom25_pilot_v1",
        "purpose": "small full-room/noise pipeline inspection before scaling to 30/6/9 subjects",
        "subject_split": SMALL_SUBJECT_SPLITS,
        "parent_subject_protocol": "first 10/3/3 subjects from the DP-RTF CIPIC 30/6/9 split",
        "class_angles_deg": CLASS_ANGLES_DEG,
        "sample_rate": args.sample_rate,
        "brir_sample_rate": args.brir_fs,
        "duration_sec": args.duration_sec,
        "brir_duration_sec": args.brir_duration_sec,
        "renderer": "SofaMyRoom 1.0.0a",
        "rendering": "path-dependent specular + diffuse BRIR",
        "reflection_order_xyz": [args.reflection_order] * 3,
        "simulate_diffuse": True,
        "number_of_rays": args.number_of_rays,
        "air_absorption": True,
        "distance_attenuation": True,
        "sofa_options": "interp=1 norm=1 resampling=1",
        "direct_direction_policy": "all labels are exact measured CIPIC horizontal directions",
        "reflection_direction_policy": "libmysofa interpolation for arbitrary path-arrival directions",
        "room_ranges": ROOM_RANGES,
        "snr_conditions_db": ["clean" if value is None else value for value in SNR_CONDITIONS_DB],
        "snr_definition": "joint stereo reverberant-speech power / aligned stereo noise power",
        "demand_scenes": list(DEMAND_SCENES),
        "noise_channels": [list(pair) for pair in DEMAND_CHANNEL_PAIRS],
        "noise_limitation": "DEMAND synchronized microphone pairs are multichannel environmental noise, not dummy-head binaural noise.",
        "normalization": "joint two-ear RMS target 0.08 before noise; final joint peak cap 0.98",
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
