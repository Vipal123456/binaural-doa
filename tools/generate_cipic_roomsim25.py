#!/usr/bin/env python3
"""Generate the formal subject/room-disjoint CIPIC-Roomsim25 dataset.

The source BRIRs are those released with DP-RTF-Learning.  Project azimuths
are positive toward the listener's right, whereas the Roomsim setup uses the
opposite sign, so every project angle is mapped to ``rir_angle = -angle``.

Generation is resumable at a 25-angle task boundary.  Each completed task is
appended to ``metadata.tasks.jsonl`` before the progress counter advances.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy
import soundfile as sf
from scipy.io import loadmat, whosmat
from scipy.signal import fftconvolve, resample_poly


CLASS_ANGLES_DEG: Tuple[int, ...] = (
    -80, -65, -55, -45, -40, -35, -30, -25, -20, -15, -10, -5,
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 55, 65, 80,
)
DOA_ALL_DEG: Tuple[int, ...] = tuple(range(-90, 91, 5))
SNR_DB: Tuple[int, ...] = (-5, 0, 5, 10, 15)
DEMAND_SCENES: Tuple[str, ...] = (
    "OOFFICE", "PCAFETER", "TMETRO", "TBUS", "SPSQUARE", "NPARK",
)
DEMAND_CHANNELS: Tuple[int, ...] = tuple(range(1, 17))
TEST_CONDITIONS: Tuple[Tuple[int, int], ...] = (
    (600, -5), (600, 0), (600, 5), (600, 10), (600, 15),
    (200, 5), (400, 5), (800, 5),
)
SPLIT_TIME_RANGES = {
    "train": (0.0, 0.6),
    "val": (0.6, 0.8),
    "test": (0.8, 1.0),
}


@dataclass(frozen=True)
class RoomSpec:
    split: str
    room_id: str
    dims_m: Tuple[float, float, float]
    subjects: Tuple[str, ...]
    rt60_ms: Tuple[int, ...]
    distances_m: Tuple[float, ...]


ROOM_SPECS: Tuple[RoomSpec, ...] = (
    RoomSpec("train", "707040", (7.0, 7.0, 4.0), ("010", "028", "124"), (0, 180, 360, 540, 720, 900), (1.0, 2.0, 3.0)),
    RoomSpec("train", "806536", (8.0, 6.5, 3.6), ("011", "012", "165"), (0, 270, 540, 810), (1.5, 2.9)),
    RoomSpec("train", "506028", (5.0, 6.0, 2.8), ("044", "127", "156"), (0, 210, 420, 630, 840), (0.5, 1.5, 2.5)),
    RoomSpec("train", "456031", (4.5, 6.0, 3.1), ("015", "017", "018"), (0, 260, 520, 780), (0.8, 2.2)),
    RoomSpec("train", "708050", (7.0, 8.0, 5.0), ("048", "050", "051"), (0, 170, 340, 510, 680, 850), (1.5, 2.0, 2.5, 3.0, 3.4)),
    RoomSpec("train", "679046", (6.7, 9.0, 4.6), ("058", "059", "060"), (0, 280, 560, 840), (3.0, 3.6)),
    RoomSpec("train", "405530", (4.0, 5.5, 3.0), ("134", "135", "137"), (0, 250, 500, 750), (0.5, 1.0)),
    RoomSpec("train", "538038", (5.3, 8.0, 3.8), ("147", "148", "152"), (0, 230, 460, 690, 920), (1.8, 2.4)),
    RoomSpec("train", "383025", (3.8, 3.0, 2.5), ("153", "154", "155"), (0, 300, 600, 900), (0.75, 1.25)),
    RoomSpec("train", "503229", (5.0, 3.2, 2.9), ("158", "162", "163"), (0, 190, 380, 570, 760, 950), (0.6, 1.2)),
    RoomSpec("val", "606035", (6.0, 6.0, 3.5), ("061", "065", "119"), (0, 220, 440, 660, 880), (1.75, 2.25)),
    RoomSpec("val", "406032", (4.0, 6.0, 3.2), ("126", "131", "133"), (0, 240, 480, 720), (0.75, 1.25)),
    RoomSpec("test", "608038", (6.0, 8.0, 3.8), ("008", "009", "033"), (200, 400, 600, 800), (0.6, 1.5, 2.4, 3.3)),
    RoomSpec("test", "507030", (5.0, 7.0, 3.0), ("021", "003", "040"), (200, 400, 600, 800), (0.7, 1.4, 2.1)),
    RoomSpec("test", "404027", (4.0, 4.0, 2.7), ("019", "020", "027"), (200, 400, 600, 800), (0.8, 1.3)),
)

ROOM_BY_SUBJECT: Dict[str, RoomSpec] = {
    subject: room for room in ROOM_SPECS for subject in room.subjects
}
SPLIT_SUBJECTS: Dict[str, Tuple[str, ...]] = {
    split: tuple(subject for room in ROOM_SPECS if room.split == split for subject in room.subjects)
    for split in ("train", "val", "test")
}
SPLIT_DIR_NAMES = {"train": "train", "val": "val", "test": "test"}
SAMPLES_PER_SUBJECT_ANGLE = {"train": 160, "val": 80}


def stable_int(*values: object) -> int:
    payload = "|".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def balanced_schedule(values: Sequence[object], count: int, seed: int) -> List[object]:
    if not values:
        raise ValueError("Cannot balance an empty value set")
    rng = random.Random(seed)
    cycle = list(values)
    rng.shuffle(cycle)
    sequence = [cycle[index % len(cycle)] for index in range(count)]
    rng.shuffle(sequence)
    return sequence


def rt_schedule(room: RoomSpec, count: int, seed: int) -> List[int]:
    """Use R0 for exactly 10% and balance the remaining reverberant RTs."""
    nonzero = tuple(value for value in room.rt60_ms if value != 0)
    zero_count = int(round(count * 0.10))
    sequence = [0] * zero_count
    sequence.extend(int(v) for v in balanced_schedule(nonzero, count - zero_count, seed + 1))
    random.Random(seed + 2).shuffle(sequence)
    return sequence


def project_to_rir_angle(project_angle_deg: int) -> int:
    return -int(project_angle_deg)


def brir_source_index(distance_index: int, project_angle_deg: int) -> int:
    rir_angle = project_to_rir_angle(project_angle_deg)
    return len(DOA_ALL_DEG) * int(distance_index) + DOA_ALL_DEG.index(rir_angle) + 1


def brir_path(
    rir_root: Path,
    subject: str,
    room: RoomSpec,
    rt60_ms: int,
    distance_index: int,
    project_angle_deg: int,
) -> Path:
    source_index = brir_source_index(distance_index, project_angle_deg)
    return rir_root / f"H{subject}_{room.room_id}" / f"R{int(rt60_ms)}_S{source_index}.mat"


def list_audio_files(root: Path) -> List[Path]:
    files = sorted(root.rglob("*.flac")) + sorted(root.rglob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No FLAC/WAV files found under {root}")
    return files


def speaker_id(path: Path) -> str:
    return path.parent.parent.name


def resample_nd(data: np.ndarray, source_sr: int, target_sr: int, axis: int = 0) -> np.ndarray:
    if int(source_sr) == int(target_sr):
        return np.asarray(data, dtype=np.float32)
    divisor = math.gcd(int(source_sr), int(target_sr))
    return resample_poly(
        data,
        int(target_sr) // divisor,
        int(source_sr) // divisor,
        axis=axis,
    ).astype(np.float32, copy=False)


def frame_active_ratio(signal: np.ndarray, sample_rate: int) -> float:
    frame = max(1, int(round(0.020 * sample_rate)))
    hop = max(1, int(round(0.010 * sample_rate)))
    if len(signal) < frame:
        return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))) > 1e-4)
    starts = range(0, len(signal) - frame + 1, hop)
    rms = np.asarray([
        np.sqrt(np.mean(np.square(signal[start : start + frame], dtype=np.float64)) + 1e-12)
        for start in starts
    ])
    threshold = max(1e-4, float(np.max(rms)) * 0.05)
    return float(np.mean(rms >= threshold))


def choose_speech_context(
    paths: Sequence[str],
    base_index: int,
    prefix: int,
    output_samples: int,
    sample_rate: int,
    seed: int,
    min_active_ratio: float = 0.70,
) -> Tuple[np.ndarray, str, int, float]:
    best: Optional[Tuple[float, float, np.ndarray, str, int]] = None
    for attempt in range(min(16, len(paths))):
        path = Path(paths[(base_index + attempt * 7919) % len(paths)])
        audio, source_sr = sf.read(path, dtype="float32", always_2d=True)
        mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
        mono = resample_nd(mono, int(source_sr), sample_rate)
        mono -= float(np.mean(mono))
        if len(mono) < output_samples:
            continue

        low = min(prefix, max(0, len(mono) - output_samples))
        high = len(mono) - output_samples
        if high < low:
            continue
        count = min(48, high - low + 1)
        candidates = np.unique(np.linspace(low, high, num=count, dtype=np.int64))
        rng = np.random.default_rng(stable_int(seed, attempt, path.name))
        candidates = candidates[rng.permutation(len(candidates))]
        for start_value in candidates:
            start = int(start_value)
            target = mono[start : start + output_samples]
            ratio = frame_active_ratio(target, sample_rate)
            power = float(np.mean(np.square(target, dtype=np.float64)))
            context_start = start - prefix
            if context_start < 0:
                context = np.pad(mono[: start + output_samples], (prefix - start, 0))
            else:
                context = mono[context_start : start + output_samples]
            candidate = (ratio, power, np.asarray(context, dtype=np.float32), str(path), start)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is not None and best[0] >= min_active_ratio:
            break

    if best is None or best[0] < min_active_ratio:
        detail = "no usable candidate" if best is None else f"best active ratio={best[0]:.3f}"
        raise RuntimeError(f"Unable to select active LibriSpeech crop: {detail}")
    ratio, _power, context, path, start = best
    expected = prefix + output_samples
    if len(context) < expected:
        context = np.pad(context, (0, expected - len(context)))
    return context[:expected], path, start, ratio


def stereo_power(signal: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(signal, dtype=np.float64))))


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> Tuple[np.ndarray, float]:
    signal_power = stereo_power(signal)
    noise = noise - np.mean(noise, axis=0, keepdims=True)
    noise_power = stereo_power(noise)
    if signal_power <= 1e-12 or noise_power <= 1e-12:
        raise RuntimeError("Cannot mix silent signal or noise")
    scale = math.sqrt(signal_power / (noise_power * (10.0 ** (float(snr_db) / 10.0))))
    scaled_noise = noise * scale
    achieved = 10.0 * math.log10(signal_power / stereo_power(scaled_noise))
    return (signal + scaled_noise).astype(np.float32, copy=False), float(achieved)


def joint_normalize(signal: np.ndarray, target_rms: float = 0.08) -> np.ndarray:
    power = stereo_power(signal)
    if power > 1e-12:
        signal = signal * (target_rms / math.sqrt(power))
    peak = float(np.max(np.abs(signal)))
    if peak > 0.98:
        signal = signal * (0.98 / peak)
    return np.asarray(signal, dtype=np.float32)


def render_context(context: np.ndarray, brir: np.ndarray, prefix: int, length: int) -> np.ndarray:
    channels = [fftconvolve(context, brir[:, channel], mode="full") for channel in range(2)]
    stereo = np.stack([channel[prefix : prefix + length] for channel in channels], axis=1)
    if stereo.shape != (length, 2):
        raise RuntimeError(f"Unexpected rendered shape: {stereo.shape}")
    return stereo.astype(np.float32, copy=False)


_WORKER_CONFIG: Dict[str, object] = {}


def init_worker(config: Mapping[str, object]) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = dict(config)
    _load_brir_cached.cache_clear()
    _load_noise_cached.cache_clear()


@lru_cache(maxsize=768)
def _load_brir_cached(path_string: str, target_sr: int) -> np.ndarray:
    payload = loadmat(path_string, variable_names=["data", "Fs"])
    data = np.asarray(payload["data"], dtype=np.float32)
    source_sr = int(round(float(payload["Fs"][0, 0])))
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Expected [time,2] BRIR at {path_string}, got {data.shape}")
    data = resample_nd(data, source_sr, int(target_sr), axis=0)
    if not np.all(np.isfinite(data)) or float(np.max(np.abs(data))) <= 1e-10:
        raise ValueError(f"Invalid BRIR data: {path_string}")
    return data


@lru_cache(maxsize=16)
def _load_noise_cached(path_string: str, target_sr: int) -> np.ndarray:
    audio, source_sr = sf.read(path_string, dtype="float32", always_2d=True)
    mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
    mono = resample_nd(mono, int(source_sr), int(target_sr))
    mono -= float(np.mean(mono))
    return mono


def demand_target_start(
    scene_path: Path,
    split: str,
    unit_interval_value: float,
    sample_rate: int,
    output_samples: int,
    prefix: int,
) -> int:
    info = sf.info(scene_path)
    frames = int(round(info.frames * sample_rate / info.samplerate))
    fraction_start, fraction_end = SPLIT_TIME_RANGES[split]
    guard = int(round(2.0 * sample_rate))
    low = int(math.floor(frames * fraction_start)) + guard + prefix
    high = int(math.floor(frames * fraction_end)) - guard - output_samples
    if high <= low:
        raise RuntimeError(f"DEMAND partition too short: {scene_path} {split}")
    value = min(max(float(unit_interval_value), 0.0), 1.0)
    return low + int(round(value * (high - low)))


def choose_noise_angle(target_angle: int, task_seed: int, class_index: int) -> int:
    candidates = [angle for angle in CLASS_ANGLES_DEG if abs(angle - target_angle) >= 20]
    return candidates[stable_int(task_seed, class_index, "noise_angle") % len(candidates)]


def render_task(task: Mapping[str, object]) -> Dict[str, object]:
    config = _WORKER_CONFIG
    sample_rate = int(config["sample_rate"])
    output_samples = int(config["output_samples"])
    rir_root = Path(str(config["rir_root"]))
    demand_root = Path(str(config["demand_root"]))
    output_root = Path(str(config["output_root"]))
    speech_paths = list(config[f"{task['split']}_speech_paths"])
    room = ROOM_BY_SUBJECT[str(task["subject_id"])]
    task_seed = int(task["task_seed"])

    target_brirs: List[np.ndarray] = []
    target_paths: List[Path] = []
    noise_brirs: List[np.ndarray] = []
    noise_paths: List[Path] = []
    noise_angles: List[int] = []
    noise_distance_indices: List[int] = []

    for class_index, angle in enumerate(CLASS_ANGLES_DEG):
        target_path = brir_path(
            rir_root, str(task["subject_id"]), room, int(task["rt60_ms"]),
            int(task["distance_index"]), angle,
        )
        noise_angle = choose_noise_angle(angle, task_seed, class_index)
        noise_distance_index = stable_int(task_seed, class_index, "noise_distance") % len(room.distances_m)
        noise_path = brir_path(
            rir_root, str(task["subject_id"]), room, int(task["rt60_ms"]),
            noise_distance_index, noise_angle,
        )
        target_paths.append(target_path)
        noise_paths.append(noise_path)
        target_brirs.append(_load_brir_cached(str(target_path), sample_rate))
        noise_brirs.append(_load_brir_cached(str(noise_path), sample_rate))
        noise_angles.append(noise_angle)
        noise_distance_indices.append(noise_distance_index)

    prefix = max(max(len(value) for value in target_brirs), max(len(value) for value in noise_brirs)) - 1
    speech_context, speech_path, speech_start, active_ratio = choose_speech_context(
        speech_paths,
        int(task["speech_index"]),
        prefix,
        output_samples,
        sample_rate,
        stable_int(task_seed, "speech"),
    )

    noise_path = demand_root / str(task["noise_scene"]) / f"ch{int(task['noise_channel']):02d}.wav"
    noise_source = _load_noise_cached(str(noise_path), sample_rate)
    noise_start = demand_target_start(
        noise_path,
        str(task["split"]),
        float(task["noise_u"]),
        sample_rate,
        output_samples,
        prefix,
    )
    noise_context = noise_source[noise_start - prefix : noise_start + output_samples]
    if len(noise_context) != prefix + output_samples:
        raise RuntimeError(f"Unexpected DEMAND context length for {noise_path}")

    rows: List[Dict[str, object]] = []
    split_root = output_root / SPLIT_DIR_NAMES[str(task["split"])]
    wav_root = split_root / "binaural"
    for class_index, angle in enumerate(CLASS_ANGLES_DEG):
        clean_reverb = render_context(speech_context, target_brirs[class_index], prefix, output_samples)
        rendered_noise = render_context(noise_context, noise_brirs[class_index], prefix, output_samples)
        mixed, achieved_snr = mix_at_snr(clean_reverb, rendered_noise, float(task["snr_db"]))
        if abs(achieved_snr - float(task["snr_db"])) > 0.01:
            raise RuntimeError(f"SNR mismatch: target={task['snr_db']} achieved={achieved_snr}")
        mixed = joint_normalize(mixed)
        if not np.all(np.isfinite(mixed)) or float(np.max(np.abs(mixed))) > 0.981:
            raise RuntimeError("Invalid generated mixture")

        file_id = f"{task['task_id']}_c{class_index:02d}"
        relative_wav = Path("binaural") / f"{file_id}.wav"
        sf.write(wav_root / relative_wav.name, mixed, sample_rate, subtype="PCM_16")
        rows.append({
            "file_id": file_id,
            "wav_path": str(relative_wav),
            "split": task["split"],
            "class_index": class_index,
            "azimuth_deg": angle,
            "rir_azimuth_deg": project_to_rir_angle(angle),
            "subject_id": task["subject_id"],
            "room_id": room.room_id,
            "room_x_m": room.dims_m[0],
            "room_y_m": room.dims_m[1],
            "room_z_m": room.dims_m[2],
            "rt60_s": float(task["rt60_ms"]) / 1000.0,
            "distance_m": room.distances_m[int(task["distance_index"])],
            "target_snr_db": task["snr_db"],
            "achieved_snr_db": round(achieved_snr, 8),
            "speech_path": speech_path,
            "speech_speaker_id": speaker_id(Path(speech_path)),
            "speech_target_start_sample": speech_start,
            "speech_active_ratio": round(active_ratio, 6),
            "noise_scene": task["noise_scene"],
            "noise_channel": task["noise_channel"],
            "noise_target_start_sample": noise_start,
            "noise_azimuth_deg": noise_angles[class_index],
            "noise_rir_azimuth_deg": project_to_rir_angle(noise_angles[class_index]),
            "noise_distance_m": room.distances_m[noise_distance_indices[class_index]],
            "target_brir_path": str(target_paths[class_index]),
            "noise_brir_path": str(noise_paths[class_index]),
            "renderer": "DP-RTF Roomsim_Campbell CIPIC BRIR",
            "seed": task_seed,
        })
    return {"task_id": task["task_id"], "rows": rows}


def make_train_or_val_tasks(split: str, seed: int, smoke: bool = False) -> List[Dict[str, object]]:
    tasks: List[Dict[str, object]] = []
    subjects = SPLIT_SUBJECTS[split]
    realizations = 1 if smoke else SAMPLES_PER_SUBJECT_ANGLE[split]
    selected_subjects = subjects[:1] if smoke else subjects
    task_counter = 0
    for subject_position, subject in enumerate(selected_subjects):
        room = ROOM_BY_SUBJECT[subject]
        if smoke:
            rt_values = [next(value for value in room.rt60_ms if value != 0)]
            distance_values = [0]
            snr_values = [-5 if split == "train" else 5]
            scene_values = [DEMAND_SCENES[subject_position % len(DEMAND_SCENES)]]
            channel_values = [1]
        else:
            subject_seed = stable_int(seed, split, subject)
            rt_values = rt_schedule(room, realizations, subject_seed + 10)
            distance_values = balanced_schedule(tuple(range(len(room.distances_m))), realizations, subject_seed + 20)
            snr_values = balanced_schedule(SNR_DB, realizations, subject_seed + 30)
            scene_values = balanced_schedule(DEMAND_SCENES, realizations, subject_seed + 40)
            channel_values = balanced_schedule(DEMAND_CHANNELS, realizations, subject_seed + 50)
        for realization in range(realizations):
            task_seed = stable_int(seed, split, subject, realization)
            task_id = f"{split}_s{subject}_r{realization:03d}"
            tasks.append({
                "task_id": task_id,
                "split": split,
                "subject_id": subject,
                "rt60_ms": int(rt_values[realization]),
                "distance_index": int(distance_values[realization]),
                "snr_db": int(snr_values[realization]),
                "noise_scene": str(scene_values[realization]),
                "noise_channel": int(channel_values[realization]),
                "noise_u": (stable_int(task_seed, "noise_start") % 1_000_001) / 1_000_000.0,
                "speech_index": subject_position * realizations + realization,
                "task_seed": task_seed,
            })
            task_counter += 1
    return tasks


def make_test_tasks(seed: int, smoke: bool = False) -> List[Dict[str, object]]:
    if smoke:
        subject = SPLIT_SUBJECTS["test"][0]
        room = ROOM_BY_SUBJECT[subject]
        return [{
            "task_id": f"test_s{subject}_d00_q00_n00_r00",
            "split": "test",
            "subject_id": subject,
            "rt60_ms": 600,
            "distance_index": 0,
            "snr_db": -5,
            "noise_scene": DEMAND_SCENES[0],
            "noise_channel": 1,
            "noise_u": 0.5,
            "speech_index": 0,
            "task_seed": stable_int(seed, "test", subject, room.room_id, "smoke"),
        }]

    tasks: List[Dict[str, object]] = []
    for room in (value for value in ROOM_SPECS if value.split == "test"):
        for distance_index, _distance in enumerate(room.distances_m):
            for condition_index, (rt60_ms, snr_db) in enumerate(TEST_CONDITIONS):
                for scene_index, scene in enumerate(DEMAND_SCENES):
                    for realization in range(2):
                        shared_key = (room.room_id, distance_index, condition_index, scene_index, realization)
                        speech_index = stable_int(seed, "test_speech", *shared_key)
                        noise_u = (stable_int(seed, "test_noise", *shared_key) % 1_000_001) / 1_000_000.0
                        channel = DEMAND_CHANNELS[stable_int(seed, "test_channel", *shared_key) % len(DEMAND_CHANNELS)]
                        for subject in room.subjects:
                            task_seed = stable_int(seed, "test", subject, *shared_key)
                            task_id = (
                                f"test_s{subject}_d{distance_index:02d}_q{condition_index:02d}_"
                                f"n{scene_index:02d}_r{realization:02d}"
                            )
                            tasks.append({
                                "task_id": task_id,
                                "split": "test",
                                "subject_id": subject,
                                "rt60_ms": rt60_ms,
                                "distance_index": distance_index,
                                "snr_db": snr_db,
                                "noise_scene": scene,
                                "noise_channel": channel,
                                "noise_u": noise_u,
                                "speech_index": speech_index,
                                "task_seed": task_seed,
                            })
    return tasks


def make_tasks(mode: str, seed: int) -> Dict[str, List[Dict[str, object]]]:
    smoke = mode == "smoke"
    return {
        "train": make_train_or_val_tasks("train", seed, smoke=smoke),
        "val": make_train_or_val_tasks("val", seed, smoke=smoke),
        "test": make_test_tasks(seed, smoke=smoke),
    }


def validate_input_layout(args: argparse.Namespace) -> Dict[str, List[str]]:
    speech_roots = {
        "train": args.train_speech_root,
        "val": args.val_speech_root,
        "test": args.test_speech_root,
    }
    speech_paths: Dict[str, List[str]] = {}
    speaker_groups = []
    for split, root in speech_roots.items():
        paths = list_audio_files(root)
        speech_paths[split] = [str(path) for path in paths]
        speaker_groups.append({speaker_id(path) for path in paths})
    check_disjoint(speaker_groups, "LibriSpeech speaker")
    check_disjoint(SPLIT_SUBJECTS.values(), "CIPIC subject")
    for scene in DEMAND_SCENES:
        for channel in DEMAND_CHANNELS:
            path = args.demand_root / scene / f"ch{channel:02d}.wav"
            if not path.is_file():
                raise FileNotFoundError(path)
    return speech_paths


def check_disjoint(groups: Iterable[Iterable[str]], name: str) -> None:
    values = [set(group) for group in groups]
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            overlap = values[left] & values[right]
            if overlap:
                raise RuntimeError(f"{name} leakage: {sorted(overlap)[:20]}")


def inventory_required_brirs(rir_root: Path, output_path: Path, include_sha256: bool) -> Dict[str, object]:
    required: Dict[str, Tuple[str, RoomSpec, int, int, int]] = {}
    for room in ROOM_SPECS:
        for subject in room.subjects:
            for rt60_ms in room.rt60_ms:
                for distance_index in range(len(room.distances_m)):
                    for angle in CLASS_ANGLES_DEG:
                        path = brir_path(rir_root, subject, room, rt60_ms, distance_index, angle)
                        required[str(path)] = (subject, room, rt60_ms, distance_index, angle)

    rows = []
    for index, path_string in enumerate(sorted(required), start=1):
        path = Path(path_string)
        if not path.is_file():
            raise FileNotFoundError(path)
        subject, room, rt60_ms, distance_index, angle = required[path_string]
        metadata = {name: shape for name, shape, _kind in whosmat(path)}
        fs = float(loadmat(path, variable_names=["Fs"])["Fs"][0, 0])
        if set(metadata) != {"Fs", "data"} or len(metadata["data"]) != 2 or metadata["data"][1] != 2:
            raise RuntimeError(f"Invalid BRIR schema: {path} {metadata}")
        digest = ""
        if include_sha256:
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
        rows.append({
            "path": path_string,
            "subject_id": subject,
            "room_id": room.room_id,
            "rt60_ms": rt60_ms,
            "distance_m": room.distances_m[distance_index],
            "project_azimuth_deg": angle,
            "rir_azimuth_deg": project_to_rir_angle(angle),
            "fs": fs,
            "num_samples": metadata["data"][0],
            "num_channels": metadata["data"][1],
            "size_bytes": path.stat().st_size,
            "sha256": digest,
        })
        if index % 2000 == 0:
            print(f"[inventory] {index}/{len(required)}", flush=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    inventory_digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return {
        "required_paths": len(rows),
        "missing_paths": 0,
        "sha256_included": include_sha256,
        "inventory_sha256": inventory_digest,
    }


def load_completed_tasks(task_jsonl: Path) -> Dict[str, Dict[str, object]]:
    completed: Dict[str, Dict[str, object]] = {}
    if not task_jsonl.is_file():
        return completed
    with task_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            task_id = str(payload["task_id"])
            if task_id in completed:
                raise RuntimeError(f"Duplicate task {task_id} in {task_jsonl}:{line_number}")
            completed[task_id] = payload
    return completed


def finalize_metadata(split_root: Path, expected_tasks: int) -> Dict[str, object]:
    task_jsonl = split_root / "metadata.tasks.jsonl"
    completed = load_completed_tasks(task_jsonl)
    if len(completed) != expected_tasks:
        raise RuntimeError(f"Incomplete split {split_root}: {len(completed)}/{expected_tasks} tasks")
    rows = [row for task_id in sorted(completed) for row in completed[task_id]["rows"]]
    file_ids = [str(row["file_id"]) for row in rows]
    if len(file_ids) != len(set(file_ids)):
        raise RuntimeError(f"Duplicate file IDs in {split_root}")
    metadata_path = split_root / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "num_tasks": len(completed),
        "num_clips": len(rows),
        "metadata_path": str(metadata_path),
    }


def generate_split(
    split: str,
    tasks: Sequence[Mapping[str, object]],
    worker_config: Mapping[str, object],
    workers: int,
) -> Dict[str, object]:
    split_root = Path(str(worker_config["output_root"])) / SPLIT_DIR_NAMES[split]
    wav_root = split_root / "binaural"
    wav_root.mkdir(parents=True, exist_ok=True)
    task_jsonl = split_root / "metadata.tasks.jsonl"
    completed = load_completed_tasks(task_jsonl)
    expected_ids = {str(task["task_id"]) for task in tasks}
    unexpected = set(completed) - expected_ids
    if unexpected:
        raise RuntimeError(f"Existing metadata contains unexpected tasks: {sorted(unexpected)[:10]}")
    pending = [task for task in tasks if str(task["task_id"]) not in completed]
    print(f"[{split}] completed={len(completed)} pending={len(pending)} total={len(tasks)}", flush=True)
    start_time = time.time()
    if pending:
        with task_jsonl.open("a", encoding="utf-8") as output_handle:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=init_worker,
                initargs=(worker_config,),
            ) as executor:
                for offset, payload in enumerate(executor.map(render_task, pending, chunksize=1), start=1):
                    output_handle.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n")
                    output_handle.flush()
                    os.fsync(output_handle.fileno())
                    done = len(completed) + offset
                    if offset == 1 or done % 10 == 0 or done == len(tasks):
                        elapsed = max(time.time() - start_time, 1e-6)
                        rate = offset / elapsed
                        eta = (len(pending) - offset) / rate if rate > 0 else float("inf")
                        progress = {
                            "split": split,
                            "completed_tasks": done,
                            "total_tasks": len(tasks),
                            "completed_clips": done * len(CLASS_ANGLES_DEG),
                            "total_clips": len(tasks) * len(CLASS_ANGLES_DEG),
                            "elapsed_sec_this_run": elapsed,
                            "eta_sec": eta,
                        }
                        (split_root / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
                        print(
                            f"[{split}] tasks {done}/{len(tasks)} clips {done * 25}/{len(tasks) * 25} "
                            f"rate={rate:.3f} task/s eta={eta / 60.0:.1f} min",
                            flush=True,
                        )
    report = finalize_metadata(split_root, len(tasks))
    report["duration_sec_this_run"] = time.time() - start_time
    return report


def handedness_check(rir_root: Path) -> Dict[str, object]:
    room = next(value for value in ROOM_SPECS if value.room_id == "507030")
    subject = "003"
    results = {}
    for project_angle in (-80, 80):
        path = brir_path(rir_root, subject, room, 0, 0, project_angle)
        data = np.asarray(loadmat(path, variable_names=["data"])["data"], dtype=np.float64)
        onset = int(np.argmax(np.max(np.abs(data), axis=1) > np.max(np.abs(data)) * 0.05))
        segment = data[onset : onset + 256]
        energy = np.sum(segment ** 2, axis=0)
        ild_lr_db = float(10.0 * np.log10((energy[0] + 1e-20) / (energy[1] + 1e-20)))
        peak_indices = np.argmax(np.abs(data), axis=0)
        results[str(project_angle)] = {
            "path": str(path),
            "ild_left_over_right_db": ild_lr_db,
            "peak_index_left": int(peak_indices[0]),
            "peak_index_right": int(peak_indices[1]),
        }
    passed = (
        results["80"]["ild_left_over_right_db"] < 0
        and results["80"]["peak_index_right"] < results["80"]["peak_index_left"]
        and results["-80"]["ild_left_over_right_db"] > 0
        and results["-80"]["peak_index_left"] < results["-80"]["peak_index_right"]
    )
    if not passed:
        raise RuntimeError(f"BRIR handedness check failed: {results}")
    return {"passed": True, "project_convention": "positive azimuth is listener right", "details": results}


def quality_check(
    output_root: Path,
    tasks_by_split: Mapping[str, Sequence[Mapping[str, object]]],
    sample_rate: int,
    output_samples: int,
    rir_root: Path,
    inspect_limit: int,
) -> Dict[str, object]:
    report: Dict[str, object] = {"passed": True, "handedness": handedness_check(rir_root), "splits": {}}
    speaker_sets = []
    for split, tasks in tasks_by_split.items():
        split_root = output_root / SPLIT_DIR_NAMES[split]
        metadata_path = split_root / "metadata.csv"
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected = len(tasks) * len(CLASS_ANGLES_DEG)
        if len(rows) != expected:
            raise RuntimeError(f"{split} metadata rows {len(rows)} != {expected}")
        class_counts = Counter(int(row["class_index"]) for row in rows)
        if set(class_counts) != set(range(len(CLASS_ANGLES_DEG))) or len(set(class_counts.values())) != 1:
            raise RuntimeError(f"Unbalanced {split} classes: {class_counts}")
        snr_error = max(abs(float(row["target_snr_db"]) - float(row["achieved_snr_db"])) for row in rows)
        if snr_error > 0.2:
            raise RuntimeError(f"{split} SNR error {snr_error}")
        speakers = {row["speech_speaker_id"] for row in rows}
        speaker_sets.append(speakers)

        step = max(1, len(rows) // max(1, inspect_limit))
        inspected = rows[::step][:inspect_limit]
        max_peak = 0.0
        min_rms = float("inf")
        for row in inspected:
            path = split_root / row["wav_path"]
            info = sf.info(path)
            if info.samplerate != sample_rate or info.channels != 2 or info.frames != output_samples:
                raise RuntimeError(f"Invalid WAV format: {path} {info}")
            audio, _ = sf.read(path, dtype="float32", always_2d=True)
            if not np.all(np.isfinite(audio)):
                raise RuntimeError(f"Non-finite WAV: {path}")
            max_peak = max(max_peak, float(np.max(np.abs(audio))))
            min_rms = min(min_rms, math.sqrt(stereo_power(audio)))
        if max_peak > 0.981 or min_rms < 1e-5:
            raise RuntimeError(f"Invalid levels in {split}: peak={max_peak} min_rms={min_rms}")
        report["splits"][split] = {
            "num_tasks": len(tasks),
            "num_clips": len(rows),
            "clips_per_class": next(iter(class_counts.values())),
            "num_speech_speakers": len(speakers),
            "max_snr_error_db": snr_error,
            "wav_files_inspected": len(inspected),
            "max_abs_peak": max_peak,
            "min_rms": min_rms,
        }
    check_disjoint(speaker_sets, "generated speech speaker")
    return report


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def write_manifest(
    args: argparse.Namespace,
    output_root: Path,
    tasks_by_split: Mapping[str, Sequence[Mapping[str, object]]],
    inventory: Mapping[str, object],
    split_reports: Mapping[str, object],
    quality: Mapping[str, object],
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    manifest = {
        "name": "librispeech_cipic_roomsim25_v1",
        "mode": args.mode,
        "created_unix_time": time.time(),
        "generator": str(Path(__file__).resolve()),
        "git_commit": git_commit(repo_root),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "soundfile_version": sf.__version__,
        "seed": args.seed,
        "sample_rate": args.sample_rate,
        "duration_sec": args.duration_sec,
        "class_angles_deg": list(CLASS_ANGLES_DEG),
        "project_azimuth_convention": "positive toward listener right",
        "roomsim_mapping": "rir_azimuth_deg = -project_azimuth_deg",
        "snr_db": list(SNR_DB),
        "test_conditions_rt60_ms_snr_db": [list(value) for value in TEST_CONDITIONS],
        "train_val_r0_fraction": 0.10,
        "demand_scenes": list(DEMAND_SCENES),
        "demand_time_partitions": SPLIT_TIME_RANGES,
        "noise_rendering": (
            "One DEMAND channel is treated as noise content and rendered through a second "
            "CIPIC Roomsim BRIR in the same subject/room/RT condition."
        ),
        "normalization": "one common scalar for both ears after SNR mixing",
        "input_roots": {
            "rir": str(args.rir_root),
            "demand": str(args.demand_root),
            "train_speech": str(args.train_speech_root),
            "val_speech": str(args.val_speech_root),
            "test_speech": str(args.test_speech_root),
        },
        "rooms": [asdict(room) for room in ROOM_SPECS],
        "task_counts": {split: len(tasks) for split, tasks in tasks_by_split.items()},
        "clip_counts": {split: len(tasks) * len(CLASS_ANGLES_DEG) for split, tasks in tasks_by_split.items()},
        "inventory": dict(inventory),
        "split_reports": dict(split_reports),
        "quality_report": "quality_report.json",
        "quality_passed": bool(quality["passed"]),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rir_root", type=Path, default=Path("/disk2/bywang/data/RIR-CIPIC"))
    parser.add_argument("--demand_root", type=Path, default=Path("/disk2/bywang/data/demand"))
    parser.add_argument("--train_speech_root", type=Path, default=Path("/disk2/bywang/data/LibriSpeech/train-clean-100"))
    parser.add_argument("--val_speech_root", type=Path, default=Path("/disk2/bywang/data/LibriSpeech/LibriSpeech_dev/dev-clean"))
    parser.add_argument("--test_speech_root", type=Path, default=Path("/disk2/bywang/data/LibriSpeech/LibriSpeech_test/test-clean"))
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration_sec", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hash_brir", action="store_true")
    parser.add_argument("--inventory_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"Output already exists; use --resume only for the same protocol: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    speech_paths = validate_input_layout(args)
    tasks_all = make_tasks(args.mode, args.seed)
    tasks_by_split = {split: tasks_all[split] for split in args.splits}
    expected_clips = {split: len(tasks) * len(CLASS_ANGLES_DEG) for split, tasks in tasks_by_split.items()}
    print(f"mode={args.mode} expected_clips={expected_clips}", flush=True)

    inventory_path = output_root / "brir_inventory.csv"
    if inventory_path.is_file() and args.resume:
        inventory = {
            "required_paths": sum(1 for _ in inventory_path.open("r", encoding="utf-8")) - 1,
            "missing_paths": 0,
            "sha256_included": args.hash_brir,
            "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        }
    else:
        inventory = inventory_required_brirs(args.rir_root, inventory_path, args.hash_brir)
    print(f"inventory={inventory}", flush=True)
    if args.inventory_only:
        return

    worker_config: Dict[str, object] = {
        "sample_rate": args.sample_rate,
        "output_samples": int(round(args.sample_rate * args.duration_sec)),
        "rir_root": str(args.rir_root.resolve()),
        "demand_root": str(args.demand_root.resolve()),
        "output_root": str(output_root),
    }
    for split, paths in speech_paths.items():
        worker_config[f"{split}_speech_paths"] = paths

    split_reports = {}
    for split in args.splits:
        split_reports[split] = generate_split(
            split, tasks_by_split[split], worker_config, max(1, args.workers)
        )
    inspect_limit = 100000 if args.mode == "smoke" else 1000
    quality = quality_check(
        output_root,
        tasks_by_split,
        args.sample_rate,
        int(round(args.sample_rate * args.duration_sec)),
        args.rir_root,
        inspect_limit,
    )
    (output_root / "quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    write_manifest(args, output_root, tasks_by_split, inventory, split_reports, quality)
    print(json.dumps({"output_root": str(output_root), "quality_passed": quality["passed"], "clips": expected_clips}, indent=2), flush=True)


if __name__ == "__main__":
    main()
