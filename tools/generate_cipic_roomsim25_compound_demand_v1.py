#!/usr/bin/env python3
"""Generate a paired CIPIC25 compound-interference stress test.

Each base realization contains reverberant target speech, one directional
non-speech DNS3 interferer, and a binaural diffuse background whose temporal
texture comes from a speech-audited DEMAND scene. Six gain conditions share
the exact same source content and spatial geometry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import scipy
import soundfile as sf
from netCDF4 import Dataset
from scipy.signal import csd, resample_poly, welch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_dns3_noise_inventory import (
    MAX_CLIPPED_FRACTION,
    MIN_ACTIVE_RATIO,
    MAX_INACTIVE_SECONDS,
    frame_activity,
    max_false_run,
    resample_mono,
)
from tools.generate_cipic_roomsim25 import (
    CLASS_ANGLES_DEG,
    ROOM_BY_SUBJECT,
    ROOM_SPECS,
    SPLIT_SUBJECTS,
    _load_brir_cached,
    balanced_schedule,
    brir_path,
    choose_speech_context,
    git_commit,
    handedness_check,
    joint_normalize,
    load_completed_tasks,
    list_audio_files,
    project_to_rir_angle,
    render_context,
    speaker_id,
    stable_int,
)
from tools.generate_cipic_roomsim25_directional_dns_v4 import (
    active_sample_mask,
    load_noise_context,
    load_noise_inventory,
    noise_angle_schedule,
    ordered_noise_records,
)


DATASET_NAME = "librispeech_cipic_roomsim25_compound_demand_v1"
DEMAND_SCENES: Tuple[str, ...] = (
    "DKITCHEN",
    "DWASHING",
    "NFIELD",
    "NRIVER",
    "STRAFFIC",
    "TCAR",
)
CONDITIONS: Tuple[Tuple[str, float, float | None], ...] = (
    ("R_dir0_nodiff", 0.0, None),
    ("A_sir5_diff5", 5.0, 5.0),
    ("B_sir0_diff5", 0.0, 5.0),
    ("C_sirm5_diff5", -5.0, 5.0),
    ("D_sir0_diff10", 0.0, 10.0),
    ("E_sir0_diff0", 0.0, 0.0),
)
RT60_MS = 600
DEMAND_CHANNEL = 1
DEMAND_CONTEXT_SECONDS = 2.5
DEMAND_ENVELOPE_SECONDS = 0.100
HRTF_DIRECTION_COUNT = 24
MAX_BRIR_PREFIX_SECONDS = 0.95


_WORKER_CONFIG: Dict[str, object] = {}


def init_worker(config: Mapping[str, object]) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = dict(config)
    _load_brir_cached.cache_clear()
    _load_demand_cached.cache_clear()
    _load_hrtf_coherence.cache_clear()


def selected_distance_indices(subject: str) -> Tuple[int, int]:
    distances = ROOM_BY_SUBJECT[subject].distances_m
    if len(distances) < 2:
        raise RuntimeError(f"Test room for subject {subject} has fewer than two distances")
    return 0, len(distances) - 1


def _overlaps(start: int, end: int, intervals: Sequence[Tuple[int, int]]) -> bool:
    return any(start < other_end and end > other_start for other_start, other_end in intervals)


def audit_demand_scenes(
    demand_root: Path,
    sample_rate: int,
    context_samples: int,
) -> Dict[str, object]:
    """Build a deterministic non-overlapping, no-speech segment inventory."""
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    torch.set_num_threads(1)
    model = load_silero_vad(onnx=True)
    scenes: Dict[str, object] = {}
    for scene in DEMAND_SCENES:
        path = demand_root / scene / f"ch{DEMAND_CHANNEL:02d}.wav"
        if not path.is_file():
            raise FileNotFoundError(path)
        audio, source_sr = sf.read(path, dtype="float32", always_2d=True)
        mono = resample_mono(audio, int(source_sr))
        if sample_rate != 16000:
            divisor = math.gcd(16000, int(sample_rate))
            mono = resample_poly(mono, sample_rate // divisor, 16000 // divisor).astype(np.float32)
        timestamps = get_speech_timestamps(
            torch.from_numpy(mono),
            model,
            sampling_rate=sample_rate,
            threshold=0.5,
            min_speech_duration_ms=100,
            min_silence_duration_ms=100,
            return_seconds=False,
        )
        speech_intervals = [(int(row["start"]), int(row["end"])) for row in timestamps]
        starts: List[int] = []
        rejected = Counter()
        max_inactive_frames = int(round(MAX_INACTIVE_SECONDS / 0.010))
        for start in range(0, len(mono) - context_samples + 1, context_samples):
            end = start + context_samples
            segment = mono[start:end]
            active, _threshold = frame_activity(segment)
            active_ratio = float(np.mean(active)) if active.size else 0.0
            clipped_fraction = float(np.mean(np.abs(segment) >= 0.999))
            rms = math.sqrt(float(np.mean(np.square(segment, dtype=np.float64))) + 1e-20)
            reasons = []
            if _overlaps(start, end, speech_intervals):
                reasons.append("silero_speech")
            if active_ratio < MIN_ACTIVE_RATIO:
                reasons.append("low_activity")
            if active.size and max_false_run(active) > max_inactive_frames:
                reasons.append("long_inactive_gap")
            if clipped_fraction > MAX_CLIPPED_FRACTION:
                reasons.append("clipped")
            if rms < 1e-5:
                reasons.append("near_silent")
            if reasons:
                rejected.update(reasons)
            else:
                starts.append(start)
        if not starts:
            raise RuntimeError(f"No eligible DEMAND segments for {scene}")
        scenes[scene] = {
            "path": str(path.resolve()),
            "source_sample_rate": int(source_sr),
            "duration_sec": round(len(mono) / sample_rate, 6),
            "speech_segments": len(speech_intervals),
            "eligible_starts_samples": starts,
            "eligible_segments": len(starts),
            "rejected": dict(rejected),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {
        "method": "non-overlapping 2.5 s windows; Silero VAD plus activity/clipping audit",
        "sample_rate": sample_rate,
        "channel_policy": "ch01 only; synchronized DEMAND array channels are not treated as independent",
        "manual_listening_audit": False,
        "scenes": scenes,
    }


def make_test_bundles(
    mode: str,
    seed: int,
    noise_records: Sequence[Mapping[str, object]],
    demand_inventory: Mapping[str, object],
) -> List[Dict[str, object]]:
    records = ordered_noise_records(noise_records, "test", seed)
    rooms = [room for room in ROOM_SPECS if room.split == "test"]
    if mode == "smoke":
        rooms = rooms[:1]
    scenes = DEMAND_SCENES if mode == "full" else DEMAND_SCENES[:2]
    angles = CLASS_ANGLES_DEG if mode == "full" else (-80, 0, 80)
    bundles: List[Dict[str, object]] = []
    noise_cursor = 0
    for room in rooms:
        subjects = room.subjects if mode == "full" else room.subjects[:1]
        for subject in subjects:
            distance_indices = selected_distance_indices(subject) if mode == "full" else (0,)
            for target_distance_index in distance_indices:
                for target_angle in angles:
                    group_seed = stable_int(seed, subject, target_distance_index, target_angle)
                    noise_angles = noise_angle_schedule(int(target_angle), len(scenes), group_seed + 1)
                    noise_distances = balanced_schedule(
                        tuple(range(len(room.distances_m))), len(scenes), group_seed + 2
                    )
                    for scene_index, scene in enumerate(scenes):
                        record = records[noise_cursor % len(records)]
                        noise_cursor += 1
                        starts = list(demand_inventory["scenes"][scene]["eligible_starts_samples"])
                        row_seed = stable_int(group_seed, scene)
                        demand_start = int(starts[stable_int(row_seed, "demand_start") % len(starts)])
                        dns_starts = list(record["active_starts_sec"])
                        dns_start = float(dns_starts[stable_int(row_seed, "dns_start") % len(dns_starts)])
                        task_id = (
                            f"test_s{subject}_d{target_distance_index:02d}_"
                            f"a{CLASS_ANGLES_DEG.index(target_angle):02d}_n{scene_index:02d}"
                        )
                        bundles.append({
                            "task_id": task_id,
                            "split": "test",
                            "subject_id": subject,
                            "target_angle_deg": int(target_angle),
                            "target_distance_index": int(target_distance_index),
                            "noise_angle_deg": int(noise_angles[scene_index]),
                            "noise_distance_index": int(noise_distances[scene_index]),
                            "speech_index": int(stable_int(row_seed, "speech_index")),
                            "noise_path": str(record["path"]),
                            "noise_source_id": str(record["source_id"]),
                            "noise_source_kind": str(record["source_kind"]),
                            "noise_target_start_sec": dns_start,
                            "demand_scene": scene,
                            "demand_path": str(demand_inventory["scenes"][scene]["path"]),
                            "demand_start_sample": demand_start,
                            "seed": int(row_seed),
                            "conditions": CONDITIONS,
                        })
    return bundles


def make_bundles(
    mode: str,
    seed: int,
    noise_by_split: Mapping[str, Sequence[Mapping[str, object]]],
    demand_inventory: Mapping[str, object],
) -> Dict[str, List[Dict[str, object]]]:
    return {"test": make_test_bundles(mode, seed, noise_by_split["test"], demand_inventory)}


@lru_cache(maxsize=8)
def _load_demand_cached(path_string: str, sample_rate: int) -> np.ndarray:
    audio, source_sr = sf.read(path_string, dtype="float32", always_2d=True)
    mono = np.asarray(audio[:, 0], dtype=np.float32)
    if int(source_sr) != sample_rate:
        divisor = math.gcd(int(source_sr), sample_rate)
        mono = resample_poly(mono, sample_rate // divisor, int(source_sr) // divisor).astype(np.float32)
    mono -= float(np.mean(mono))
    return mono


def _unit_vectors(positions: np.ndarray) -> np.ndarray:
    azimuth = np.deg2rad(positions[:, 0])
    elevation = np.deg2rad(positions[:, 1])
    return np.stack(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ],
        axis=1,
    )


def farthest_direction_indices(positions: np.ndarray, count: int) -> np.ndarray:
    vectors = _unit_vectors(np.asarray(positions, dtype=np.float64))
    if count > len(vectors):
        raise ValueError(f"Requested {count} directions from only {len(vectors)} positions")
    front = np.asarray([1.0, 0.0, 0.0])
    selected = [int(np.argmax(vectors @ front))]
    available = np.ones(len(vectors), dtype=bool)
    available[selected[0]] = False
    while len(selected) < count:
        similarity = np.max(vectors @ vectors[selected].T, axis=1)
        similarity[~available] = np.inf
        index = int(np.argmin(similarity))
        selected.append(index)
        available[index] = False
    return np.asarray(selected, dtype=np.int64)


@lru_cache(maxsize=16)
def _load_hrtf_coherence(
    subject: str,
    hrtf_root_string: str,
    sample_rate: int,
    fft_length: int,
    direction_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(hrtf_root_string) / f"subject_{subject}.sofa"
    with Dataset(str(path), "r") as database:
        positions = np.asarray(database.variables["SourcePosition"][:], dtype=np.float64)
        indices = farthest_direction_indices(positions, direction_count)
        ir = np.asarray(database.variables["Data.IR"][indices], dtype=np.float32)
        source_sr = int(round(float(np.asarray(database.variables["Data.SamplingRate"][:]).reshape(-1)[0])))
    if source_sr != sample_rate:
        divisor = math.gcd(source_sr, sample_rate)
        ir = resample_poly(ir, sample_rate // divisor, source_sr // divisor, axis=-1).astype(np.float32)
    transfer = np.fft.rfft(ir, n=fft_length, axis=-1)
    power_left = np.mean(np.square(np.abs(transfer[:, 0])), axis=0)
    power_right = np.mean(np.square(np.abs(transfer[:, 1])), axis=0)
    cross = np.mean(transfer[:, 0] * np.conj(transfer[:, 1]), axis=0)
    gamma = cross / np.sqrt(power_left * power_right + 1e-20)
    magnitude = np.abs(gamma)
    gamma *= np.minimum(1.0, 0.999999 / np.maximum(magnitude, 1e-20))
    gamma[0] = complex(float(np.real(gamma[0])), 0.0)
    if fft_length % 2 == 0:
        gamma[-1] = complex(float(np.real(gamma[-1])), 0.0)
    return gamma.astype(np.complex64), indices


def synthesize_diffuse_noise(row: Mapping[str, object], output_samples: int, sample_rate: int) -> Tuple[np.ndarray, Dict[str, object]]:
    context_samples = int(round(DEMAND_CONTEXT_SECONDS * sample_rate))
    reference = _load_demand_cached(str(row["demand_path"]), sample_rate)
    start = int(row["demand_start_sample"])
    segment = np.asarray(reference[start : start + context_samples], dtype=np.float64)
    if len(segment) != context_samples:
        raise RuntimeError(f"DEMAND segment out of range: {row['demand_path']} start={start}")
    segment -= float(np.mean(segment))
    envelope_samples = max(1, int(round(DEMAND_ENVELOPE_SECONDS * sample_rate)))
    envelope_kernel = np.ones(envelope_samples, dtype=np.float64) / envelope_samples
    envelope = np.sqrt(np.maximum(np.convolve(np.square(segment), envelope_kernel, mode="same"), 1e-10))
    envelope /= math.sqrt(float(np.mean(np.square(envelope))) + 1e-20)

    magnitude = np.abs(np.fft.rfft(segment))
    rng = np.random.default_rng(stable_int(row["seed"], "diffuse_carriers"))
    carriers = []
    for _channel in range(2):
        phase = rng.uniform(-math.pi, math.pi, size=magnitude.shape)
        phase[0] = 0.0
        if context_samples % 2 == 0:
            phase[-1] = 0.0
        carrier = np.fft.irfft(magnitude * np.exp(1j * phase), n=context_samples)
        carrier *= envelope
        carrier -= float(np.mean(carrier))
        carrier /= math.sqrt(float(np.mean(np.square(carrier))) + 1e-20)
        carriers.append(carrier)

    spectra = np.fft.rfft(np.stack(carriers, axis=0), axis=1)
    gamma, direction_indices = _load_hrtf_coherence(
        str(row["subject_id"]),
        str(_WORKER_CONFIG["hrtf_root"]),
        sample_rate,
        context_samples,
        HRTF_DIRECTION_COUNT,
    )
    left = spectra[0]
    right = np.conj(gamma) * spectra[0] + np.sqrt(np.maximum(0.0, 1.0 - np.square(np.abs(gamma)))) * spectra[1]
    diffuse = np.stack(
        [np.fft.irfft(left, n=context_samples), np.fft.irfft(right, n=context_samples)], axis=1
    )
    crop_start = (context_samples - output_samples) // 2
    diffuse = np.asarray(diffuse[crop_start : crop_start + output_samples], dtype=np.float32)
    diffuse -= np.mean(diffuse, axis=0, keepdims=True)
    diffuse /= math.sqrt(float(np.mean(np.square(diffuse, dtype=np.float64))) + 1e-20)
    frequencies = np.fft.rfftfreq(context_samples, 1.0 / sample_rate)
    low = np.abs(gamma[(frequencies >= 200.0) & (frequencies <= 1500.0)])
    high = np.abs(gamma[(frequencies >= 4000.0) & (frequencies <= 7500.0)])
    return diffuse, {
        "diffuse_field_model": "CIPIC-HRTF-derived coherence-constrained synthesis",
        "diffuse_hrtf_direction_count": HRTF_DIRECTION_COUNT,
        "diffuse_hrtf_direction_indices": ";".join(str(int(value)) for value in direction_indices),
        "diffuse_target_coherence_low_mean": round(float(np.mean(low)), 8),
        "diffuse_target_coherence_high_mean": round(float(np.mean(high)), 8),
    }


def mix_compound(
    target: np.ndarray,
    directional: np.ndarray,
    diffuse: np.ndarray,
    sir_db: float,
    diffuse_snr_db: float | None,
    sample_rate: int,
) -> Tuple[np.ndarray, float, float | None, float]:
    mask = active_sample_mask(target, sample_rate)
    target_power = float(np.mean(np.square(target[mask], dtype=np.float64)))
    directional = directional - np.mean(directional, axis=0, keepdims=True)
    directional_power = float(np.mean(np.square(directional[mask], dtype=np.float64)))
    diffuse = diffuse - np.mean(diffuse, axis=0, keepdims=True)
    diffuse_power = float(np.mean(np.square(diffuse[mask], dtype=np.float64)))
    if min(target_power, directional_power, diffuse_power) <= 1e-12:
        raise RuntimeError("Cannot mix a silent target, directional interferer, or diffuse field")
    directional_gain = math.sqrt(target_power / (directional_power * 10.0 ** (sir_db / 10.0)))
    scaled_directional = directional * directional_gain
    achieved_sir = 10.0 * math.log10(
        target_power / float(np.mean(np.square(scaled_directional[mask], dtype=np.float64)))
    )
    mixed = target + scaled_directional
    achieved_snr = None
    if diffuse_snr_db is not None:
        diffuse_gain = math.sqrt(target_power / (diffuse_power * 10.0 ** (diffuse_snr_db / 10.0)))
        scaled_diffuse = diffuse * diffuse_gain
        achieved_snr = 10.0 * math.log10(
            target_power / float(np.mean(np.square(scaled_diffuse[mask], dtype=np.float64)))
        )
        mixed = mixed + scaled_diffuse
    return mixed.astype(np.float32), float(achieved_sir), achieved_snr, float(np.mean(mask))


def render_bundle(bundle: Mapping[str, object]) -> Dict[str, object]:
    config = _WORKER_CONFIG
    sample_rate = int(config["sample_rate"])
    output_samples = int(config["output_samples"])
    rir_root = Path(str(config["rir_root"]))
    output_root = Path(str(config["output_root"]))
    subject = str(bundle["subject_id"])
    room = ROOM_BY_SUBJECT[subject]
    target_path = brir_path(
        rir_root, subject, room, RT60_MS, int(bundle["target_distance_index"]), int(bundle["target_angle_deg"])
    )
    noise_path = brir_path(
        rir_root, subject, room, RT60_MS, int(bundle["noise_distance_index"]), int(bundle["noise_angle_deg"])
    )
    target_brir = _load_brir_cached(str(target_path), sample_rate)
    noise_brir = _load_brir_cached(str(noise_path), sample_rate)
    prefix = max(len(target_brir), len(noise_brir)) - 1
    if prefix > int(round(MAX_BRIR_PREFIX_SECONDS * sample_rate)):
        raise RuntimeError(f"BRIR prefix exceeds audited DNS context: {prefix}")
    speech_paths = list(config["test_speech_paths"])
    speech, speech_path, speech_start, speech_active = choose_speech_context(
        speech_paths,
        int(bundle["speech_index"]),
        prefix,
        output_samples,
        sample_rate,
        stable_int(bundle["seed"], "speech"),
    )
    dns = load_noise_context(bundle, prefix, output_samples, sample_rate)
    target = render_context(speech, target_brir, prefix, output_samples)
    directional = render_context(dns, noise_brir, prefix, output_samples)
    diffuse, diffuse_meta = synthesize_diffuse_noise(bundle, output_samples, sample_rate)
    rows = []
    for condition_index, (condition, sir_db, diffuse_snr_db) in enumerate(bundle["conditions"]):
        mixed, achieved_sir, achieved_snr, active_ratio = mix_compound(
            target, directional, diffuse, float(sir_db), diffuse_snr_db, sample_rate
        )
        mixed = joint_normalize(mixed)
        file_id = f"{bundle['task_id']}_c{condition_index:02d}"
        relative_wav = Path("binaural") / f"{file_id}.wav"
        sf.write(output_root / "test" / relative_wav, mixed, sample_rate, subtype="PCM_16")
        rows.append({
            "file_id": file_id,
            "wav_path": str(relative_wav),
            "split": "test",
            "condition": condition,
            "class_index": CLASS_ANGLES_DEG.index(int(bundle["target_angle_deg"])),
            "azimuth_deg": int(bundle["target_angle_deg"]),
            "rir_azimuth_deg": project_to_rir_angle(int(bundle["target_angle_deg"])),
            "subject_id": subject,
            "room_id": room.room_id,
            "room_x_m": room.dims_m[0],
            "room_y_m": room.dims_m[1],
            "room_z_m": room.dims_m[2],
            "rt60_s": RT60_MS / 1000.0,
            "distance_m": room.distances_m[int(bundle["target_distance_index"])],
            "target_sir_db": float(sir_db),
            "achieved_sir_db": round(achieved_sir, 8),
            "target_diffuse_snr_db": "" if diffuse_snr_db is None else float(diffuse_snr_db),
            "achieved_diffuse_snr_db": "" if achieved_snr is None else round(achieved_snr, 8),
            "diffuse_present": int(diffuse_snr_db is not None),
            "power_metric": "target-active-sample binaural power after spatial rendering",
            "target_active_sample_ratio": round(active_ratio, 6),
            "speech_path": speech_path,
            "speech_speaker_id": speaker_id(Path(speech_path)),
            "speech_target_start_sample": int(speech_start),
            "speech_active_ratio": round(float(speech_active), 6),
            "noise_content_path": str(bundle["noise_path"]),
            "noise_source_id": str(bundle["noise_source_id"]),
            "noise_source_kind": str(bundle["noise_source_kind"]),
            "noise_content_start_sec": float(bundle["noise_target_start_sec"]),
            "noise_azimuth_deg": int(bundle["noise_angle_deg"]),
            "noise_rir_azimuth_deg": project_to_rir_angle(int(bundle["noise_angle_deg"])),
            "noise_distance_m": room.distances_m[int(bundle["noise_distance_index"])],
            "angular_separation_deg": abs(int(bundle["target_angle_deg"]) - int(bundle["noise_angle_deg"])),
            "demand_scene": str(bundle["demand_scene"]),
            "demand_path": str(bundle["demand_path"]),
            "demand_channel": DEMAND_CHANNEL,
            "demand_start_sec": round(int(bundle["demand_start_sample"]) / sample_rate, 6),
            **diffuse_meta,
            "target_brir_path": str(target_path),
            "noise_brir_path": str(noise_path),
            "renderer": "DP-RTF Roomsim-CIPIC target and directional interferer",
            "paired_test_key": str(bundle["task_id"]),
            "seed": int(bundle["seed"]),
        })
    return {"task_id": str(bundle["task_id"]), "rows": rows}


def finalize_metadata(split_root: Path, expected_tasks: int, expected_clips: int) -> Dict[str, object]:
    completed = load_completed_tasks(split_root / "metadata.tasks.jsonl")
    if len(completed) != expected_tasks:
        raise RuntimeError(f"Incomplete test split: {len(completed)}/{expected_tasks}")
    rows = [row for task_id in sorted(completed) for row in completed[task_id]["rows"]]
    if len(rows) != expected_clips:
        raise RuntimeError(f"Unexpected clip count: {len(rows)}/{expected_clips}")
    if len({row["file_id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate file IDs")
    path = split_root / "metadata.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"num_tasks": expected_tasks, "num_clips": expected_clips, "metadata_path": str(path)}


def generate_test(bundles: Sequence[Mapping[str, object]], config: Mapping[str, object], workers: int) -> Dict[str, object]:
    split_root = Path(str(config["output_root"])) / "test"
    (split_root / "binaural").mkdir(parents=True, exist_ok=True)
    task_jsonl = split_root / "metadata.tasks.jsonl"
    completed = load_completed_tasks(task_jsonl)
    expected_ids = {str(bundle["task_id"]) for bundle in bundles}
    if set(completed) - expected_ids:
        raise RuntimeError("Output contains tasks from a different protocol")
    pending = [bundle for bundle in bundles if str(bundle["task_id"]) not in completed]
    expected_clips = len(bundles) * len(CONDITIONS)
    completed_clips = sum(len(payload["rows"]) for payload in completed.values())
    print(f"[test] tasks={len(completed)}/{len(bundles)} clips={completed_clips}/{expected_clips}", flush=True)
    started = time.time()
    if pending:
        with task_jsonl.open("a", encoding="utf-8") as output_handle:
            with ProcessPoolExecutor(
                max_workers=max(1, workers),
                initializer=init_worker,
                initargs=(config,),
                mp_context=mp.get_context("spawn"),
            ) as executor:
                for offset, payload in enumerate(executor.map(render_bundle, pending, chunksize=1), start=1):
                    output_handle.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n")
                    output_handle.flush()
                    os.fsync(output_handle.fileno())
                    completed_clips += len(payload["rows"])
                    done = len(completed) + offset
                    if offset == 1 or done % 25 == 0 or done == len(bundles):
                        elapsed = max(time.time() - started, 1e-6)
                        rate = offset / elapsed
                        eta = (len(pending) - offset) / rate if rate else float("inf")
                        progress = {
                            "completed_tasks": done,
                            "total_tasks": len(bundles),
                            "completed_clips": completed_clips,
                            "total_clips": expected_clips,
                            "elapsed_sec_this_run": elapsed,
                            "eta_sec": eta,
                        }
                        (split_root / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
                        print(
                            f"[test] tasks={done}/{len(bundles)} clips={completed_clips}/{expected_clips} "
                            f"rate={rate:.3f} task/s eta={eta/60:.1f} min",
                            flush=True,
                        )
    report = finalize_metadata(split_root, len(bundles), expected_clips)
    report["duration_sec_this_run"] = time.time() - started
    return report


def quality_check(
    output_root: Path,
    bundles: Sequence[Mapping[str, object]],
    sample_rate: int,
    output_samples: int,
    rir_root: Path,
    full: bool,
) -> Dict[str, object]:
    with (output_root / "test" / "metadata.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = len(bundles) * len(CONDITIONS)
    if len(rows) != expected:
        raise RuntimeError(f"Metadata row count {len(rows)} != {expected}")
    condition_names = {value[0] for value in CONDITIONS}
    condition_counts = Counter(row["condition"] for row in rows)
    if set(condition_counts) != condition_names or len(set(condition_counts.values())) != 1:
        raise RuntimeError(f"Unbalanced conditions: {condition_counts}")
    grouped: Dict[str, List[Mapping[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["paired_test_key"], []).append(row)
    if len(grouped) != len(bundles) or any({row["condition"] for row in values} != condition_names for values in grouped.values()):
        raise RuntimeError("Broken six-condition pairing")
    class_by_condition = Counter((row["condition"], int(row["class_index"])) for row in rows)
    scene_by_condition = Counter((row["condition"], row["demand_scene"]) for row in rows)
    if full:
        if set(class_by_condition.values()) != {108}:
            raise RuntimeError(f"Unexpected class-condition counts: {Counter(class_by_condition.values())}")
        if set(scene_by_condition.values()) != {450}:
            raise RuntimeError(f"Unexpected scene-condition counts: {Counter(scene_by_condition.values())}")
    sir_error = max(abs(float(row["target_sir_db"]) - float(row["achieved_sir_db"])) for row in rows)
    diffuse_rows = [row for row in rows if row["target_diffuse_snr_db"] != ""]
    diffuse_error = max(
        abs(float(row["target_diffuse_snr_db"]) - float(row["achieved_diffuse_snr_db"]))
        for row in diffuse_rows
    )
    if sir_error > 0.1 or diffuse_error > 0.1:
        raise RuntimeError(f"Gain error too large: SIR={sir_error}, diffuse SNR={diffuse_error}")
    inspect_limit = min(len(rows), 1000 if full else len(rows))
    step = max(1, len(rows) // inspect_limit)
    inspected = rows[::step][:inspect_limit]
    max_peak = 0.0
    min_rms = float("inf")
    for row in inspected:
        path = output_root / "test" / row["wav_path"]
        info = sf.info(path)
        if info.samplerate != sample_rate or info.channels != 2 or info.frames != output_samples:
            raise RuntimeError(f"Invalid WAV: {path}: {info}")
        audio, _ = sf.read(path, dtype="float32", always_2d=True)
        max_peak = max(max_peak, float(np.max(np.abs(audio))))
        min_rms = min(min_rms, math.sqrt(float(np.mean(np.square(audio, dtype=np.float64)))))
    report = {
        "passed": True,
        "handedness": handedness_check(rir_root),
        "num_base_realizations": len(grouped),
        "num_clips": len(rows),
        "condition_counts": dict(condition_counts),
        "class_condition_count_values": dict(Counter(class_by_condition.values())),
        "scene_condition_count_values": dict(Counter(scene_by_condition.values())),
        "subject_counts": dict(Counter(row["subject_id"] for row in rows)),
        "max_sir_error_db": sir_error,
        "max_diffuse_snr_error_db": diffuse_error,
        "wav_files_inspected": len(inspected),
        "max_abs_peak": max_peak,
        "min_rms": min_rms,
    }
    if not report["handedness"]["passed"]:
        raise RuntimeError("BRIR handedness check failed")
    return report


def diffuse_coherence_audit(
    bundles: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    full: bool,
) -> Dict[str, object]:
    """Validate ensemble coherence against one subject-specific HRTF target."""
    init_worker(config)
    subject = str(bundles[0]["subject_id"])
    candidates = [bundle for bundle in bundles if str(bundle["subject_id"]) == subject]
    audit_count = min(len(candidates), 32 if full else 6)
    estimates = []
    frequencies = None
    sample_rate = int(config["sample_rate"])
    output_samples = int(config["output_samples"])
    for bundle in candidates[:audit_count]:
        diffuse, _metadata = synthesize_diffuse_noise(bundle, output_samples, sample_rate)
        frequencies, power_left = welch(
            diffuse[:, 0], fs=sample_rate, nperseg=1024, noverlap=512
        )
        _frequencies, power_right = welch(
            diffuse[:, 1], fs=sample_rate, nperseg=1024, noverlap=512
        )
        # scipy.csd(x, y) returns conj(X)Y. x=R,y=L therefore estimates L*conj(R).
        _frequencies, cross_lr = csd(
            diffuse[:, 1], diffuse[:, 0], fs=sample_rate, nperseg=1024, noverlap=512
        )
        estimates.append(cross_lr / np.sqrt(power_left * power_right + 1e-20))
    estimated = np.mean(estimates, axis=0)
    context_samples = int(round(DEMAND_CONTEXT_SECONDS * sample_rate))
    target, _indices = _load_hrtf_coherence(
        subject,
        str(config["hrtf_root"]),
        sample_rate,
        context_samples,
        HRTF_DIRECTION_COUNT,
    )
    target_frequencies = np.fft.rfftfreq(context_samples, 1.0 / sample_rate)
    interpolated = np.interp(frequencies, target_frequencies, target.real) + 1j * np.interp(
        frequencies, target_frequencies, target.imag
    )
    mask = (frequencies >= 200.0) & (frequencies <= 7500.0)
    nmse = float(
        np.mean(np.abs(estimated[mask] - interpolated[mask]) ** 2)
        / (np.mean(np.abs(interpolated[mask]) ** 2) + 1e-20)
    )
    nmse_db = 10.0 * math.log10(nmse + 1e-20)
    threshold_db = -10.0 if full else -3.0
    if nmse_db > threshold_db:
        raise RuntimeError(
            f"Diffuse coherence audit failed: NMSE={nmse_db:.3f} dB > {threshold_db:.1f} dB"
        )
    return {
        "passed": True,
        "subject_id": subject,
        "num_realizations": audit_count,
        "frequency_range_hz": [200, 7500],
        "coherence_nmse_db": nmse_db,
        "threshold_db": threshold_db,
        "estimated_abs_mean": float(np.mean(np.abs(estimated[mask]))),
        "target_abs_mean": float(np.mean(np.abs(interpolated[mask]))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rir_root", type=Path, default=Path("/disk2/bywang/data/RIR-CIPIC"))
    parser.add_argument("--hrtf_root", type=Path, default=Path("/disk2/bywang/data/HRTF"))
    parser.add_argument("--demand_root", type=Path, default=Path("/disk2/bywang/data/demand"))
    parser.add_argument(
        "--test_speech_root", type=Path,
        default=Path("/disk2/bywang/data/LibriSpeech/LibriSpeech_test/test-clean"),
    )
    parser.add_argument(
        "--noise_inventory", type=Path,
        default=Path("data/dns3_directional_v4_inventory/dns3_noise_inventory.csv"),
    )
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration_sec", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--inventory_source", type=Path,
        default=Path("data/librispeech_cipic_roomsim25_directional_dns_v4/brir_inventory.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_rate != 16000:
        raise ValueError("This fixed protocol is defined only at 16 kHz")
    output_root = args.output_root.resolve()
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"Output exists; use --resume only for this exact protocol: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    speech_paths = [str(path) for path in list_audio_files(args.test_speech_root)]
    if not speech_paths:
        raise FileNotFoundError(f"No test speech under {args.test_speech_root}")
    noise_by_split = load_noise_inventory(args.noise_inventory.resolve())
    context_samples = int(round(DEMAND_CONTEXT_SECONDS * args.sample_rate))
    demand_inventory_path = output_root / "demand_segment_inventory.json"
    if demand_inventory_path.is_file() and args.resume:
        demand_inventory = json.loads(demand_inventory_path.read_text(encoding="utf-8"))
    else:
        demand_inventory = audit_demand_scenes(args.demand_root.resolve(), args.sample_rate, context_samples)
        demand_inventory_path.write_text(json.dumps(demand_inventory, indent=2), encoding="utf-8")
    bundles = make_test_bundles(args.mode, args.seed, noise_by_split["test"], demand_inventory)
    expected_clips = len(bundles) * len(CONDITIONS)
    expected = 16200 if args.mode == "full" else 36
    if expected_clips != expected:
        raise RuntimeError(f"Protocol count mismatch: {expected_clips} != {expected}")
    print(f"mode={args.mode} base_realizations={len(bundles)} expected_clips={expected_clips}", flush=True)

    brir_inventory_path = output_root / "brir_inventory.csv"
    if not brir_inventory_path.is_file():
        shutil.copy2(args.inventory_source, brir_inventory_path)
    brir_inventory = {
        "source": str(args.inventory_source.resolve()),
        "sha256": hashlib.sha256(brir_inventory_path.read_bytes()).hexdigest(),
    }
    config: Dict[str, object] = {
        "sample_rate": args.sample_rate,
        "output_samples": int(round(args.sample_rate * args.duration_sec)),
        "rir_root": str(args.rir_root.resolve()),
        "hrtf_root": str(args.hrtf_root.resolve()),
        "output_root": str(output_root),
        "test_speech_paths": speech_paths,
    }
    report = generate_test(bundles, config, args.workers)
    quality = quality_check(
        output_root,
        bundles,
        args.sample_rate,
        int(round(args.sample_rate * args.duration_sec)),
        args.rir_root,
        args.mode == "full",
    )
    quality["diffuse_coherence"] = diffuse_coherence_audit(
        bundles, config, args.mode == "full"
    )
    (output_root / "quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    noise_summary_path = args.noise_inventory.with_name("dns3_noise_inventory_summary.json")
    dns_inventory = json.loads(noise_summary_path.read_text(encoding="utf-8"))
    manifest = {
        "name": DATASET_NAME,
        "mode": args.mode,
        "created_unix_time": time.time(),
        "generator": str(Path(__file__).resolve()),
        "git_commit": git_commit(ROOT),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "soundfile_version": sf.__version__,
        "seed": args.seed,
        "sample_rate": args.sample_rate,
        "duration_sec": args.duration_sec,
        "class_angles_deg": list(CLASS_ANGLES_DEG),
        "rt60_ms": RT60_MS,
        "conditions": [
            {"name": name, "sir_db": sir, "diffuse_snr_db": diffuse_snr}
            for name, sir, diffuse_snr in CONDITIONS
        ],
        "demand_scenes": list(DEMAND_SCENES),
        "demand_inventory": "demand_segment_inventory.json",
        "diffuse_generation": {
            "method": "frequency-domain coherence-constrained synthesis",
            "coherence_target": "subject-specific CIPIC HRTF average over 24 farthest-point directions",
            "texture": "DEMAND ch01 phase-randomized spectrum plus shared 100 ms envelope",
            "reference": "Habets, Cohen, and Gannot, JASA 2008, DOI 10.1121/1.2987429",
        },
        "selected_distance_policy": "nearest and farthest distance for every test subject",
        "rooms": [asdict(room) for room in ROOM_SPECS if room.split == "test"],
        "base_realizations": len(bundles),
        "clip_count": expected_clips,
        "normalization": "one common scalar for both ears after target + directional + diffuse mixing",
        "sir_snr_definition": "binaural component power on target-active samples after spatial rendering",
        "dns_inventory": {
            "path": str(args.noise_inventory.resolve()),
            "inventory_sha256": dns_inventory.get("inventory_sha256"),
            "human_sound_exclusion": dns_inventory.get("rejection_reason_counts", {}),
        },
        "brir_inventory": brir_inventory,
        "generation_report": report,
        "quality_report": "quality_report.json",
        "quality_passed": bool(quality["passed"]),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "clips": expected_clips, "quality_passed": True}, indent=2))


if __name__ == "__main__":
    main()
