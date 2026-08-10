#!/usr/bin/env python3
"""Generate the CIPIC Roomsim25 nonstationary Habets-ANF v3 dataset.

Targets use the released DP-RTF Roomsim-CIPIC BRIRs. Additive noise is a
two-channel spherical diffuse field generated with Habets' ANF implementation.
The independent ANF excitations use a DEMAND-derived long-term spectrum and a
shared nonstationary DEMAND energy envelope. Train/validation examples are
sampled independently for every subject-angle; test examples are paired across
the complete RT60 x SNR grid.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import scipy
import soundfile as sf
from anf_generator import estimate_coherence, generate_signals
from anf_generator.CoherenceMatrix import Parameters as ANFParameters
from scipy.io import loadmat
from scipy.signal import fftconvolve, firwin2, welch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_cipic_roomsim25 import (
    CLASS_ANGLES_DEG,
    DEMAND_CHANNELS,
    ROOM_BY_SUBJECT,
    ROOM_SPECS,
    SPLIT_DIR_NAMES,
    SPLIT_SUBJECTS,
    balanced_schedule,
    brir_path,
    check_disjoint,
    choose_speech_context,
    demand_target_start,
    git_commit,
    handedness_check,
    inventory_required_brirs,
    joint_normalize,
    list_audio_files,
    load_completed_tasks,
    mix_at_snr,
    project_to_rir_angle,
    render_context,
    speaker_id,
    stable_int,
    _load_brir_cached,
    _load_noise_cached,
)


SNR_DB: Tuple[int, ...] = (-5, 0, 5, 10, 15)
# Keep the noise-content distribution fixed across splits so the formal test
# isolates unseen subjects and rooms. Waveform intervals remain disjoint.
TRAIN_NOISE_SCENES: Tuple[str, ...] = ("OOFFICE", "PCAFETER", "TMETRO")
TEST_NOISE_SCENES: Tuple[str, ...] = TRAIN_NOISE_SCENES
TEST_RT60_MS: Tuple[int, ...] = (200, 400, 600, 800)
SPLIT_TIME_RANGES = {"train": (0.0, 0.6), "val": (0.6, 0.8), "test": (0.8, 1.0)}
TRAIN_REALIZATIONS_PER_SNR_SCENE = 9
VAL_REALIZATIONS_PER_SNR_SCENE = 4
ANTHRO_FALLBACK_M = 0.145
ANF_NFFT = 512
ANF_SHAPING_FIR_TAPS = 513
ANF_ENVELOPE_SAMPLES = 1601  # 100 ms at 16 kHz.


_WORKER_CONFIG: Dict[str, object] = {}


def init_worker(config: Mapping[str, object]) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = dict(config)
    _load_brir_cached.cache_clear()
    _load_noise_cached.cache_clear()


def load_subject_spacings(path: Path, fallback_m: float = ANTHRO_FALLBACK_M) -> Dict[str, Dict[str, object]]:
    payload = loadmat(path)
    subject_ids = np.asarray(payload["id"]).reshape(-1).astype(int)
    measurements = np.asarray(payload["X"], dtype=np.float64)
    raw = {int(subject_id): float(measurements[index, 0]) for index, subject_id in enumerate(subject_ids)}
    result: Dict[str, Dict[str, object]] = {}
    for subject in sorted({value for values in SPLIT_SUBJECTS.values() for value in values}):
        value_cm = raw.get(int(subject), float("nan"))
        if np.isfinite(value_cm) and 10.0 <= value_cm <= 20.0:
            result[subject] = {"spacing_m": value_cm / 100.0, "source": "CIPIC_X1_head_width_proxy"}
        else:
            result[subject] = {"spacing_m": float(fallback_m), "source": "fallback_0.145m"}
    return result


def exact_rt_schedule(room, count: int, zero_count: int, seed: int) -> List[int]:
    nonzero = tuple(value for value in room.rt60_ms if value != 0)
    values = [0] * int(zero_count)
    values.extend(int(v) for v in balanced_schedule(nonzero, count - zero_count, seed + 1))
    random.Random(seed + 2).shuffle(values)
    return values


def recipe(split: str, subject: str, angle: int, rt60_ms: int, distance_index: int,
           snr_db: int, scene: str, seed: int, speech_index: int | None = None) -> Dict[str, object]:
    channel_a = DEMAND_CHANNELS[stable_int(seed, "noise_channel_a") % len(DEMAND_CHANNELS)]
    return {
        "split": split,
        "subject_id": subject,
        "angle_deg": int(angle),
        "rt60_ms": int(rt60_ms),
        "distance_index": int(distance_index),
        "snr_db": int(snr_db),
        "noise_scene": scene,
        "noise_channel_a": int(channel_a),
        # The same physical DEMAND channel gives the two excerpts comparable
        # long-term spectra; temporal non-overlap supplies independent content.
        "noise_channel_b": int(channel_a),
        "noise_u_a": (stable_int(seed, "noise_u_a") % 1_000_001) / 1_000_000.0,
        "noise_u_b": (stable_int(seed, "noise_u_b") % 1_000_001) / 1_000_000.0,
        "speech_index": int(stable_int(seed, "speech_index") if speech_index is None else speech_index),
        "seed": int(seed),
    }


def make_train_val_bundles(split: str, seed: int, smoke: bool = False) -> List[Dict[str, object]]:
    scenes = TRAIN_NOISE_SCENES
    repeats = TRAIN_REALIZATIONS_PER_SNR_SCENE if split == "train" else VAL_REALIZATIONS_PER_SNR_SCENE
    subjects = SPLIT_SUBJECTS[split][:1] if smoke else SPLIT_SUBJECTS[split]
    angles = CLASS_ANGLES_DEG[:1] if smoke else CLASS_ANGLES_DEG
    bundles: List[Dict[str, object]] = []
    total_groups = len(subjects) * len(angles)
    group_index = 0
    for subject in subjects:
        room = ROOM_BY_SUBJECT[subject]
        for angle in angles:
            if smoke:
                combinations = [(SNR_DB[0], scenes[0], 0)]
                zero_count = 0
            else:
                combinations = [(snr, scene, rep) for snr in SNR_DB for scene in scenes for rep in range(repeats)]
                if split == "train":
                    # 750 groups alternate 13/14 R0 samples: exactly 10,125 R0 clips.
                    zero_count = 14 if group_index < total_groups // 2 else 13
                else:
                    # Validation must match the reverberant-only formal test.
                    zero_count = 0
            group_seed = stable_int(seed, split, subject, angle)
            random.Random(group_seed).shuffle(combinations)
            rt_values = exact_rt_schedule(room, len(combinations), zero_count, group_seed + 10)
            distance_values = balanced_schedule(tuple(range(len(room.distances_m))), len(combinations), group_seed + 20)
            rows = []
            for row_index, ((snr, scene, rep), rt60_ms, distance_index) in enumerate(
                zip(combinations, rt_values, distance_values)
            ):
                row_seed = stable_int(group_seed, row_index, snr, scene, rep)
                rows.append(recipe(split, subject, angle, rt60_ms, distance_index, snr, scene, row_seed))
            bundles.append({
                "task_id": f"{split}_s{subject}_a{CLASS_ANGLES_DEG.index(angle):02d}",
                "split": split,
                "paired_test": False,
                "recipes": rows,
            })
            group_index += 1
    return bundles


def make_test_bundles(seed: int, smoke: bool = False) -> List[Dict[str, object]]:
    bundles: List[Dict[str, object]] = []
    test_rooms = [room for room in ROOM_SPECS if room.split == "test"]
    if smoke:
        test_rooms = test_rooms[:1]
    for room in test_rooms:
        subjects = room.subjects[:1] if smoke else room.subjects
        distances = range(1 if smoke else len(room.distances_m))
        angles = CLASS_ANGLES_DEG[:1] if smoke else CLASS_ANGLES_DEG
        scenes = TEST_NOISE_SCENES[:1] if smoke else TEST_NOISE_SCENES
        for subject in subjects:
            for distance_index in distances:
                for angle in angles:
                    for scene in scenes:
                        base_seed = stable_int(seed, "test", subject, distance_index, angle, scene)
                        rts = TEST_RT60_MS[:1] if smoke else TEST_RT60_MS
                        snrs = SNR_DB[:1] if smoke else SNR_DB
                        shared_speech_index = stable_int(base_seed, "paired_speech")
                        rows = [
                            recipe(
                                "test", subject, angle, rt60_ms, distance_index, snr, scene,
                                stable_int(base_seed, rt60_ms, snr), shared_speech_index,
                            )
                            for rt60_ms in rts for snr in snrs
                        ]
                        # ANF excitation and speech are generated once and shared by all RT/SNR rows.
                        for row in rows:
                            row["noise_channel_a"] = rows[0]["noise_channel_a"]
                            row["noise_channel_b"] = rows[0]["noise_channel_b"]
                            row["noise_u_a"] = rows[0]["noise_u_a"]
                            row["noise_u_b"] = rows[0]["noise_u_b"]
                        bundles.append({
                            "task_id": (
                                f"test_s{subject}_d{distance_index:02d}_a{CLASS_ANGLES_DEG.index(angle):02d}_"
                                f"n{TEST_NOISE_SCENES.index(scene):02d}"
                            ),
                            "split": "test",
                            "paired_test": True,
                            "recipes": rows,
                        })
    return bundles


def make_bundles(mode: str, seed: int) -> Dict[str, List[Dict[str, object]]]:
    smoke = mode == "smoke"
    return {
        "train": make_train_val_bundles("train", seed, smoke),
        "val": make_train_val_bundles("val", seed, smoke),
        "test": make_test_bundles(seed, smoke),
    }


def validate_input_layout(args: argparse.Namespace) -> Dict[str, List[str]]:
    roots = {"train": args.train_speech_root, "val": args.val_speech_root, "test": args.test_speech_root}
    paths = {split: [str(path) for path in list_audio_files(root)] for split, root in roots.items()}
    check_disjoint(({speaker_id(Path(path)) for path in values} for values in paths.values()), "LibriSpeech speaker")
    check_disjoint(SPLIT_SUBJECTS.values(), "CIPIC subject")
    for scene in dict.fromkeys(TRAIN_NOISE_SCENES + TEST_NOISE_SCENES):
        for channel in DEMAND_CHANNELS:
            path = args.demand_root / scene / f"ch{channel:02d}.wav"
            if not path.is_file():
                raise FileNotFoundError(path)
    return paths


def _noise_start(path: Path, split: str, value: float, sample_rate: int, length: int) -> int:
    # Reuse the audited time-partition implementation, with no convolution prefix.
    return demand_target_start(path, split, value, sample_rate, length, 0)


def generate_diffuse_noise(row: Mapping[str, object], length: int, spacing_m: float) -> Tuple[np.ndarray, Dict[str, object]]:
    config = _WORKER_CONFIG
    sample_rate = int(config["sample_rate"])
    demand_root = Path(str(config["demand_root"]))
    context = ANF_NFFT
    source_length = length + 2 * context
    scene = str(row["noise_scene"])
    path_a = demand_root / scene / f"ch{int(row['noise_channel_a']):02d}.wav"
    path_b = demand_root / scene / f"ch{int(row['noise_channel_b']):02d}.wav"
    start_a = _noise_start(path_a, str(row["split"]), float(row["noise_u_a"]), sample_rate, source_length)
    source_a = _load_noise_cached(str(path_a), sample_rate)[start_a:start_a + source_length]
    if len(source_a) != source_length:
        raise RuntimeError("Unexpected DEMAND segment length")

    # Use two non-overlapping excerpts to estimate the scene spectrum and
    # temporal envelope without leaking waveform intervals across data splits.
    base_u = float(row["noise_u_b"])
    for attempt in range(16):
        value = (base_u + attempt * 0.3819660112501051) % 1.0
        start_b = _noise_start(path_b, str(row["split"]), value, sample_rate, source_length)
        if abs(start_b - start_a) < source_length:
            continue
        source_b = _load_noise_cached(str(path_b), sample_rate)[start_b:start_b + source_length]
        if len(source_b) == source_length:
            break
    else:
        raise RuntimeError(f"Unable to obtain non-overlapping DEMAND references: {path_a}")

    source_a = np.asarray(source_a - np.mean(source_a), dtype=np.float64)
    source_b = np.asarray(source_b - np.mean(source_b), dtype=np.float64)
    source_a /= math.sqrt(float(np.mean(np.square(source_a))) + 1e-12)
    source_b /= math.sqrt(float(np.mean(np.square(source_b))) + 1e-12)

    # Habets ANF needs equal-spectrum, mutually uncorrelated excitations. Raw
    # real excerpts do not reliably satisfy that condition. Instead, estimate
    # one scene spectrum and filter two independent Gaussian realizations.
    reference = np.concatenate([source_a, source_b])
    psd_freq, psd = welch(reference, fs=sample_rate, nperseg=2048, noverlap=1024)
    spectral_gain = np.sqrt(np.maximum(psd, 1e-12))
    spectral_gain /= math.sqrt(float(np.mean(np.square(spectral_gain))) + 1e-12)
    shaping_filter = firwin2(
        ANF_SHAPING_FIR_TAPS,
        psd_freq / (sample_rate / 2.0),
        spectral_gain,
    )
    rng = np.random.default_rng(stable_int(row["seed"], "anf_nonstationary_excitation"))
    white = rng.standard_normal((2, source_length + ANF_SHAPING_FIR_TAPS - 1))
    colored = np.stack([
        fftconvolve(channel, shaping_filter, mode="valid")[:source_length]
        for channel in white
    ])

    # A common slowly varying envelope preserves equal instantaneous power in
    # expectation while transferring real DEMAND nonstationarity. The Gaussian
    # carriers remain independent, so the ANF controls the spatial coherence.
    reference_power = 0.5 * (np.square(source_a) + np.square(source_b))
    envelope_window = np.ones(ANF_ENVELOPE_SAMPLES, dtype=np.float64) / ANF_ENVELOPE_SAMPLES
    envelope = np.sqrt(np.maximum(fftconvolve(reference_power, envelope_window, mode="same"), 1e-8))
    envelope /= math.sqrt(float(np.mean(np.square(envelope))) + 1e-12)
    inputs = colored * envelope[None, :]
    inputs -= np.mean(inputs, axis=1, keepdims=True)
    inputs /= np.sqrt(np.mean(np.square(inputs), axis=1, keepdims=True) + 1e-12)
    input_corr = float(np.mean(inputs[0] * inputs[1]))
    half = float(spacing_m) / 2.0
    params = ANFParameters(
        mic_positions=np.asarray([[-half, 0.0, 0.0], [half, 0.0, 0.0]], dtype=np.float64),
        sc_type="spherical",
        sample_frequency=sample_rate,
        nfft=ANF_NFFT,
    )
    outputs, target, _matrix = generate_signals(inputs, params, decomposition="evd", processing="balance+smooth")
    noise = np.asarray(outputs.T[context:context + length], dtype=np.float32)
    if noise.shape != (length, 2) or not np.all(np.isfinite(noise)):
        raise RuntimeError(f"Invalid ANF output: {noise.shape}")
    metadata = {
        "noise_path_a": str(path_a), "noise_path_b": str(path_b),
        "noise_start_a": int(start_a), "noise_start_b": int(start_b),
        "noise_excitation": "independent_gaussian_DEMAND_PSD_and_100ms_envelope",
        "noise_shaping_fir_taps": ANF_SHAPING_FIR_TAPS,
        "noise_envelope_samples": ANF_ENVELOPE_SAMPLES,
        "noise_input_zero_lag_corr": round(float(input_corr), 8),
    }
    if bool(config.get("validate_noise_coherence", False)):
        estimated = estimate_coherence(noise.T.astype(np.float64), ANF_NFFT)
        target_lr = target.matrix[0, 1, 1:]
        estimated_lr = estimated[0, 1, 1:]
        nmse = float(
            np.mean(np.abs(estimated_lr - target_lr) ** 2)
            / (np.mean(np.abs(target_lr) ** 2) + 1e-12)
        )
        metadata["anf_coherence_nmse_db"] = round(10.0 * math.log10(nmse + 1e-12), 6)
    return noise, metadata


def _render_rows(bundle: Mapping[str, object]) -> List[Dict[str, object]]:
    config = _WORKER_CONFIG
    sample_rate = int(config["sample_rate"])
    output_samples = int(config["output_samples"])
    rir_root = Path(str(config["rir_root"]))
    output_root = Path(str(config["output_root"]))
    split = str(bundle["split"])
    speech_paths = list(config[f"{split}_speech_paths"])
    spacings = dict(config["subject_spacings"])
    rendered: List[Dict[str, object]] = []

    shared_speech = None
    shared_noise = None
    shared_noise_meta = None
    shared_speech_meta = None
    if bool(bundle["paired_test"]):
        recipes = list(bundle["recipes"])
        first = recipes[0]
        room = ROOM_BY_SUBJECT[str(first["subject_id"])]
        brirs = [
            _load_brir_cached(str(brir_path(rir_root, str(first["subject_id"]), room, int(rt),
                                             int(first["distance_index"]), int(first["angle_deg"]))), sample_rate)
            for rt in TEST_RT60_MS[:1] if len(recipes) == 1
        ]
        if len(recipes) > 1:
            brirs = [
                _load_brir_cached(str(brir_path(rir_root, str(first["subject_id"]), room, int(rt),
                                                 int(first["distance_index"]), int(first["angle_deg"]))), sample_rate)
                for rt in TEST_RT60_MS
            ]
        prefix = max(len(value) for value in brirs) - 1
        shared_speech, path, start, active = choose_speech_context(
            speech_paths, int(first["speech_index"]), prefix, output_samples, sample_rate,
            stable_int(first["seed"], "speech"),
        )
        shared_speech_meta = (path, start, active, prefix)
        spacing = float(spacings[str(first["subject_id"])]["spacing_m"])
        shared_noise, shared_noise_meta = generate_diffuse_noise(first, output_samples, spacing)

    clean_cache: Dict[int, Tuple[np.ndarray, str]] = {}
    for row_index, row in enumerate(bundle["recipes"]):
        subject = str(row["subject_id"])
        room = ROOM_BY_SUBJECT[subject]
        brir_file = brir_path(rir_root, subject, room, int(row["rt60_ms"]), int(row["distance_index"]),
                              int(row["angle_deg"]))
        brir = _load_brir_cached(str(brir_file), sample_rate)
        if shared_speech is None:
            prefix = len(brir) - 1
            speech, speech_path, speech_start, active_ratio = choose_speech_context(
                speech_paths, int(row["speech_index"]), prefix, output_samples, sample_rate,
                stable_int(row["seed"], "speech"),
            )
            spacing = float(spacings[subject]["spacing_m"])
            noise, noise_meta = generate_diffuse_noise(row, output_samples, spacing)
            clean_reverb = render_context(speech, brir, prefix, output_samples)
        else:
            speech_path, speech_start, active_ratio, prefix = shared_speech_meta
            noise, noise_meta = shared_noise, shared_noise_meta
            rt_key = int(row["rt60_ms"])
            if rt_key not in clean_cache:
                clean_cache[rt_key] = (render_context(shared_speech, brir, prefix, output_samples), str(brir_file))
            clean_reverb = clean_cache[rt_key][0]
            spacing = float(spacings[subject]["spacing_m"])
        mixed, achieved_snr = mix_at_snr(clean_reverb, noise, float(row["snr_db"]))
        mixed = joint_normalize(mixed)
        class_index = CLASS_ANGLES_DEG.index(int(row["angle_deg"]))
        file_id = f"{bundle['task_id']}_r{row_index:03d}"
        relative_wav = Path("binaural") / f"{file_id}.wav"
        wav_path = output_root / SPLIT_DIR_NAMES[split] / relative_wav
        sf.write(wav_path, mixed, sample_rate, subtype="PCM_16")
        rendered.append({
            "file_id": file_id, "wav_path": str(relative_wav), "split": split,
            "class_index": class_index, "azimuth_deg": int(row["angle_deg"]),
            "rir_azimuth_deg": project_to_rir_angle(int(row["angle_deg"])),
            "subject_id": subject, "room_id": room.room_id,
            "room_x_m": room.dims_m[0], "room_y_m": room.dims_m[1], "room_z_m": room.dims_m[2],
            "rt60_s": int(row["rt60_ms"]) / 1000.0,
            "distance_m": room.distances_m[int(row["distance_index"])],
            "target_snr_db": int(row["snr_db"]), "achieved_snr_db": round(achieved_snr, 8),
            "speech_path": speech_path, "speech_speaker_id": speaker_id(Path(speech_path)),
            "speech_target_start_sample": int(speech_start), "speech_active_ratio": round(float(active_ratio), 6),
            "noise_scene": row["noise_scene"], "noise_channel_a": row["noise_channel_a"],
            "noise_channel_b": row["noise_channel_b"], **noise_meta,
            "noise_field_model": "Habets_ANF_spherical",
            "anf_nfft": ANF_NFFT, "anf_decomposition": "evd", "anf_processing": "balance+smooth",
            "ear_spacing_m": spacing, "ear_spacing_source": spacings[subject]["source"],
            "target_brir_path": str(brir_file), "renderer": "DP-RTF Roomsim_Campbell CIPIC BRIR",
            "paired_test_key": bundle["task_id"] if bool(bundle["paired_test"]) else "",
            "seed": row["seed"],
        })
    return rendered


def render_bundle(bundle: Mapping[str, object]) -> Dict[str, object]:
    return {"task_id": bundle["task_id"], "rows": _render_rows(bundle)}


def finalize_metadata(split_root: Path, expected_tasks: int, expected_clips: int) -> Dict[str, object]:
    completed = load_completed_tasks(split_root / "metadata.tasks.jsonl")
    if len(completed) != expected_tasks:
        raise RuntimeError(f"Incomplete split {split_root}: {len(completed)}/{expected_tasks}")
    rows = [row for task_id in sorted(completed) for row in completed[task_id]["rows"]]
    if len(rows) != expected_clips:
        raise RuntimeError(f"Unexpected clip count {len(rows)}/{expected_clips}")
    if len({row["file_id"] for row in rows}) != len(rows):
        raise RuntimeError(f"Duplicate file IDs in {split_root}")
    path = split_root / "metadata.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"num_tasks": expected_tasks, "num_clips": expected_clips, "metadata_path": str(path)}


def generate_split(split: str, bundles: Sequence[Mapping[str, object]], config: Mapping[str, object], workers: int) -> Dict[str, object]:
    split_root = Path(str(config["output_root"])) / SPLIT_DIR_NAMES[split]
    (split_root / "binaural").mkdir(parents=True, exist_ok=True)
    task_jsonl = split_root / "metadata.tasks.jsonl"
    completed = load_completed_tasks(task_jsonl)
    expected_ids = {str(bundle["task_id"]) for bundle in bundles}
    if set(completed) - expected_ids:
        raise RuntimeError(f"Unexpected existing tasks in {split_root}")
    pending = [bundle for bundle in bundles if str(bundle["task_id"]) not in completed]
    expected_clips = sum(len(bundle["recipes"]) for bundle in bundles)
    completed_clips = sum(len(payload["rows"]) for payload in completed.values())
    print(f"[{split}] tasks={len(completed)}/{len(bundles)} clips={completed_clips}/{expected_clips}", flush=True)
    start_time = time.time()
    if pending:
        with task_jsonl.open("a", encoding="utf-8") as output_handle:
            with ProcessPoolExecutor(
                max_workers=workers,
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
                    if offset == 1 or done % 10 == 0 or done == len(bundles):
                        elapsed = max(time.time() - start_time, 1e-6)
                        rate = offset / elapsed
                        eta = (len(pending) - offset) / rate if rate else float("inf")
                        progress = {
                            "split": split, "completed_tasks": done, "total_tasks": len(bundles),
                            "completed_clips": completed_clips, "total_clips": expected_clips,
                            "elapsed_sec_this_run": elapsed, "eta_sec": eta,
                        }
                        (split_root / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
                        print(f"[{split}] tasks={done}/{len(bundles)} clips={completed_clips}/{expected_clips} "
                              f"rate={rate:.3f} task/s eta={eta/60:.1f} min", flush=True)
    report = finalize_metadata(split_root, len(bundles), expected_clips)
    report["duration_sec_this_run"] = time.time() - start_time
    return report


def quality_check(output_root: Path, bundles_by_split: Mapping[str, Sequence[Mapping[str, object]]],
                  sample_rate: int, output_samples: int, rir_root: Path, inspect_limit: int) -> Dict[str, object]:
    report: Dict[str, object] = {"passed": True, "handedness": handedness_check(rir_root), "splits": {}}
    generated_speakers = []
    for split, bundles in bundles_by_split.items():
        path = output_root / split / "metadata.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected = sum(len(bundle["recipes"]) for bundle in bundles)
        if len(rows) != expected:
            raise RuntimeError(f"{split}: metadata {len(rows)} != {expected}")
        class_counts = Counter(int(row["class_index"]) for row in rows)
        if not class_counts or max(class_counts.values()) != min(class_counts.values()):
            raise RuntimeError(f"Unbalanced classes: {split} {class_counts}")
        snr_counts = Counter(int(float(row["target_snr_db"])) for row in rows)
        if set(snr_counts) != set(SNR_DB) and len(rows) > 1:
            raise RuntimeError(f"Missing SNR condition: {split} {snr_counts}")
        snr_error = max(abs(float(row["target_snr_db"]) - float(row["achieved_snr_db"])) for row in rows)
        if snr_error > 0.1:
            raise RuntimeError(f"SNR error: {split} {snr_error}")
        generated_speakers.append({row["speech_speaker_id"] for row in rows})
        step = max(1, len(rows) // max(1, inspect_limit))
        inspected = rows[::step][:inspect_limit]
        max_peak, min_rms = 0.0, float("inf")
        for row in inspected:
            wav_path = output_root / split / row["wav_path"]
            info = sf.info(wav_path)
            if info.samplerate != sample_rate or info.channels != 2 or info.frames != output_samples:
                raise RuntimeError(f"Invalid WAV: {wav_path} {info}")
            audio, _ = sf.read(wav_path, dtype="float32", always_2d=True)
            max_peak = max(max_peak, float(np.max(np.abs(audio))))
            min_rms = min(min_rms, float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))))
        report["splits"][split] = {
            "num_tasks": len(bundles), "num_clips": len(rows), "clips_per_class": dict(class_counts),
            "snr_counts": dict(snr_counts), "num_speech_speakers": len(generated_speakers[-1]),
            "max_snr_error_db": snr_error, "wav_files_inspected": len(inspected),
            "max_abs_peak": max_peak, "min_rms": min_rms,
        }
    check_disjoint(generated_speakers, "generated speech speaker")
    return report


def write_manifest(args: argparse.Namespace, output_root: Path, bundles: Mapping[str, Sequence[Mapping[str, object]]],
                   spacings: Mapping[str, Mapping[str, object]], inventory: Mapping[str, object],
                   reports: Mapping[str, object], quality: Mapping[str, object]) -> None:
    manifest = {
        "name": "librispeech_cipic_roomsim25_anf_nonstationary_v3", "mode": args.mode,
        "created_unix_time": time.time(), "generator": str(Path(__file__).resolve()),
        "git_commit": git_commit(Path(__file__).resolve().parents[1]), "python_version": sys.version,
        "numpy_version": np.__version__, "scipy_version": scipy.__version__, "soundfile_version": sf.__version__,
        "seed": args.seed, "sample_rate": args.sample_rate, "duration_sec": args.duration_sec,
        "class_angles_deg": list(CLASS_ANGLES_DEG), "snr_db": list(SNR_DB),
        "train_noise_scenes": list(TRAIN_NOISE_SCENES), "test_noise_scenes": list(TEST_NOISE_SCENES),
        "test_rt60_ms": list(TEST_RT60_MS), "demand_time_partitions": SPLIT_TIME_RANGES,
        "r0_fraction": {"train": 0.10, "val": 0.0, "test": 0.0},
        "noise_rendering": (
            "Habets ANF spherical diffuse field from two independent Gaussian excitations with "
            "a shared DEMAND-derived scene spectrum and 100 ms nonstationary energy envelope"
        ),
        "anf": {
            "nfft": ANF_NFFT,
            "decomposition": "evd",
            "processing": "balance+smooth",
            "shaping_fir_taps": ANF_SHAPING_FIR_TAPS,
            "envelope_samples": ANF_ENVELOPE_SAMPLES,
        },
        "ear_spacing": {"method": "CIPIC X1 head-width proxy", "fallback_m": ANTHRO_FALLBACK_M,
                        "subjects": dict(spacings)},
        "normalization": "one common scalar for both ears after SNR mixing",
        "rooms": [asdict(room) for room in ROOM_SPECS],
        "task_counts": {split: len(values) for split, values in bundles.items()},
        "clip_counts": {split: sum(len(value["recipes"]) for value in values) for split, values in bundles.items()},
        "inventory": dict(inventory), "split_reports": dict(reports),
        "quality_report": "quality_report.json", "quality_passed": bool(quality["passed"]),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rir_root", type=Path, default=Path("/disk2/bywang/data/RIR-CIPIC"))
    parser.add_argument("--demand_root", type=Path, default=Path("/disk2/bywang/data/demand"))
    parser.add_argument("--anthro_path", type=Path, default=Path("/disk2/bywang/data/HRTF/anthropometry/anthro.mat"))
    parser.add_argument("--train_speech_root", type=Path, default=Path("/disk2/bywang/data/LibriSpeech/train-clean-100"))
    parser.add_argument("--val_speech_root", type=Path, default=Path("/disk2/bywang/data/LibriSpeech/LibriSpeech_dev/dev-clean"))
    parser.add_argument("--test_speech_root", type=Path, default=Path("/disk2/bywang/data/LibriSpeech/LibriSpeech_test/test-clean"))
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration_sec", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=12, help="Workers per split")
    parser.add_argument("--parallel_splits", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hash_brir", action="store_true")
    parser.add_argument("--inventory_source", type=Path,
                        default=Path("/disk2/bywang/DOA-net/data/librispeech_cipic_roomsim25_v1/brir_inventory.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"Output exists; use --resume only for this exact protocol: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    speech_paths = validate_input_layout(args)
    spacings = load_subject_spacings(args.anthro_path)
    all_bundles = make_bundles(args.mode, args.seed)
    bundles = {split: all_bundles[split] for split in args.splits}
    expected = {split: sum(len(bundle["recipes"]) for bundle in values) for split, values in bundles.items()}
    print(f"mode={args.mode} expected_clips={expected}", flush=True)

    inventory_path = output_root / "brir_inventory.csv"
    if inventory_path.is_file() and args.resume:
        inventory = {"required_paths": sum(1 for _ in inventory_path.open(encoding="utf-8")) - 1,
                     "missing_paths": 0, "sha256_included": args.hash_brir,
                     "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest()}
    elif args.inventory_source.is_file() and not args.hash_brir:
        shutil.copy2(args.inventory_source, inventory_path)
        inventory = {"required_paths": sum(1 for _ in inventory_path.open(encoding="utf-8")) - 1,
                     "missing_paths": 0, "sha256_included": False,
                     "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
                     "source": str(args.inventory_source)}
    else:
        inventory = inventory_required_brirs(args.rir_root, inventory_path, args.hash_brir)

    config: Dict[str, object] = {
        "sample_rate": args.sample_rate, "output_samples": int(round(args.sample_rate * args.duration_sec)),
        "rir_root": str(args.rir_root.resolve()), "demand_root": str(args.demand_root.resolve()),
        "output_root": str(output_root), "subject_spacings": spacings,
        "validate_noise_coherence": args.mode == "smoke",
    }
    for split, paths in speech_paths.items():
        config[f"{split}_speech_paths"] = paths

    reports: Dict[str, object] = {}
    if args.parallel_splits and len(bundles) > 1:
        with ThreadPoolExecutor(max_workers=len(bundles)) as executor:
            futures = {executor.submit(generate_split, split, values, config, max(1, args.workers)): split
                       for split, values in bundles.items()}
            for future in as_completed(futures):
                reports[futures[future]] = future.result()
    else:
        for split in ("test", "val", "train"):
            if split in bundles:
                reports[split] = generate_split(split, bundles[split], config, max(1, args.workers))

    quality = quality_check(output_root, bundles, args.sample_rate, int(round(args.sample_rate * args.duration_sec)),
                            args.rir_root, 1000 if args.mode == "full" else 100000)
    (output_root / "quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    write_manifest(args, output_root, bundles, spacings, inventory, reports, quality)
    print(json.dumps({"output_root": str(output_root), "quality_passed": quality["passed"],
                      "clips": expected}, indent=2), flush=True)


if __name__ == "__main__":
    main()
