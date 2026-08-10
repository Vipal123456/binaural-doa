#!/usr/bin/env python3
"""Generate Roomsim25 mixtures with one source-disjoint directional DNS interferer."""

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
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import scipy
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_cipic_roomsim25 import (
    CLASS_ANGLES_DEG,
    ROOM_BY_SUBJECT,
    ROOM_SPECS,
    SPLIT_DIR_NAMES,
    SPLIT_SUBJECTS,
    _load_brir_cached,
    balanced_schedule,
    brir_path,
    check_disjoint,
    choose_speech_context,
    git_commit,
    handedness_check,
    inventory_required_brirs,
    joint_normalize,
    list_audio_files,
    load_completed_tasks,
    project_to_rir_angle,
    render_context,
    resample_nd,
    speaker_id,
    stable_int,
)


DATASET_NAME = "librispeech_cipic_roomsim25_directional_dns_v4"
EVAL_SIR_DB: Tuple[int, ...] = (-5, 0, 5, 10, 15)
TRAIN_SIR_BINS: Tuple[Tuple[float, float], ...] = ((-5.0, 0.0), (0.0, 5.0), (5.0, 10.0), (10.0, 15.0))
TEST_CONDITIONS: Tuple[Tuple[int, int], ...] = (
    (600, -5), (600, 0), (600, 5), (600, 10), (600, 15),
    (200, 5), (400, 5), (800, 5),
)
TRAIN_REALIZATIONS = 160
VAL_REALIZATIONS = 80
TEST_REALIZATIONS = 12
MIN_NOISE_SEPARATION_DEG = 20
MAX_BRIR_PREFIX_SECONDS = 0.95
TRAIN_R0_FRACTION = 0.10
VAL_R0_FRACTION = 0.10


_WORKER_CONFIG: Dict[str, object] = {}


def init_worker(config: Mapping[str, object]) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = dict(config)
    _load_brir_cached.cache_clear()
    _load_noise_cached.cache_clear()


def load_noise_inventory(path: Path) -> Dict[str, List[Dict[str, object]]]:
    summary_path = path.with_name("dns3_noise_inventory_summary.json")
    if not summary_path.is_file():
        raise FileNotFoundError(f"DNS inventory summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    activity_filter = summary.get("activity_filter", {})
    if int(summary.get("inventory_schema_version", 0)) < 2 or not activity_filter.get(
        "framewise_dc_removal", False
    ):
        raise RuntimeError(
            "DNS inventory predates framewise-DC activity filtering; rebuild it before generation"
        )
    result: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row["eligible"]).lower() != "true":
                continue
            split = str(row["split"])
            if split not in result:
                raise RuntimeError(f"Invalid DNS split {split!r}")
            starts = [float(value) for value in str(row["active_starts_sec"]).split(";") if value]
            if not starts:
                raise RuntimeError(f"Eligible DNS row has no active starts: {row['path']}")
            result[split].append({
                "path": str(row["path"]),
                "source_id": str(row["source_id"]),
                "source_kind": str(row["source_kind"]),
                "active_starts_sec": starts,
            })
    for split, rows in result.items():
        if not rows:
            raise RuntimeError(f"DNS inventory has no eligible {split} files")
        rows.sort(key=lambda row: (str(row["source_id"]), str(row["path"])))
    source_sets = [{str(row["source_id"]) for row in result[split]} for split in ("train", "val", "test")]
    check_disjoint(source_sets, "DNS source")
    return result


def ordered_noise_records(records: Sequence[Mapping[str, object]], split: str, seed: int) -> List[Mapping[str, object]]:
    return sorted(records, key=lambda row: stable_int(seed, split, row["source_id"], row["path"]))


def exact_rt_schedule(room, count: int, zero_fraction: float, seed: int) -> List[int]:
    zero_count = int(round(count * zero_fraction))
    nonzero = tuple(value for value in room.rt60_ms if value != 0)
    values = [0] * zero_count
    values.extend(int(value) for value in balanced_schedule(nonzero, count - zero_count, seed + 1))
    random.Random(seed + 2).shuffle(values)
    return values


def separation_bin(delta: int) -> str:
    if delta <= 40:
        return "20_40"
    if delta <= 80:
        return "45_80"
    return "85_160"


def noise_angle_schedule(target_angle: int, count: int, seed: int) -> List[int]:
    grouped: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for angle in CLASS_ANGLES_DEG:
        delta = abs(int(angle) - int(target_angle))
        if delta < MIN_NOISE_SEPARATION_DEG:
            continue
        side = "left" if angle < target_angle else "right"
        grouped[(side, separation_bin(delta))].append(int(angle))
    strata = tuple(sorted(grouped))
    if not strata:
        raise RuntimeError(f"No valid interferer angle for target {target_angle}")
    selected_strata = balanced_schedule(strata, count, seed)
    counters: Counter = Counter()
    result: List[int] = []
    for stratum in selected_strata:
        candidates = sorted(grouped[stratum], key=lambda value: stable_int(seed, stratum, value))
        result.append(candidates[counters[stratum] % len(candidates)])
        counters[stratum] += 1
    return result


def continuous_train_sir(bin_index: int, seed: int) -> float:
    low, high = TRAIN_SIR_BINS[bin_index]
    unit = ((stable_int(seed, "sir") % 1_000_000) + 0.5) / 1_000_000.0
    return round(low + unit * (high - low), 6)


def make_recipe(
    split: str,
    subject: str,
    target_angle: int,
    noise_angle: int,
    rt60_ms: int,
    target_distance_index: int,
    noise_distance_index: int,
    sir_db: float,
    speech_index: int,
    noise_record: Mapping[str, object],
    seed: int,
) -> Dict[str, object]:
    starts = list(noise_record["active_starts_sec"])
    start_sec = float(starts[stable_int(seed, "noise_start") % len(starts)])
    return {
        "split": split,
        "subject_id": subject,
        "target_angle_deg": int(target_angle),
        "noise_angle_deg": int(noise_angle),
        "rt60_ms": int(rt60_ms),
        "target_distance_index": int(target_distance_index),
        "noise_distance_index": int(noise_distance_index),
        "sir_db": float(sir_db),
        "speech_index": int(speech_index),
        "noise_path": str(noise_record["path"]),
        "noise_source_id": str(noise_record["source_id"]),
        "noise_source_kind": str(noise_record["source_kind"]),
        "noise_target_start_sec": start_sec,
        "seed": int(seed),
    }


def make_train_val_bundles(
    split: str,
    seed: int,
    noise_records: Sequence[Mapping[str, object]],
    smoke: bool,
) -> List[Dict[str, object]]:
    subjects = SPLIT_SUBJECTS[split][:1] if smoke else SPLIT_SUBJECTS[split]
    angles = (-80, -30, 0, 30, 80) if smoke else CLASS_ANGLES_DEG
    count = (8 if split == "train" else 10) if smoke else (
        TRAIN_REALIZATIONS if split == "train" else VAL_REALIZATIONS
    )
    zero_fraction = TRAIN_R0_FRACTION if split == "train" else VAL_R0_FRACTION
    records = ordered_noise_records(noise_records, split, seed)
    bundles: List[Dict[str, object]] = []
    noise_cursor = 0
    for subject in subjects:
        room = ROOM_BY_SUBJECT[subject]
        for target_angle in angles:
            group_seed = stable_int(seed, split, subject, target_angle)
            if split == "train":
                repeats = count // len(TRAIN_SIR_BINS)
                conditions = [(index, repetition) for index in range(len(TRAIN_SIR_BINS)) for repetition in range(repeats)]
            else:
                repeats = count // len(EVAL_SIR_DB)
                conditions = [(sir, repetition) for sir in EVAL_SIR_DB for repetition in range(repeats)]
            random.Random(group_seed).shuffle(conditions)
            rt_values = exact_rt_schedule(room, count, zero_fraction, group_seed + 10)
            target_distances = balanced_schedule(tuple(range(len(room.distances_m))), count, group_seed + 20)
            noise_distances = balanced_schedule(tuple(range(len(room.distances_m))), count, group_seed + 30)
            noise_angles = noise_angle_schedule(int(target_angle), count, group_seed + 40)
            rows: List[Dict[str, object]] = []
            for index, condition in enumerate(conditions):
                row_seed = stable_int(group_seed, index, condition)
                sir_db = continuous_train_sir(int(condition[0]), row_seed) if split == "train" else float(condition[0])
                noise_record = records[noise_cursor % len(records)]
                noise_cursor += 1
                rows.append(make_recipe(
                    split, subject, int(target_angle), noise_angles[index], rt_values[index],
                    int(target_distances[index]), int(noise_distances[index]), sir_db,
                    stable_int(row_seed, "speech_index"), noise_record, row_seed,
                ))
            bundles.append({
                "task_id": f"{split}_s{subject}_a{CLASS_ANGLES_DEG.index(target_angle):02d}",
                "split": split,
                "paired_test": False,
                "recipes": rows,
            })
    return bundles


def make_test_bundles(
    seed: int,
    noise_records: Sequence[Mapping[str, object]],
    smoke: bool,
) -> List[Dict[str, object]]:
    records = ordered_noise_records(noise_records, "test", seed)
    rooms = [room for room in ROOM_SPECS if room.split == "test"]
    if smoke:
        rooms = rooms[:1]
    angles = (-80, -30, 0, 30, 80) if smoke else CLASS_ANGLES_DEG
    realization_count = 4 if smoke else TEST_REALIZATIONS
    bundles: List[Dict[str, object]] = []
    noise_cursor = 0
    for room in rooms:
        subjects = room.subjects[:1] if smoke else room.subjects
        distances = range(1 if smoke else len(room.distances_m))
        for subject in subjects:
            for target_distance_index in distances:
                for target_angle in angles:
                    group_seed = stable_int(seed, "test", subject, target_distance_index, target_angle)
                    noise_angles = noise_angle_schedule(int(target_angle), realization_count, group_seed + 10)
                    noise_distances = balanced_schedule(
                        tuple(range(len(room.distances_m))), realization_count, group_seed + 20
                    )
                    for realization in range(realization_count):
                        base_seed = stable_int(group_seed, realization)
                        noise_record = records[noise_cursor % len(records)]
                        noise_cursor += 1
                        shared = make_recipe(
                            "test", subject, int(target_angle), noise_angles[realization], 600,
                            int(target_distance_index), int(noise_distances[realization]), 5.0,
                            stable_int(base_seed, "speech_index"), noise_record, base_seed,
                        )
                        rows = []
                        for rt60_ms, sir_db in TEST_CONDITIONS:
                            row = dict(shared)
                            row["rt60_ms"] = int(rt60_ms)
                            row["sir_db"] = float(sir_db)
                            row["seed"] = stable_int(base_seed, rt60_ms, sir_db)
                            rows.append(row)
                        bundles.append({
                            "task_id": (
                                f"test_s{subject}_d{target_distance_index:02d}_"
                                f"a{CLASS_ANGLES_DEG.index(target_angle):02d}_r{realization:02d}"
                            ),
                            "split": "test",
                            "paired_test": True,
                            "recipes": rows,
                        })
    return bundles


def make_bundles(
    mode: str,
    seed: int,
    noise_by_split: Mapping[str, Sequence[Mapping[str, object]]],
) -> Dict[str, List[Dict[str, object]]]:
    smoke = mode == "smoke"
    return {
        "train": make_train_val_bundles("train", seed, noise_by_split["train"], smoke),
        "val": make_train_val_bundles("val", seed, noise_by_split["val"], smoke),
        "test": make_test_bundles(seed, noise_by_split["test"], smoke),
    }


def validate_inputs(args: argparse.Namespace) -> Dict[str, List[str]]:
    roots = {"train": args.train_speech_root, "val": args.val_speech_root, "test": args.test_speech_root}
    paths = {split: [str(path) for path in list_audio_files(root)] for split, root in roots.items()}
    check_disjoint(({speaker_id(Path(path)) for path in values} for values in paths.values()), "LibriSpeech speaker")
    check_disjoint(SPLIT_SUBJECTS.values(), "CIPIC subject")
    return paths


@lru_cache(maxsize=64)
def _load_noise_cached(path_string: str, sample_rate: int) -> np.ndarray:
    audio, source_sr = sf.read(path_string, dtype="float32", always_2d=True)
    mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
    mono = resample_nd(mono, int(source_sr), int(sample_rate))
    mono -= float(np.mean(mono))
    return mono


def load_noise_context(row: Mapping[str, object], prefix: int, output_samples: int, sample_rate: int) -> np.ndarray:
    mono = _load_noise_cached(str(row["noise_path"]), sample_rate)
    target_start = int(round(float(row["noise_target_start_sec"]) * sample_rate))
    if target_start < prefix or target_start + output_samples > len(mono):
        raise RuntimeError(
            f"DNS context out of range: {row['noise_path']} start={target_start} prefix={prefix} len={len(mono)}"
        )
    context = np.asarray(mono[target_start - prefix : target_start + output_samples], dtype=np.float32)
    context = context - float(np.mean(context))
    return context


def active_sample_mask(stereo: np.ndarray, sample_rate: int) -> np.ndarray:
    mono_energy = np.mean(np.square(np.asarray(stereo, dtype=np.float64)), axis=1)
    frame = max(1, int(round(0.020 * sample_rate)))
    hop = max(1, int(round(0.010 * sample_rate)))
    starts = range(0, max(1, len(mono_energy) - frame + 1), hop)
    values = [(start, math.sqrt(float(np.mean(mono_energy[start : start + frame])) + 1e-12)) for start in starts]
    if not values:
        return np.ones(len(stereo), dtype=bool)
    threshold = max(1e-4, 0.05 * float(np.percentile([value for _start, value in values], 95.0)))
    mask = np.zeros(len(stereo), dtype=bool)
    for start, value in values:
        if value >= threshold:
            mask[start : min(len(mask), start + frame)] = True
    return mask if np.any(mask) else np.ones(len(stereo), dtype=bool)


def mix_at_active_sir(target: np.ndarray, interferer: np.ndarray, sir_db: float, sample_rate: int) -> Tuple[np.ndarray, float, float]:
    mask = active_sample_mask(target, sample_rate)
    target_power = float(np.mean(np.square(target[mask], dtype=np.float64)))
    interferer = interferer - np.mean(interferer, axis=0, keepdims=True)
    interferer_power = float(np.mean(np.square(interferer[mask], dtype=np.float64)))
    if target_power <= 1e-12 or interferer_power <= 1e-12:
        raise RuntimeError("Cannot mix silent target or directional interferer")
    gain = math.sqrt(target_power / (interferer_power * 10.0 ** (float(sir_db) / 10.0)))
    scaled = interferer * gain
    achieved = 10.0 * math.log10(
        target_power / float(np.mean(np.square(scaled[mask], dtype=np.float64)))
    )
    return (target + scaled).astype(np.float32), float(achieved), float(np.mean(mask))


def render_one(row: Mapping[str, object], shared: Mapping[str, object] | None = None) -> Tuple[np.ndarray, Dict[str, object]]:
    config = _WORKER_CONFIG
    sample_rate = int(config["sample_rate"])
    output_samples = int(config["output_samples"])
    rir_root = Path(str(config["rir_root"]))
    split = str(row["split"])
    subject = str(row["subject_id"])
    room = ROOM_BY_SUBJECT[subject]
    target_path = brir_path(
        rir_root, subject, room, int(row["rt60_ms"]), int(row["target_distance_index"]),
        int(row["target_angle_deg"]),
    )
    noise_path = brir_path(
        rir_root, subject, room, int(row["rt60_ms"]), int(row["noise_distance_index"]),
        int(row["noise_angle_deg"]),
    )
    target_brir = _load_brir_cached(str(target_path), sample_rate)
    noise_brir = _load_brir_cached(str(noise_path), sample_rate)
    prefix = int(shared["prefix"]) if shared is not None else max(len(target_brir), len(noise_brir)) - 1
    if prefix > int(round(MAX_BRIR_PREFIX_SECONDS * sample_rate)):
        raise RuntimeError(f"BRIR prefix exceeds audited DNS context: {prefix}")
    speech_paths = list(config[f"{split}_speech_paths"])
    if shared is None:
        speech, speech_path, speech_start, speech_active = choose_speech_context(
            speech_paths, int(row["speech_index"]), prefix, output_samples, sample_rate,
            stable_int(row["seed"], "speech"),
        )
        noise = load_noise_context(row, prefix, output_samples, sample_rate)
    else:
        speech = np.asarray(shared["speech"])
        speech_path = str(shared["speech_path"])
        speech_start = int(shared["speech_start"])
        speech_active = float(shared["speech_active"])
        noise = np.asarray(shared["noise"])
    clean = render_context(speech, target_brir, prefix, output_samples)
    directional = render_context(noise, noise_brir, prefix, output_samples)
    mixed, achieved_sir, target_activity = mix_at_active_sir(
        clean, directional, float(row["sir_db"]), sample_rate
    )
    mixed = joint_normalize(mixed)
    metadata = {
        "split": split,
        "class_index": CLASS_ANGLES_DEG.index(int(row["target_angle_deg"])),
        "azimuth_deg": int(row["target_angle_deg"]),
        "rir_azimuth_deg": project_to_rir_angle(int(row["target_angle_deg"])),
        "subject_id": subject,
        "room_id": room.room_id,
        "room_x_m": room.dims_m[0], "room_y_m": room.dims_m[1], "room_z_m": room.dims_m[2],
        "rt60_s": int(row["rt60_ms"]) / 1000.0,
        "distance_m": room.distances_m[int(row["target_distance_index"])],
        "target_sir_db": float(row["sir_db"]),
        "achieved_sir_db": round(achieved_sir, 8),
        "interference_metric": "active-frame binaural SIR",
        "target_active_sample_ratio": round(target_activity, 6),
        "speech_path": speech_path,
        "speech_speaker_id": speaker_id(Path(speech_path)),
        "speech_target_start_sample": speech_start,
        "speech_active_ratio": round(float(speech_active), 6),
        "noise_content_path": str(row["noise_path"]),
        "noise_source_id": str(row["noise_source_id"]),
        "noise_source_kind": str(row["noise_source_kind"]),
        "noise_content_start_sec": float(row["noise_target_start_sec"]),
        "noise_azimuth_deg": int(row["noise_angle_deg"]),
        "noise_rir_azimuth_deg": project_to_rir_angle(int(row["noise_angle_deg"])),
        "noise_distance_m": room.distances_m[int(row["noise_distance_index"])],
        "angular_separation_deg": abs(int(row["target_angle_deg"]) - int(row["noise_angle_deg"])),
        "target_brir_path": str(target_path),
        "noise_brir_path": str(noise_path),
        "renderer": "DP-RTF Roomsim_Campbell CIPIC BRIR for target and interferer",
        "seed": int(row["seed"]),
    }
    return mixed, metadata


def render_bundle(bundle: Mapping[str, object]) -> Dict[str, object]:
    config = _WORKER_CONFIG
    output_root = Path(str(config["output_root"]))
    split = str(bundle["split"])
    rows = list(bundle["recipes"])
    shared = None
    if bool(bundle["paired_test"]):
        sample_rate = int(config["sample_rate"])
        output_samples = int(config["output_samples"])
        rir_root = Path(str(config["rir_root"]))
        first = rows[0]
        subject = str(first["subject_id"])
        room = ROOM_BY_SUBJECT[subject]
        brirs = []
        for row in rows:
            brirs.extend([
                _load_brir_cached(str(brir_path(
                    rir_root, subject, room, int(row["rt60_ms"]), int(row["target_distance_index"]),
                    int(row["target_angle_deg"]),
                )), sample_rate),
                _load_brir_cached(str(brir_path(
                    rir_root, subject, room, int(row["rt60_ms"]), int(row["noise_distance_index"]),
                    int(row["noise_angle_deg"]),
                )), sample_rate),
            ])
        prefix = max(len(value) for value in brirs) - 1
        speech_paths = list(config["test_speech_paths"])
        speech, path, start, active = choose_speech_context(
            speech_paths, int(first["speech_index"]), prefix, output_samples, sample_rate,
            stable_int(first["seed"], "paired_speech"),
        )
        noise = load_noise_context(first, prefix, output_samples, sample_rate)
        shared = {"prefix": prefix, "speech": speech, "speech_path": path,
                  "speech_start": start, "speech_active": active, "noise": noise}

    rendered = []
    for index, row in enumerate(rows):
        try:
            audio, metadata = render_one(row, shared)
        except Exception as exc:
            context = {
                "task_id": bundle["task_id"],
                "row_index": index,
                "recipe": row,
            }
            raise RuntimeError(f"Failed directional DNS render: {json.dumps(context, sort_keys=True)}") from exc
        file_id = f"{bundle['task_id']}_q{index:02d}"
        relative = Path("binaural") / f"{file_id}.wav"
        sf.write(output_root / split / relative, audio, int(config["sample_rate"]), subtype="PCM_16")
        metadata.update({
            "file_id": file_id,
            "wav_path": str(relative),
            "paired_test_key": str(bundle["task_id"]) if bool(bundle["paired_test"]) else "",
        })
        rendered.append(metadata)
    return {"task_id": str(bundle["task_id"]), "rows": rendered}


def finalize_metadata(split_root: Path, expected_tasks: int, expected_clips: int) -> Dict[str, object]:
    completed = load_completed_tasks(split_root / "metadata.tasks.jsonl")
    if len(completed) != expected_tasks:
        raise RuntimeError(f"Incomplete split {split_root}: {len(completed)}/{expected_tasks}")
    rows = [row for task_id in sorted(completed) for row in completed[task_id]["rows"]]
    if len(rows) != expected_clips or len({row["file_id"] for row in rows}) != expected_clips:
        raise RuntimeError(f"Invalid metadata rows in {split_root}: {len(rows)}/{expected_clips}")
    path = split_root / "metadata.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"num_tasks": expected_tasks, "num_clips": expected_clips, "metadata_path": str(path)}


def generate_split(split: str, bundles: Sequence[Mapping[str, object]], config: Mapping[str, object], workers: int) -> Dict[str, object]:
    split_root = Path(str(config["output_root"])) / split
    (split_root / "binaural").mkdir(parents=True, exist_ok=True)
    task_jsonl = split_root / "metadata.tasks.jsonl"
    completed = load_completed_tasks(task_jsonl)
    expected_ids = {str(bundle["task_id"]) for bundle in bundles}
    unexpected = set(completed) - expected_ids
    if unexpected:
        raise RuntimeError(f"Unexpected existing {split} tasks: {sorted(unexpected)[:10]}")
    pending = [bundle for bundle in bundles if str(bundle["task_id"]) not in completed]
    expected_clips = sum(len(bundle["recipes"]) for bundle in bundles)
    completed_clips = sum(len(payload["rows"]) for payload in completed.values())
    print(f"[{split}] tasks={len(completed)}/{len(bundles)} clips={completed_clips}/{expected_clips}", flush=True)
    started = time.time()
    if pending:
        with task_jsonl.open("a", encoding="utf-8") as output_handle:
            with ProcessPoolExecutor(
                max_workers=max(1, workers), initializer=init_worker, initargs=(config,),
                mp_context=mp.get_context("spawn"),
            ) as executor:
                for offset, payload in enumerate(executor.map(render_bundle, pending, chunksize=1), start=1):
                    output_handle.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n")
                    output_handle.flush()
                    os.fsync(output_handle.fileno())
                    completed_clips += len(payload["rows"])
                    done = len(completed) + offset
                    if offset == 1 or done % 10 == 0 or done == len(bundles):
                        elapsed = max(time.time() - started, 1e-6)
                        rate = offset / elapsed
                        progress = {
                            "split": split, "completed_tasks": done, "total_tasks": len(bundles),
                            "completed_clips": completed_clips, "total_clips": expected_clips,
                            "elapsed_sec_this_run": elapsed,
                            "eta_sec": (len(pending) - offset) / rate if rate else None,
                        }
                        (split_root / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
                        print(
                            f"[{split}] tasks={done}/{len(bundles)} clips={completed_clips}/{expected_clips} "
                            f"rate={rate:.3f} task/s eta={progress['eta_sec'] / 60.0:.1f} min",
                            flush=True,
                        )
    report = finalize_metadata(split_root, len(bundles), expected_clips)
    report["duration_sec_this_run"] = time.time() - started
    return report


def quality_check(
    output_root: Path,
    bundles_by_split: Mapping[str, Sequence[Mapping[str, object]]],
    sample_rate: int,
    output_samples: int,
    rir_root: Path,
    inspect_limit: int,
) -> Dict[str, object]:
    report: Dict[str, object] = {"passed": True, "handedness": handedness_check(rir_root), "splits": {}}
    source_sets, speaker_sets = [], []
    for split, bundles in bundles_by_split.items():
        metadata_path = output_root / split / "metadata.csv"
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected = sum(len(bundle["recipes"]) for bundle in bundles)
        if len(rows) != expected:
            raise RuntimeError(f"{split}: metadata {len(rows)} != {expected}")
        class_counts = Counter(int(row["class_index"]) for row in rows)
        if set(class_counts) != {CLASS_ANGLES_DEG.index(angle) for angle in (
            (-80, -30, 0, 30, 80) if len(rows) < 1000 else CLASS_ANGLES_DEG
        )} or len(set(class_counts.values())) != 1:
            raise RuntimeError(f"Unbalanced classes: {split} {class_counts}")
        separations = [int(row["angular_separation_deg"]) for row in rows]
        if min(separations) < MIN_NOISE_SEPARATION_DEG:
            raise RuntimeError(f"Invalid interferer separation in {split}")
        sir_error = max(abs(float(row["target_sir_db"]) - float(row["achieved_sir_db"])) for row in rows)
        if sir_error > 0.1:
            raise RuntimeError(f"SIR error: {split} {sir_error}")
        source_sets.append({row["noise_source_id"] for row in rows})
        speaker_sets.append({row["speech_speaker_id"] for row in rows})
        step = max(1, len(rows) // max(1, inspect_limit))
        inspected = rows[::step][:inspect_limit]
        peaks, rms_values = [], []
        for row in inspected:
            path = output_root / split / row["wav_path"]
            info = sf.info(path)
            if info.samplerate != sample_rate or info.channels != 2 or info.frames != output_samples:
                raise RuntimeError(f"Invalid WAV: {path} {info}")
            audio, _ = sf.read(path, dtype="float32", always_2d=True)
            peaks.append(float(np.max(np.abs(audio))))
            rms_values.append(math.sqrt(float(np.mean(np.square(audio, dtype=np.float64)))))
        report["splits"][split] = {
            "num_tasks": len(bundles), "num_clips": len(rows),
            "clips_per_class": dict(class_counts), "num_dns_sources": len(source_sets[-1]),
            "num_speech_speakers": len(speaker_sets[-1]), "max_sir_error_db": sir_error,
            "minimum_angular_separation_deg": min(separations), "wav_files_inspected": len(inspected),
            "max_abs_peak": max(peaks), "min_rms": min(rms_values),
        }
    check_disjoint(source_sets, "generated DNS source")
    check_disjoint(speaker_sets, "generated speech speaker")
    return report


def write_manifest(
    args: argparse.Namespace,
    output_root: Path,
    bundles: Mapping[str, Sequence[Mapping[str, object]]],
    inventory: Mapping[str, object],
    reports: Mapping[str, object],
    quality: Mapping[str, object],
) -> None:
    manifest = {
        "name": DATASET_NAME, "mode": args.mode, "created_unix_time": time.time(),
        "generator": str(Path(__file__).resolve()), "git_commit": git_commit(ROOT),
        "python_version": sys.version, "numpy_version": np.__version__,
        "scipy_version": scipy.__version__, "soundfile_version": sf.__version__,
        "seed": args.seed, "sample_rate": args.sample_rate, "duration_sec": args.duration_sec,
        "class_angles_deg": list(CLASS_ANGLES_DEG),
        "training_sir_distribution_db": "equal-count continuous strata [-5,0), [0,5), [5,10), [10,15]",
        "validation_sir_db": list(EVAL_SIR_DB),
        "test_conditions_rt60_ms_sir_db": [list(value) for value in TEST_CONDITIONS],
        "r0_fraction": {"train": TRAIN_R0_FRACTION, "val": VAL_R0_FRACTION, "test": 0.0},
        "interferer": {
            "content": "DNS Challenge 3 wideband noise after AudioSet human-sound, filename, and Silero VAD exclusion",
            "spatialization": "one directional point interferer rendered through a second Roomsim-CIPIC BRIR",
            "same_subject_room_rt60_as_target": True,
            "independent_distance": True,
            "minimum_angular_separation_deg": MIN_NOISE_SEPARATION_DEG,
            "angle_strata_deg": ["20-40", "45-80", "85-160"],
        },
        "sir_definition": "binaural target/interferer power on target-active samples after BRIR rendering",
        "normalization": "one common scalar for both ears after SIR mixing",
        "noise_inventory": dict(inventory),
        "rooms": [asdict(room) for room in ROOM_SPECS],
        "task_counts": {split: len(values) for split, values in bundles.items()},
        "clip_counts": {split: sum(len(bundle["recipes"]) for bundle in values) for split, values in bundles.items()},
        "split_reports": dict(reports), "quality_report": "quality_report.json",
        "quality_passed": bool(quality["passed"]),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rir_root", type=Path, default=Path("/disk2/bywang/data/RIR-CIPIC"))
    parser.add_argument("--train_speech_root", type=Path, default=Path("/disk2/bywang/data/LibriSpeech/train-clean-100"))
    parser.add_argument("--val_speech_root", type=Path, default=Path("/disk2/bywang/data/LibriSpeech/LibriSpeech_dev/dev-clean"))
    parser.add_argument("--test_speech_root", type=Path, default=Path("/disk2/bywang/data/LibriSpeech/LibriSpeech_test/test-clean"))
    parser.add_argument("--noise_inventory", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration_sec", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=6, help="Workers per split")
    parser.add_argument("--parallel_splits", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hash_brir", action="store_true")
    parser.add_argument("--inventory_source", type=Path,
                        default=Path("data/librispeech_cipic_roomsim25_v1/brir_inventory.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"Output exists; use --resume only for this protocol: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    speech_paths = validate_inputs(args)
    noise_by_split = load_noise_inventory(args.noise_inventory.resolve())
    all_bundles = make_bundles(args.mode, args.seed, noise_by_split)
    bundles = {split: all_bundles[split] for split in args.splits}
    expected = {split: sum(len(bundle["recipes"]) for bundle in values) for split, values in bundles.items()}
    print(f"mode={args.mode} expected_clips={expected}", flush=True)

    inventory_path = output_root / "brir_inventory.csv"
    if inventory_path.is_file() and args.resume:
        brir_inventory = {
            "required_paths": sum(1 for _ in inventory_path.open(encoding="utf-8")) - 1,
            "sha256_included": args.hash_brir,
            "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        }
    elif args.inventory_source.is_file() and not args.hash_brir:
        shutil.copy2(args.inventory_source, inventory_path)
        brir_inventory = {
            "required_paths": sum(1 for _ in inventory_path.open(encoding="utf-8")) - 1,
            "sha256_included": False,
            "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
            "source": str(args.inventory_source.resolve()),
        }
    else:
        brir_inventory = inventory_required_brirs(args.rir_root, inventory_path, args.hash_brir)

    config: Dict[str, object] = {
        "sample_rate": args.sample_rate,
        "output_samples": int(round(args.sample_rate * args.duration_sec)),
        "rir_root": str(args.rir_root.resolve()),
        "output_root": str(output_root),
    }
    for split, paths in speech_paths.items():
        config[f"{split}_speech_paths"] = paths

    reports: Dict[str, object] = {}
    if args.parallel_splits and len(bundles) > 1:
        with ThreadPoolExecutor(max_workers=len(bundles)) as executor:
            futures = {
                executor.submit(generate_split, split, values, config, max(1, args.workers)): split
                for split, values in bundles.items()
            }
            for future in as_completed(futures):
                reports[futures[future]] = future.result()
    else:
        for split in ("test", "val", "train"):
            if split in bundles:
                reports[split] = generate_split(split, bundles[split], config, max(1, args.workers))

    quality = quality_check(
        output_root, bundles, args.sample_rate, int(round(args.sample_rate * args.duration_sec)),
        args.rir_root, 1000 if args.mode == "full" else 100000,
    )
    (output_root / "quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    noise_summary_path = args.noise_inventory.with_name("dns3_noise_inventory_summary.json")
    noise_inventory = json.loads(noise_summary_path.read_text(encoding="utf-8"))
    noise_inventory["path"] = str(args.noise_inventory.resolve())
    noise_inventory["brir_inventory"] = brir_inventory
    write_manifest(args, output_root, bundles, noise_inventory, reports, quality)
    print(json.dumps({"output_root": str(output_root), "quality_passed": quality["passed"],
                      "clips": expected}, indent=2), flush=True)


if __name__ == "__main__":
    main()
