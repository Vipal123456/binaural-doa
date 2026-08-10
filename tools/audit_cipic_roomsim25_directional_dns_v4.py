#!/usr/bin/env python3
"""Full release-gate audit for the directional DNS v4 CIPIC dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_dns3_noise_inventory import (
    ACTIVE_HOP_SECONDS,
    MAX_INACTIVE_SECONDS,
    MIN_ACTIVE_RATIO,
    TARGET_SAMPLE_RATE,
    frame_activity,
    max_false_run,
    resample_mono,
)
from tools.generate_cipic_roomsim25 import CLASS_ANGLES_DEG, ROOM_BY_SUBJECT, brir_path
from tools.generate_cipic_roomsim25_directional_dns_v4 import (
    DATASET_NAME,
    EVAL_SIR_DB,
    TEST_CONDITIONS,
    TRAIN_SIR_BINS,
    load_noise_inventory,
    make_bundles,
)


EXPECTED_CLIPS = {"train": 120_000, "val": 12_000, "test": 64_800}
EXPECTED_TASKS = {"train": 750, "val": 150, "test": 8_100}
RECIPE_FIELDS = {
    "split": "split",
    "subject_id": "subject_id",
    "azimuth_deg": "target_angle_deg",
    "noise_azimuth_deg": "noise_angle_deg",
    "target_sir_db": "sir_db",
    "noise_content_path": "noise_path",
    "noise_source_id": "noise_source_id",
    "noise_source_kind": "noise_source_kind",
    "noise_content_start_sec": "noise_target_start_sec",
    "seed": "seed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/librispeech_cipic_roomsim25_directional_dns_v4"),
    )
    parser.add_argument(
        "--noise-inventory",
        type=Path,
        default=Path("data/dns3_directional_v4_inventory/dns3_noise_inventory.csv"),
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, errors: List[str]) -> Mapping[str, object]:
    if not path.is_file():
        errors.append(f"missing JSON file: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid JSON {path}: {type(exc).__name__}: {exc}")
        return {}


def read_task_log(path: Path, errors: List[str]) -> Dict[str, Mapping[str, object]]:
    tasks: Dict[str, Mapping[str, object]] = {}
    if not path.is_file():
        errors.append(f"missing task log: {path}")
        return tasks
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
                task_id = str(payload["task_id"])
            except Exception as exc:
                errors.append(f"invalid task JSON {path}:{line_number}: {exc}")
                continue
            if task_id in tasks:
                errors.append(f"duplicate task id in {path}: {task_id}")
            tasks[task_id] = payload
    return tasks


def close_enough(actual: object, expected: object, tolerance: float = 1e-6) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= tolerance
        except (TypeError, ValueError):
            return False
    return str(actual) == str(expected)


def sir_bin(value: float) -> int:
    for index, (low, high) in enumerate(TRAIN_SIR_BINS):
        if low <= value < high:
            return index
    return -1


def inspect_wav(payload: Tuple[str, str]) -> Mapping[str, object]:
    file_id, path_string = payload
    path = Path(path_string)
    try:
        info = sf.info(path)
        audio, _ = sf.read(path, dtype="float32", always_2d=True)
        finite = bool(np.all(np.isfinite(audio)))
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        rms = math.sqrt(float(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0
        channel_rms = np.sqrt(np.mean(np.square(audio, dtype=np.float64), axis=0)).tolist()
        audio_hash = hashlib.blake2b(audio.tobytes(), digest_size=16).hexdigest()
        return {
            "file_id": file_id,
            "path": path_string,
            "header": (info.samplerate, info.channels, info.frames, info.format, info.subtype),
            "finite": finite,
            "peak": peak,
            "rms": rms,
            "channel_rms": channel_rms,
            "audio_hash": audio_hash,
            "error": "",
        }
    except Exception as exc:
        return {
            "file_id": file_id,
            "path": path_string,
            "error": f"{type(exc).__name__}: {exc}",
        }


def inspect_noise_windows(
    payload: Tuple[str, Sequence[float]],
) -> Tuple[str, List[Tuple[float, float, int, bool, str]]]:
    path_string, starts_sec = payload
    try:
        audio, sample_rate = sf.read(path_string, dtype="float32", always_2d=True)
        mono = resample_mono(audio, int(sample_rate))
    except Exception as exc:
        return path_string, [(0.0, 0.0, 10**9, False, f"{type(exc).__name__}: {exc}")]
    max_gap = int(round(MAX_INACTIVE_SECONDS / ACTIVE_HOP_SECONDS))
    results = []
    for start_sec in starts_sec:
        start = int(round(float(start_sec) * TARGET_SAMPLE_RATE))
        window = mono[start : start + 2 * TARGET_SAMPLE_RATE]
        active, _threshold = frame_activity(window)
        ratio = float(np.mean(active)) if active.size else 0.0
        gap = max_false_run(active)
        valid = bool(
            len(window) == 2 * TARGET_SAMPLE_RATE
            and ratio + 1e-12 >= MIN_ACTIVE_RATIO
            and gap <= max_gap
        )
        results.append((float(start_sec), ratio, gap, valid, ""))
    return path_string, results


def validate_protocol_groups(
    rows_by_split: Mapping[str, Sequence[Mapping[str, str]]],
    errors: List[str],
    warnings: List[str],
) -> Mapping[str, object]:
    stats: Dict[str, object] = {}

    train_groups: Dict[Tuple[str, int], List[Mapping[str, str]]] = defaultdict(list)
    for row in rows_by_split["train"]:
        train_groups[(row["subject_id"], int(row["azimuth_deg"]))].append(row)
    bad_train = []
    for key, rows in train_groups.items():
        sir_counts = Counter(sir_bin(float(row["target_sir_db"])) for row in rows)
        rt_counts = Counter(float(row["rt60_s"]) for row in rows)
        target_distances = Counter(float(row["distance_m"]) for row in rows)
        noise_distances = Counter(float(row["noise_distance_m"]) for row in rows)
        if (
            len(rows) != 160
            or sir_counts != {0: 40, 1: 40, 2: 40, 3: 40}
            or rt_counts.get(0.0) != 16
            or max(target_distances.values()) - min(target_distances.values()) > 1
            or max(noise_distances.values()) - min(noise_distances.values()) > 1
        ):
            bad_train.append(str(key))
    if bad_train:
        errors.append(f"unbalanced train subject-angle groups: {bad_train[:10]}")

    val_groups: Dict[Tuple[str, int], List[Mapping[str, str]]] = defaultdict(list)
    for row in rows_by_split["val"]:
        val_groups[(row["subject_id"], int(row["azimuth_deg"]))].append(row)
    bad_val = []
    for key, rows in val_groups.items():
        sir_counts = Counter(float(row["target_sir_db"]) for row in rows)
        if len(rows) != 80 or sir_counts != {float(value): 16 for value in EVAL_SIR_DB}:
            bad_val.append(str(key))
    if bad_val:
        errors.append(f"unbalanced validation subject-angle groups: {bad_val[:10]}")

    test_groups: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
    for row in rows_by_split["test"]:
        test_groups[row["paired_test_key"]].append(row)
    # TEST_CONDITIONS stores RT60 in milliseconds, while metadata.csv stores
    # rt60_s in seconds.  Normalize before comparing the paired sweep.
    expected_conditions = {(float(rt) / 1000.0, float(sir)) for rt, sir in TEST_CONDITIONS}
    fixed_fields = (
        "subject_id",
        "azimuth_deg",
        "distance_m",
        "speech_path",
        "speech_target_start_sample",
        "noise_content_path",
        "noise_content_start_sec",
        "noise_azimuth_deg",
        "noise_distance_m",
    )
    bad_test = []
    test_cells: Counter = Counter()
    for key, rows in test_groups.items():
        conditions = {(float(row["rt60_s"]), float(row["target_sir_db"])) for row in rows}
        drifting = [field for field in fixed_fields if len({row[field] for row in rows}) != 1]
        if len(rows) != 8 or conditions != expected_conditions or drifting:
            bad_test.append((key, len(rows), drifting))
        first = rows[0]
        test_cells[(first["subject_id"], first["distance_m"], first["azimuth_deg"])] += 1
    if bad_test:
        errors.append(f"invalid paired test groups: {bad_test[:10]}")
    if len(test_cells) != 675 or set(test_cells.values()) != {12}:
        errors.append(
            f"invalid test subject-distance-angle realizations: cells={len(test_cells)} "
            f"counts={Counter(test_cells.values())}"
        )

    warnings.extend(
        [
            "The test CSV is the union of an SIR sweep at RT60=0.6 s and an RT60 sweep "
            "at SIR=5 dB; report the two sweeps separately when interpreting robustness.",
            "Each CIPIC subject occurs in one simulated room, so the held-out test measures "
            "joint unseen-subject and unseen-room generalization, not isolated head generalization.",
        ]
    )
    stats.update(
        {
            "train_subject_angle_groups": len(train_groups),
            "val_subject_angle_groups": len(val_groups),
            "test_paired_groups": len(test_groups),
            "test_subject_distance_angle_cells": len(test_cells),
        }
    )
    return stats


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    inventory_path = args.noise_inventory.resolve()
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, object] = {}

    manifest = read_json(dataset_root / "manifest.json", errors)
    quality = read_json(dataset_root / "quality_report.json", errors)
    if manifest:
        if manifest.get("name") != DATASET_NAME or manifest.get("mode") != "full":
            errors.append(f"unexpected manifest identity: {manifest.get('name')} / {manifest.get('mode')}")
        if manifest.get("quality_passed") is not True:
            errors.append("manifest quality_passed is not true")
        if manifest.get("clip_counts") != EXPECTED_CLIPS:
            errors.append(f"unexpected manifest clip counts: {manifest.get('clip_counts')}")
        if int(manifest.get("sample_rate", 0)) != 16_000 or float(manifest.get("duration_sec", 0)) != 2.0:
            errors.append("unexpected sample rate or duration in manifest")
    if quality and quality.get("passed") is not True:
        errors.append("generator quality report did not pass")

    if not inventory_path.is_file():
        errors.append(f"missing noise inventory: {inventory_path}")
        noise_by_split = {"train": [], "val": [], "test": []}
    else:
        try:
            noise_by_split = load_noise_inventory(inventory_path)
            actual_inventory_hash = sha256(inventory_path)
            manifest_hash = manifest.get("noise_inventory", {}).get("inventory_sha256") if manifest else None
            if manifest_hash != actual_inventory_hash:
                errors.append(
                    f"noise inventory hash mismatch: manifest={manifest_hash} actual={actual_inventory_hash}"
                )
            stats["noise_inventory_sha256"] = actual_inventory_hash
        except Exception as exc:
            errors.append(f"failed to load noise inventory: {type(exc).__name__}: {exc}")
            noise_by_split = {"train": [], "val": [], "test": []}

    expected_bundles = make_bundles("full", 42, noise_by_split) if all(noise_by_split.values()) else {}
    rows_by_split: Dict[str, List[Mapping[str, str]]] = {}
    split_sets: Dict[str, Mapping[str, set]] = {}
    selected_noise: Dict[str, set] = defaultdict(set)
    wav_payloads: List[Tuple[str, str]] = []

    for split in ("train", "val", "test"):
        split_root = dataset_root / split
        metadata_path = split_root / "metadata.csv"
        if not metadata_path.is_file():
            errors.append(f"missing metadata CSV: {metadata_path}")
            rows: List[Mapping[str, str]] = []
        else:
            with metadata_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        rows_by_split[split] = rows
        if len(rows) != EXPECTED_CLIPS[split]:
            errors.append(f"{split}: metadata rows {len(rows)} != {EXPECTED_CLIPS[split]}")
        file_ids = [row.get("file_id", "") for row in rows]
        if len(set(file_ids)) != len(rows):
            errors.append(f"{split}: duplicate file_id values")

        tasks = read_task_log(split_root / "metadata.tasks.jsonl", errors)
        if len(tasks) != EXPECTED_TASKS[split]:
            errors.append(f"{split}: task count {len(tasks)} != {EXPECTED_TASKS[split]}")
        task_rows = [row for payload in tasks.values() for row in payload.get("rows", [])]
        if {row.get("file_id") for row in task_rows} != set(file_ids):
            errors.append(f"{split}: metadata.csv and task log file_id sets differ")

        if expected_bundles:
            expected_tasks = {bundle["task_id"]: bundle for bundle in expected_bundles[split]}
            if set(tasks) != set(expected_tasks):
                errors.append(f"{split}: generated and expected task IDs differ")
            mismatch_count = 0
            for task_id in sorted(set(tasks) & set(expected_tasks)):
                actual_rows = tasks[task_id].get("rows", [])
                recipes = expected_tasks[task_id]["recipes"]
                if len(actual_rows) != len(recipes):
                    mismatch_count += 1
                    continue
                for actual, recipe in zip(actual_rows, recipes):
                    for actual_field, recipe_field in RECIPE_FIELDS.items():
                        if not close_enough(actual.get(actual_field), recipe[recipe_field]):
                            mismatch_count += 1
                            break
            if mismatch_count:
                errors.append(f"{split}: {mismatch_count} task recipes differ from deterministic protocol")

        wav_root = split_root / "binaural"
        wav_files = list(wav_root.glob("*.wav")) if wav_root.is_dir() else []
        wav_names = {path.name for path in wav_files}
        metadata_names = {Path(row.get("wav_path", "")).name for row in rows}
        missing_wavs = metadata_names - wav_names
        orphan_wavs = wav_names - metadata_names
        if missing_wavs:
            errors.append(f"{split}: {len(missing_wavs)} metadata WAVs are missing")
        if orphan_wavs:
            errors.append(f"{split}: {len(orphan_wavs)} orphan WAVs are not in metadata")

        class_counts = Counter(int(row["class_index"]) for row in rows)
        expected_per_class = EXPECTED_CLIPS[split] // len(CLASS_ANGLES_DEG)
        if set(class_counts) != set(range(len(CLASS_ANGLES_DEG))) or set(class_counts.values()) != {
            expected_per_class
        }:
            errors.append(f"{split}: unbalanced classes {class_counts}")

        recipe_keys = Counter(
            (
                row["subject_id"],
                row["azimuth_deg"],
                row["rt60_s"],
                row["distance_m"],
                row["target_sir_db"],
                row["speech_path"],
                row["speech_target_start_sample"],
                row["noise_content_path"],
                row["noise_content_start_sec"],
                row["noise_azimuth_deg"],
                row["noise_distance_m"],
            )
            for row in rows
        )
        if max(recipe_keys.values(), default=0) > 1:
            errors.append(f"{split}: exact duplicate generation recipes found")

        max_sir_error = 0.0
        for row in rows:
            try:
                target_sir = float(row["target_sir_db"])
                achieved_sir = float(row["achieved_sir_db"])
                separation = abs(int(row["azimuth_deg"]) - int(row["noise_azimuth_deg"]))
                max_sir_error = max(max_sir_error, abs(target_sir - achieved_sir))
                if separation < 20 or int(row["angular_separation_deg"]) != separation:
                    errors.append(f"{split}: invalid angular separation in {row.get('file_id')}")
                    break
                noise_path = str(Path(row["noise_content_path"]).resolve())
                selected_noise[noise_path].add(round(float(row["noise_content_start_sec"]), 6))
                for path_field in ("speech_path", "noise_content_path", "target_brir_path", "noise_brir_path"):
                    if not Path(row[path_field]).is_file():
                        errors.append(f"{split}: missing source path {row[path_field]}")
                        break
            except Exception as exc:
                errors.append(f"{split}: malformed metadata row {row.get('file_id')}: {exc}")
                break
        if max_sir_error > 1e-4:
            errors.append(f"{split}: SIR error {max_sir_error} dB exceeds tolerance")

        split_sets[split] = {
            "noise_sources": {row["noise_source_id"] for row in rows},
            "speech_speakers": {row["speech_speaker_id"] for row in rows},
            "subjects": {row["subject_id"] for row in rows},
        }
        for row in rows:
            wav_payloads.append((row["file_id"], str((split_root / row["wav_path"]).resolve())))
        stats[split] = {
            "metadata_rows": len(rows),
            "task_count": len(tasks),
            "wav_count": len(wav_files),
            "classes": dict(class_counts),
            "max_sir_error_db": max_sir_error,
            "noise_sources": len(split_sets[split]["noise_sources"]),
            "speech_speakers": len(split_sets[split]["speech_speakers"]),
            "subjects": sorted(split_sets[split]["subjects"]),
        }

    for kind in ("noise_sources", "speech_speakers", "subjects"):
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = split_sets.get(left, {}).get(kind, set()) & split_sets.get(right, {}).get(kind, set())
            if overlap:
                errors.append(f"{kind} leakage between {left} and {right}: {sorted(overlap)[:10]}")

    if all(rows_by_split.get(split) for split in ("train", "val", "test")):
        stats["protocol_groups"] = validate_protocol_groups(rows_by_split, errors, warnings)

    print(f"Auditing {len(wav_payloads)} generated WAV files", flush=True)
    wav_errors = 0
    audio_hashes: Dict[str, str] = {}
    rms_values: List[float] = []
    peak_values: List[float] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for index, result in enumerate(executor.map(inspect_wav, wav_payloads), start=1):
            if result.get("error"):
                errors.append(f"WAV read error {result['path']}: {result['error']}")
                wav_errors += 1
            else:
                if tuple(result["header"]) != (16_000, 2, 32_000, "WAV", "PCM_16"):
                    errors.append(f"invalid WAV header {result['path']}: {result['header']}")
                    wav_errors += 1
                if not result["finite"] or float(result["rms"]) <= 1e-4:
                    errors.append(f"silent/non-finite WAV {result['path']}: rms={result.get('rms')}")
                    wav_errors += 1
                if float(result["peak"]) > 0.98005:
                    errors.append(f"WAV exceeds normalization peak {result['path']}: {result['peak']}")
                    wav_errors += 1
                if min(float(value) for value in result["channel_rms"]) <= 1e-6:
                    errors.append(f"silent WAV channel {result['path']}: {result['channel_rms']}")
                    wav_errors += 1
                audio_hash = str(result["audio_hash"])
                if audio_hash in audio_hashes:
                    errors.append(
                        f"exact duplicate WAV content: {audio_hashes[audio_hash]} and {result['path']}"
                    )
                    wav_errors += 1
                else:
                    audio_hashes[audio_hash] = str(result["path"])
                rms_values.append(float(result["rms"]))
                peak_values.append(float(result["peak"]))
            if index % 10_000 == 0:
                print(f"[wav-audit] {index}/{len(wav_payloads)}", flush=True)
            if wav_errors >= 100:
                errors.append("WAV audit stopped after 100 errors")
                break
    if rms_values:
        stats["audio"] = {
            "inspected": len(rms_values),
            "rms_quantiles": np.quantile(rms_values, [0.0, 0.001, 0.01, 0.5, 0.99, 1.0]).tolist(),
            "peak_quantiles": np.quantile(peak_values, [0.0, 0.5, 0.99, 1.0]).tolist(),
        }

    allowed_starts: Dict[str, set] = {}
    inventory_split: Dict[str, str] = {}
    if inventory_path.is_file():
        with inventory_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["eligible"].lower() != "true":
                    continue
                path = str(Path(row["path"]).resolve())
                allowed_starts[path] = {
                    round(float(value), 6) for value in row["active_starts_sec"].split(";") if value
                }
                inventory_split[path] = row["split"]
    membership_errors = []
    for path, starts in selected_noise.items():
        for start in starts:
            if path not in allowed_starts or start not in allowed_starts[path]:
                membership_errors.append((path, start))
    if membership_errors:
        errors.append(f"selected noise windows absent from audited inventory: {membership_errors[:10]}")

    print(f"Rechecking {sum(len(v) for v in selected_noise.values())} selected DNS windows", flush=True)
    bad_windows = []
    min_activity = 1.0
    max_gap = 0
    noise_payloads = [(path, sorted(starts)) for path, starts in selected_noise.items()]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for index, (path, results) in enumerate(executor.map(inspect_noise_windows, noise_payloads), start=1):
            for start, ratio, gap, valid, detail in results:
                min_activity = min(min_activity, ratio)
                max_gap = max(max_gap, gap)
                if not valid:
                    bad_windows.append((path, start, ratio, gap, detail))
            if index % 5_000 == 0:
                print(f"[noise-window-audit] {index}/{len(noise_payloads)}", flush=True)
    if bad_windows:
        errors.append(f"invalid selected DNS activity windows: {bad_windows[:10]}")
    stats["selected_noise_windows"] = {
        "paths": len(noise_payloads),
        "windows": sum(len(value) for value in selected_noise.values()),
        "minimum_active_ratio": min_activity,
        "maximum_inactive_frames": max_gap,
        "invalid": len(bad_windows),
    }

    train_source_counts = Counter(row["noise_source_id"] for row in rows_by_split.get("train", []))
    if train_source_counts:
        ordered = sorted(train_source_counts.values(), reverse=True)
        top_source, top_count = train_source_counts.most_common(1)[0]
        stats["train_noise_source_weighting"] = {
            "top_source": top_source,
            "top_source_count": top_count,
            "top_source_fraction": top_count / EXPECTED_CLIPS["train"],
            "top10_fraction": sum(ordered[:10]) / EXPECTED_CLIPS["train"],
            "top100_fraction": sum(ordered[:100]) / EXPECTED_CLIPS["train"],
        }
        warnings.append(
            "DNS records are sampled uniformly by eligible file, not uniformly by original source_id; "
            f"the largest source contributes {top_count}/{EXPECTED_CLIPS['train']} train clips."
        )

    report = {
        "passed": not errors,
        "dataset_root": str(dataset_root),
        "created_unix_time": time.time(),
        "elapsed_seconds": time.time() - started,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "errors": errors, "warnings": warnings}, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
