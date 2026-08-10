#!/usr/bin/env python3
"""Audit DNS3 noise, exclude human sounds, and create source-disjoint splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from silero_vad import get_speech_timestamps, load_silero_vad


HUMAN_SOUND_ROOT = "/m/0dgw9r"
TARGET_SAMPLE_RATE = 16000
OUTPUT_SECONDS = 2.0
MAX_BRIR_PREFIX_SECONDS = 0.95
MIN_FILE_SECONDS = OUTPUT_SECONDS + MAX_BRIR_PREFIX_SECONDS
VAD_THRESHOLD = 0.5
MIN_SPEECH_MS = 100
MAX_ALLOWED_SPEECH_SECONDS = 0.0
ACTIVE_FRAME_SECONDS = 0.020
ACTIVE_HOP_SECONDS = 0.010
MIN_ACTIVE_RATIO = 0.80
MAX_INACTIVE_SECONDS = 0.30
MAX_CLIPPED_FRACTION = 0.001
INVENTORY_SCHEMA_VERSION = 2

# Freesound filenames retain event names, unlike AudioSet IDs. This list is
# deliberately conservative and excludes voice as well as other bodily sounds.
HUMAN_FILENAME_PATTERN = re.compile(
    r"(?:speech|voice|talk|conversation|sing|vocal|laugh|cry|scream|shout|"
    r"yell|whisper|cough|sneeze|breath|snore|baby|child|crowd|people|person|"
    r"human|mouth|spit|hiccup|throat|munch|chew|eating|burp|gulp|sniff|snort)",
    re.IGNORECASE,
)
FREESOUND_ID_PATTERN = re.compile(r"_Freesound_(?:validated_)?(\d+)", re.IGNORECASE)
FREESOUND_CHUNK_PATTERN = re.compile(r"^(.*_Freesound_.*)_\d+$", re.IGNORECASE)


_VAD_MODEL = None


def stable_hash(*values: object) -> str:
    return hashlib.sha256("|".join(str(value) for value in values).encode("utf-8")).hexdigest()


def descendants(nodes: Mapping[str, Mapping[str, object]], root_id: str) -> set[str]:
    result: set[str] = set()
    pending = [root_id]
    while pending:
        node_id = pending.pop()
        if node_id in result:
            continue
        result.add(node_id)
        pending.extend(str(value) for value in nodes[node_id].get("child_ids", []))
    return result


def load_human_audioset_ids(metadata_root: Path, wanted_ids: set[str]) -> Tuple[set[str], set[str]]:
    ontology = json.loads((metadata_root / "ontology.json").read_text(encoding="utf-8"))
    nodes = {str(node["id"]): node for node in ontology}
    human_labels = descendants(nodes, HUMAN_SOUND_ROOT)
    matched: set[str] = set()
    human_recordings: set[str] = set()
    for name in ("balanced_train_segments.csv", "unbalanced_train_segments.csv", "eval_segments.csv"):
        path = metadata_root / name
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line in handle:
                if not line.strip() or line.startswith("#"):
                    continue
                row = next(csv.reader([line]))
                youtube_id = row[0].strip()
                if youtube_id not in wanted_ids:
                    continue
                matched.add(youtube_id)
                labels = {value.strip() for value in row[3].split(",")}
                if labels & human_labels:
                    human_recordings.add(youtube_id)
    missing = wanted_ids - matched
    if missing:
        raise RuntimeError(f"AudioSet metadata missing {len(missing)} DNS files, e.g. {sorted(missing)[:10]}")
    return human_recordings, human_labels


def source_identity(path: Path) -> Tuple[str, str]:
    match = FREESOUND_ID_PATTERN.search(path.stem)
    if match:
        return "freesound", f"freesound:{match.group(1)}"
    if "_freesound_" in path.stem.lower():
        chunk_match = FREESOUND_CHUNK_PATTERN.match(path.stem)
        source_name = chunk_match.group(1) if chunk_match else path.stem
        return "freesound", f"freesound:{source_name}"
    return "audioset", f"audioset:{path.stem}"


def resample_mono(audio: np.ndarray, source_sr: int) -> np.ndarray:
    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    if int(source_sr) != TARGET_SAMPLE_RATE:
        divisor = math.gcd(int(source_sr), TARGET_SAMPLE_RATE)
        mono = resample_poly(
            mono,
            TARGET_SAMPLE_RATE // divisor,
            int(source_sr) // divisor,
        ).astype(np.float32, copy=False)
    mono -= float(np.mean(mono))
    return mono


def frame_activity(signal: np.ndarray) -> Tuple[np.ndarray, float]:
    signal = np.asarray(signal, dtype=np.float32)
    frame = int(round(ACTIVE_FRAME_SECONDS * TARGET_SAMPLE_RATE))
    hop = int(round(ACTIVE_HOP_SECONDS * TARGET_SAMPLE_RATE))
    if len(signal) < frame:
        return np.zeros(0, dtype=bool), 0.0
    frames = np.lib.stride_tricks.sliding_window_view(signal, frame)[::hop].astype(np.float64)
    frames -= np.mean(frames, axis=1, keepdims=True)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    threshold = max(1e-4, 0.10 * float(np.percentile(rms, 95.0)))
    return rms >= threshold, threshold


def max_false_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        if bool(value):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def valid_window_starts(signal: np.ndarray) -> List[int]:
    output_samples = int(round(OUTPUT_SECONDS * TARGET_SAMPLE_RATE))
    prefix_samples = int(math.ceil(MAX_BRIR_PREFIX_SECONDS * TARGET_SAMPLE_RATE))
    if len(signal) < prefix_samples + output_samples:
        return []
    high = len(signal) - output_samples
    candidates = np.unique(np.linspace(prefix_samples, high, num=min(33, high - prefix_samples + 1), dtype=np.int64))
    max_inactive_frames = int(round(MAX_INACTIVE_SECONDS / ACTIVE_HOP_SECONDS))
    valid: List[int] = []
    for start in candidates:
        active, _threshold = frame_activity(signal[int(start) : int(start) + output_samples])
        if active.size and float(np.mean(active)) >= MIN_ACTIVE_RATIO and max_false_run(active) <= max_inactive_frames:
            valid.append(int(start))
    return valid


def init_vad_worker() -> None:
    global _VAD_MODEL
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("ORT_NUM_THREADS", "1")
    torch.set_num_threads(1)
    _VAD_MODEL = load_silero_vad(onnx=True)


def audit_file(payload: Tuple[str, bool, bool]) -> Dict[str, object]:
    path_string, audioset_human, filename_human = payload
    path = Path(path_string)
    source_kind, source_id = source_identity(path)
    reasons: List[str] = []
    try:
        info = sf.info(path)
        audio, source_sr = sf.read(path, dtype="float32", always_2d=True)
        mono = resample_mono(audio, int(source_sr))
        duration_sec = len(mono) / TARGET_SAMPLE_RATE
        finite = bool(np.all(np.isfinite(mono)))
        rms = math.sqrt(float(np.mean(np.square(mono, dtype=np.float64))) + 1e-20) if len(mono) else 0.0
        peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
        clipped_fraction = float(np.mean(np.abs(mono) >= 0.999)) if len(mono) else 1.0
        starts = valid_window_starts(mono) if finite else []
        if not finite:
            reasons.append("non_finite")
        if duration_sec + 1e-9 < MIN_FILE_SECONDS:
            reasons.append("too_short_for_brir_context")
        if rms < 1e-5:
            reasons.append("near_silent")
        if clipped_fraction > MAX_CLIPPED_FRACTION:
            reasons.append("clipped")
        if not starts:
            reasons.append("no_active_2s_window")
        if audioset_human:
            reasons.append("audioset_human_sound_label")
        if filename_human:
            reasons.append("human_sound_filename")

        speech_seconds = 0.0
        if not reasons or set(reasons) <= {"clipped"}:
            timestamps = get_speech_timestamps(
                torch.from_numpy(mono),
                _VAD_MODEL,
                sampling_rate=TARGET_SAMPLE_RATE,
                threshold=VAD_THRESHOLD,
                min_speech_duration_ms=MIN_SPEECH_MS,
                min_silence_duration_ms=100,
                return_seconds=False,
            )
            speech_seconds = sum(int(item["end"]) - int(item["start"]) for item in timestamps) / TARGET_SAMPLE_RATE
            if speech_seconds > MAX_ALLOWED_SPEECH_SECONDS:
                reasons.append("silero_speech_detected")
        return {
            "path": str(path.resolve()),
            "filename": path.name,
            "source_kind": source_kind,
            "source_id": source_id,
            "source_sample_rate": int(info.samplerate),
            "source_channels": int(info.channels),
            "source_subtype": str(info.subtype),
            "duration_sec": round(duration_sec, 6),
            "rms": round(rms, 10),
            "peak": round(peak, 10),
            "clipped_fraction": round(clipped_fraction, 10),
            "silero_speech_sec": round(speech_seconds, 6),
            "active_starts_sec": ";".join(f"{start / TARGET_SAMPLE_RATE:.6f}" for start in starts),
            "eligible": not reasons,
            "rejection_reasons": ";".join(reasons),
            "split": "",
        }
    except Exception as exc:
        return {
            "path": str(path.resolve()), "filename": path.name,
            "source_kind": source_kind, "source_id": source_id,
            "source_sample_rate": "", "source_channels": "", "source_subtype": "",
            "duration_sec": "", "rms": "", "peak": "", "clipped_fraction": "",
            "silero_speech_sec": "", "active_starts_sec": "", "eligible": False,
            "rejection_reasons": f"read_or_vad_error:{type(exc).__name__}:{exc}", "split": "",
        }


def assign_source_splits(rows: Sequence[Dict[str, object]], seed: int) -> Dict[str, int]:
    eligible_sources = sorted({str(row["source_id"]) for row in rows if bool(row["eligible"])},
                              key=lambda value: stable_hash(seed, value))
    train_end = int(round(0.80 * len(eligible_sources)))
    val_end = train_end + int(round(0.10 * len(eligible_sources)))
    source_split = {
        source_id: ("train" if index < train_end else "val" if index < val_end else "test")
        for index, source_id in enumerate(eligible_sources)
    }
    for row in rows:
        if bool(row["eligible"]):
            row["split"] = source_split[str(row["source_id"])]
    return dict(Counter(source_split.values()))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise_root", type=Path,
                        default=Path("/disk2/bywang/data/DNS3_official_extract_20260806/datasets/noise"))
    parser.add_argument("--audioset_metadata_root", type=Path,
                        default=Path("data/dns3_audioset_metadata"))
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_files", type=int, default=0, help="Debug-only limit; zero uses all files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Inventory output is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    files = sorted(args.noise_root.resolve().glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No DNS WAV files under {args.noise_root}")
    if args.max_files > 0:
        files = files[: args.max_files]

    audioset_ids = {path.stem for path in files if source_identity(path)[0] == "audioset"}
    human_audioset_ids, human_label_ids = load_human_audioset_ids(
        args.audioset_metadata_root.resolve(), audioset_ids
    )
    payloads = [
        (
            str(path),
            path.stem in human_audioset_ids,
            source_identity(path)[0] == "freesound" and bool(HUMAN_FILENAME_PATTERN.search(path.name)),
        )
        for path in files
    ]
    print(
        f"DNS files={len(files)} audioset={len(audioset_ids)} "
        f"audioset_human={len(human_audioset_ids)} workers={args.workers}",
        flush=True,
    )
    started = time.time()
    rows: List[Dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=init_vad_worker,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        for index, row in enumerate(executor.map(audit_file, payloads, chunksize=8), start=1):
            rows.append(row)
            if index == 1 or index % 500 == 0 or index == len(payloads):
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"[dns-audit] {index}/{len(payloads)} rate={index / elapsed:.2f} file/s "
                    f"eta={(len(payloads) - index) / (index / elapsed) / 60.0:.1f} min",
                    flush=True,
                )

    source_split_counts = assign_source_splits(rows, args.seed)
    rows.sort(key=lambda row: str(row["path"]))
    inventory_path = output_root / "dns3_noise_inventory.csv"
    write_csv(inventory_path, rows)
    eligible = [row for row in rows if bool(row["eligible"])]
    file_split_counts = Counter(str(row["split"]) for row in eligible)
    reason_counts = Counter(
        reason
        for row in rows
        for reason in str(row["rejection_reasons"]).split(";")
        if reason
    )
    summary = {
        "inventory_schema_version": INVENTORY_SCHEMA_VERSION,
        "created_unix_time": time.time(),
        "noise_root": str(args.noise_root.resolve()),
        "audioset_metadata_root": str(args.audioset_metadata_root.resolve()),
        "total_files": len(rows),
        "eligible_files": len(eligible),
        "rejected_files": len(rows) - len(eligible),
        "eligible_source_counts": source_split_counts,
        "eligible_file_counts": dict(file_split_counts),
        "rejection_reason_counts": dict(reason_counts),
        "human_sound_ontology_root": HUMAN_SOUND_ROOT,
        "human_sound_descendant_label_count": len(human_label_ids),
        "audioset_human_recording_count": len(human_audioset_ids),
        "freesound_human_filename_regex": HUMAN_FILENAME_PATTERN.pattern,
        "silero_vad": {
            "version": "6.2.0",
            "threshold": VAD_THRESHOLD,
            "min_speech_duration_ms": MIN_SPEECH_MS,
            "max_allowed_speech_seconds": MAX_ALLOWED_SPEECH_SECONDS,
        },
        "activity_filter": {
            "minimum_file_seconds": MIN_FILE_SECONDS,
            "output_seconds": OUTPUT_SECONDS,
            "max_brir_prefix_seconds": MAX_BRIR_PREFIX_SECONDS,
            "minimum_active_ratio": MIN_ACTIVE_RATIO,
            "maximum_inactive_seconds": MAX_INACTIVE_SECONDS,
            "framewise_dc_removal": True,
            "framewise_dc_removal_reason": (
                "prevents sparse transients from shifting digital silence above the activity threshold"
            ),
        },
        "split_method": "source_id sorted by sha256(seed|source_id), 80/10/10",
        "seed": args.seed,
        "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "elapsed_seconds": time.time() - started,
    }
    (output_root / "dns3_noise_inventory_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
