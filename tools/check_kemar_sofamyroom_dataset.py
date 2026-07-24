#!/usr/bin/env python3
"""Sanity-check KEMAR + SofaMyRoom dataset metadata and files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import soundfile as sf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_roots", type=Path, nargs="+", required=True)
    p.add_argument("--check_audio", action="store_true")
    p.add_argument("--output_json", type=Path)
    return p.parse_args()


def read_rows(root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for meta_path in sorted(root.glob("*/metadata.csv")):
        with meta_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["_metadata_path"] = str(meta_path)
                row["_dataset_root"] = str(root)
                rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No */metadata.csv found under {root}")
    return rows


def as_float(value: str) -> float:
    if value == "" or value.lower() == "clean":
        return float("nan")
    return float(value)


def numeric_summary(values: Sequence[float]) -> Dict[str, float | int | None]:
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }


def count_key(rows: Iterable[Dict[str, str]], key: str) -> Dict[str, int]:
    return dict(sorted(Counter(row.get(key, "") for row in rows).items()))


def class_balance(rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    counts = Counter(int(row["doa_class"]) for row in rows)
    vals = np.asarray([counts.get(i, 0) for i in range(72)], dtype=np.int64)
    return {
        "num_classes_present": int(np.sum(vals > 0)),
        "min": int(vals.min()),
        "mean": float(vals.mean()),
        "max": int(vals.max()),
        "missing_classes": [int(i) for i, v in enumerate(vals) if v == 0],
    }


def circular_angle_ok(row: Dict[str, str]) -> bool:
    klass = int(row["doa_class"])
    kemar = float(row["kemar_azimuth_deg"])
    az = float(row["azimuth_deg"])
    expected_kemar = (klass * 5) % 360
    expected_az = ((expected_kemar + 180.0) % 360.0) - 180.0
    return abs(kemar - expected_kemar) < 1e-5 and abs(az - expected_az) < 1e-5


def check_audio_row(row: Dict[str, str]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    wav_path = Path(row["wav_path"])
    clean_value = row.get("clean_reverb_path", "")
    brir_value = row.get("brir_path", "")
    clean_path = Path(clean_value) if clean_value else None
    brir_path = Path(brir_value) if brir_value else None
    expected_sr = int(row["sample_rate"])
    expected_len = int(round(float(row["duration_sec"]) * expected_sr))
    brir_sr = int(row["brir_fs"])
    expected_brir_len = int(round(float(row["brir_duration_sec"]) * brir_sr))

    out["wav_exists"] = wav_path.exists()
    out["clean_exists"] = None if clean_path is None else clean_path.exists()
    out["brir_exists"] = None if brir_path is None else brir_path.exists()
    required_exists = [out["wav_exists"]]
    if clean_path is not None:
        required_exists.append(bool(out["clean_exists"]))
    if brir_path is not None:
        required_exists.append(bool(out["brir_exists"]))
    if not all(required_exists):
        out["ok"] = False
        return out

    wav_info = sf.info(str(wav_path))
    out.update({
        "wav_sr_ok": wav_info.samplerate == expected_sr,
        "wav_len_ok": wav_info.frames == expected_len,
        "wav_channels_ok": wav_info.channels == 2,
    })
    if clean_path is not None:
        clean = np.load(clean_path, mmap_mode="r")
        out["clean_shape_ok"] = tuple(clean.shape) == (expected_len, 2)
    else:
        out["clean_shape_ok"] = None
    if brir_path is not None:
        brir = np.load(brir_path, mmap_mode="r")
        out["brir_shape_ok"] = tuple(brir.shape) == (expected_brir_len, 2)
    else:
        out["brir_shape_ok"] = None
    out["ok"] = bool(all(v for v in out.values() if v is not None))
    return out


def summarize_root(root: Path, check_audio: bool) -> Dict[str, object]:
    rows = read_rows(root)
    by_split: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)

    split_summary: Dict[str, object] = {}
    for split, split_rows in sorted(by_split.items()):
        target_rt60 = [float(r["target_rt60"]) for r in split_rows]
        estimated_rt60 = [as_float(r["estimated_rt60"]) for r in split_rows]
        rt60_abs_err = [
            abs(float(r["estimated_rt60"]) - float(r["target_rt60"]))
            for r in split_rows
            if r.get("estimated_rt60")
        ]
        source_clearance = [float(r["source_wall_clearance_m"]) for r in split_rows]
        receiver_clearance = [float(r["receiver_wall_clearance_m"]) for r in split_rows]
        snr_values = [as_float(r["snr_db"]) for r in split_rows]

        audio_checks = []
        if check_audio:
            for row in split_rows:
                audio_checks.append(check_audio_row(row))

        split_summary[split] = {
            "num_samples": len(split_rows),
            "class_balance": class_balance(split_rows),
            "angle_label_ok_rate": float(np.mean([circular_angle_ok(r) for r in split_rows])),
            "room_size_counts": count_key(split_rows, "room_size"),
            "room_id_counts": count_key(split_rows, "room_id"),
            "snr_counts": count_key(split_rows, "snr_db"),
            "noise_scene_counts": count_key(split_rows, "noise_scene"),
            "source_distance_counts": count_key(split_rows, "source_distance_m"),
            "target_rt60": numeric_summary(target_rt60),
            "estimated_rt60": numeric_summary(estimated_rt60),
            "rt60_abs_error": numeric_summary(rt60_abs_err),
            "source_wall_clearance_m": numeric_summary(source_clearance),
            "receiver_wall_clearance_m": numeric_summary(receiver_clearance),
            "snr_db_numeric": numeric_summary(snr_values),
        }
        if check_audio:
            split_summary[split]["audio_ok_rate"] = float(np.mean([bool(c["ok"]) for c in audio_checks]))
            failed = [split_rows[i]["file_id"] for i, c in enumerate(audio_checks) if not c["ok"]]
            split_summary[split]["audio_failed_file_ids"] = failed[:20]

    return {
        "dataset_root": str(root),
        "num_samples": len(rows),
        "splits": split_summary,
    }


def main() -> None:
    args = parse_args()
    summary = {
        "roots": [summarize_root(root, args.check_audio) for root in args.dataset_roots],
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
