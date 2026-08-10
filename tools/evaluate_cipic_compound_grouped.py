#!/usr/bin/env python3
"""Evaluate a CIPIC25 checkpoint on the paired compound-noise stress test."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_kemar_grouped import collect_rows, summarize, write_csv
from utils.config import load_config


REFERENCE_CONDITION = "R_dir0_nodiff"
CONDITION_ORDER = (
    "R_dir0_nodiff",
    "A_sir5_diff5",
    "B_sir0_diff5",
    "C_sirm5_diff5",
    "D_sir0_diff10",
    "E_sir0_diff0",
)
METRIC_FIELDS = (
    "count",
    "accuracy",
    "mae",
    "median",
    "acc_at_5",
    "acc_at_10",
    "fb_err",
    "opp_err",
    "large_err",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=50)
    return parser.parse_args()


def separation_bin(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value <= 40.0:
        return "20_40"
    if value <= 80.0:
        return "45_80"
    return "85_160"


def grouped_summary(rows: List[dict], field: str, order=None) -> List[dict]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    keys = [key for key in (order or sorted(groups)) if key in groups]
    return [{field: key, **summarize(groups[key])} for key in keys]


def paired_reference_summary(rows: List[dict]) -> List[dict]:
    groups: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for row in rows:
        groups[row["paired_test_key"]][row["condition"]] = row
    complete = {
        key: values
        for key, values in groups.items()
        if set(values) == set(CONDITION_ORDER)
    }
    if len(complete) != len(groups):
        raise RuntimeError(f"Incomplete condition pairs: {len(complete)}/{len(groups)}")
    result = []
    for condition in CONDITION_ORDER:
        reference_errors = np.asarray(
            [values[REFERENCE_CONDITION]["error_deg"] for values in complete.values()],
            dtype=np.float64,
        )
        condition_errors = np.asarray(
            [values[condition]["error_deg"] for values in complete.values()],
            dtype=np.float64,
        )
        delta = condition_errors - reference_errors
        result.append({
            "condition": condition,
            "count": len(delta),
            "reference_mae": float(reference_errors.mean()),
            "condition_mae": float(condition_errors.mean()),
            "delta_mae_vs_reference": float(delta.mean()),
            "improved_fraction": float((delta < 0.0).mean()),
            "tied_fraction": float((delta == 0.0).mean()),
            "worsened_fraction": float((delta > 0.0).mean()),
        })
    return result


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, [])
    cfg.dataset.test_root = str(Path(args.test_root).resolve())
    cfg.train.batch_size = int(args.batch_size)
    cfg.train.num_workers = int(args.num_workers)
    cfg.train.device = str(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(
        cfg=cfg,
        checkpoint_path=Path(args.checkpoint),
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        device_override=args.device,
        log_interval=int(args.log_interval),
    )
    expected_conditions = set(CONDITION_ORDER)
    found_conditions = {row["condition"] for row in rows}
    if found_conditions != expected_conditions:
        raise RuntimeError(f"Condition mismatch: {sorted(found_conditions)}")
    if len(rows) != 16200:
        raise RuntimeError(f"Expected 16200 samples, found {len(rows)}")
    for row in rows:
        row["separation_bin"] = separation_bin(float(row["angular_separation_deg"]))

    by_condition = grouped_summary(rows, "condition", CONDITION_ORDER)
    by_scene = grouped_summary(rows, "scene")
    by_distance = grouped_summary(rows, "distance_m", sorted({str(row["distance_m"]) for row in rows}, key=float))
    by_room = grouped_summary(rows, "room_id")
    by_subject = grouped_summary(rows, "subject_id")
    by_separation = grouped_summary(rows, "separation_bin", ("20_40", "45_80", "85_160"))
    paired = paired_reference_summary(rows)

    sample_fields = [
        "file_id", "paired_test_key", "condition", "true_deg", "pred_deg",
        "true_bin", "pred_bin", "error_deg", "correct", "acc_at_5", "acc_at_10",
        "large_err", "snr", "diffuse_snr_db", "rt60_s", "distance_m",
        "subject_id", "room_id", "scene", "angular_separation_deg", "separation_bin",
    ]
    sample_rows = []
    for row in rows:
        copied = dict(row)
        copied["acc_at_5"] = int(float(row["error_deg"]) <= 5.0)
        copied["acc_at_10"] = int(float(row["error_deg"]) <= 10.0)
        sample_rows.append({field: copied.get(field, "") for field in sample_fields})
    write_csv(output_dir / "per_sample.csv", sample_fields, sample_rows)
    for filename, field, values in (
        ("by_condition.csv", "condition", by_condition),
        ("by_scene.csv", "scene", by_scene),
        ("by_distance.csv", "distance_m", by_distance),
        ("by_room.csv", "room_id", by_room),
        ("by_subject.csv", "subject_id", by_subject),
        ("by_angular_separation.csv", "separation_bin", by_separation),
    ):
        write_csv(output_dir / filename, [field, *METRIC_FIELDS], values)
    write_csv(
        output_dir / "paired_vs_reference.csv",
        [
            "condition", "count", "reference_mae", "condition_mae",
            "delta_mae_vs_reference", "improved_fraction", "tied_fraction", "worsened_fraction",
        ],
        paired,
    )
    summary = {
        "count": len(rows),
        "overall": summarize(rows),
        "by_condition": by_condition,
        "paired_vs_reference": paired,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config": str(Path(args.config).resolve()),
        "test_root": str(Path(args.test_root).resolve()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved {output_dir}", flush=True)


if __name__ == "__main__":
    main()
