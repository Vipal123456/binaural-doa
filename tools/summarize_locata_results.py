#!/usr/bin/env python3
"""Aggregate per-checkpoint LOCATA Task 1 evaluation summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List


MODEL_ORDER = ("main", "sdel", "dprtf", "bil", "fnssl")
WINDOW_METRICS = (
    "mae_deg",
    "median_ae_deg",
    "acc_at_5deg",
    "acc_at_10deg",
    "acc_at_20deg",
    "coverage_at_30deg",
    "gross_error_rate_gt_30deg",
    "conditional_mae_at_30deg",
    "front_back_error_rate",
    "folded_lateral_mae_deg",
    "folded_lateral_median_ae_deg",
)


def _write_csv(path: Path, rows: List[Dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default="outputs/locata_task1_dummy_eval")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    checkpoint_rows: List[Dict] = []
    for summary_path in sorted(result_root.glob("*/summary.json")):
        run_name = summary_path.parent.name
        match = re.fullmatch(r"(main|sdel|dprtf|bil|fnssl)_seed(\d+)_(bestacc|best)", run_name)
        if match is None:
            continue
        model_name, seed, selector = match.groups()
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {
            "model": model_name,
            "seed": int(seed),
            "checkpoint_selector": (
                "best_accuracy" if selector == "bestacc" else "legacy_best_mae"
            ),
            "checkpoint_epoch": payload.get("checkpoint_epoch"),
            "model_type": payload["model_type"],
            "recording_count": payload["protocol"]["recording_count"],
            "window_count": payload["protocol"]["segment_count"],
        }
        row.update(payload["window_micro"])
        row.update(payload["recording_macro"])
        checkpoint_rows.append(row)

    if not checkpoint_rows:
        raise ValueError(f"No LOCATA summaries found below {result_root}")
    checkpoint_rows.sort(key=lambda row: (MODEL_ORDER.index(row["model"]), row["seed"]))
    _write_csv(result_root / "comparison_per_checkpoint.csv", checkpoint_rows)

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in checkpoint_rows:
        grouped[row["model"]].append(row)

    model_rows: List[Dict] = []
    for model_name in MODEL_ORDER:
        rows = grouped.get(model_name, [])
        if not rows:
            continue
        selectors = sorted({row["checkpoint_selector"] for row in rows})
        aggregate = {
            "model": model_name,
            "num_seeds": len(rows),
            "seeds": ",".join(str(row["seed"]) for row in rows),
            "checkpoint_selector": ",".join(selectors),
        }
        for metric in WINDOW_METRICS:
            values = [float(row[metric]) for row in rows]
            aggregate[f"{metric}_mean"] = mean(values)
            aggregate[f"{metric}_std"] = stdev(values) if len(values) > 1 else ""
        recording_values = [float(row["recording_macro_mae_deg"]) for row in rows]
        aggregate["recording_macro_mae_deg_mean"] = mean(recording_values)
        aggregate["recording_macro_mae_deg_std"] = (
            stdev(recording_values) if len(recording_values) > 1 else ""
        )
        model_rows.append(aggregate)

    _write_csv(result_root / "comparison_by_model.csv", model_rows)
    payload = {
        "protocol": {
            "split": "LOCATA eval/task1/dummy",
            "channels": "mic 1/3",
            "sample_rate_hz": 16000,
            "segment_seconds": 2.0,
            "hop_seconds": 1.0,
            "minimum_vad_ratio": 0.5,
            "decoder": "argmax class angle",
        },
        "checkpoint_results": checkpoint_rows,
        "model_aggregates": model_rows,
        "comparability_warning": (
            "Main and FN-SSL use best-accuracy checkpoints. SDEL, DP-RTF, and BiL "
            "use legacy best.pth files selected by KEMAR validation MAE because "
            "best_acc.pth was not saved for those runs."
        ),
    }
    (result_root / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
