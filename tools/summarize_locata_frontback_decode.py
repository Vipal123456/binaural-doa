#!/usr/bin/env python3
"""Summarize LOCATA mirror-pair/front-back decoding diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List


RUN_PATTERN = re.compile(r"(?:(dev)_)?main_seed(\d+)_(peak|rawlevel)")
METRICS = (
    "mae_deg",
    "median_ae_deg",
    "acc_at_5deg",
    "acc_at_10deg",
    "front_back_error_rate",
    "folded_lateral_mae_deg",
)


def _write_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default="outputs/locata_frontback_decode")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    per_run: List[Dict] = []
    for summary_path in sorted(result_root.glob("*/summary.json")):
        match = RUN_PATTERN.fullmatch(summary_path.parent.name)
        if match is None:
            continue
        dev_prefix, seed, preprocessing = match.groups()
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        split = "dev" if dev_prefix else "eval"
        heads = payload.get("front_back_heads", {})
        for decoder, decoder_payload in payload["decoders"].items():
            window = decoder_payload["window_micro"]
            recording = decoder_payload["recording_macro"]
            row = {
                "split": split,
                "seed": int(seed),
                "preprocessing": preprocessing,
                "decoder": decoder,
                "num_recordings": payload["protocol"]["recording_count"],
                "num_windows": payload["protocol"]["segment_count"],
                "class_fb_accuracy": heads["class_probability_mass"]["accuracy"],
                "aux_fb_accuracy": heads.get("auxiliary_head", {}).get("accuracy", ""),
            }
            row.update({metric: window[metric] for metric in METRICS})
            row["recording_macro_mae_deg"] = recording["recording_macro_mae_deg"]
            row["recording_macro_front_back_error_rate"] = recording[
                "recording_macro_front_back_error_rate"
            ]
            per_run.append(row)

    per_run.sort(
        key=lambda row: (
            row["split"],
            row["preprocessing"],
            row["decoder"],
            row["seed"],
        )
    )
    _write_csv(result_root / "comparison_per_run.csv", per_run)

    grouped: Dict[tuple, List[Dict]] = defaultdict(list)
    for row in per_run:
        grouped[(row["split"], row["preprocessing"], row["decoder"])].append(row)

    aggregate_rows: List[Dict] = []
    aggregate_metrics = (
        "class_fb_accuracy",
        "aux_fb_accuracy",
        *METRICS,
        "recording_macro_mae_deg",
        "recording_macro_front_back_error_rate",
    )
    for (split, preprocessing, decoder), rows in sorted(grouped.items()):
        aggregate = {
            "split": split,
            "preprocessing": preprocessing,
            "decoder": decoder,
            "num_seeds": len(rows),
            "seeds": ",".join(str(row["seed"]) for row in rows),
        }
        for metric in aggregate_metrics:
            values = [float(row[metric]) for row in rows if row[metric] != ""]
            aggregate[f"{metric}_mean"] = mean(values) if values else ""
            aggregate[f"{metric}_std"] = stdev(values) if len(values) > 1 else ""
        aggregate_rows.append(aggregate)
    _write_csv(result_root / "comparison_by_protocol_decoder.csv", aggregate_rows)

    print(f"Wrote {len(per_run)} per-run rows and {len(aggregate_rows)} aggregates")


if __name__ == "__main__":
    main()
