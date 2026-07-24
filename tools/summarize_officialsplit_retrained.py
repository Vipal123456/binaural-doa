#!/usr/bin/env python3
"""Summarize official-split retrained KEMAR evaluations for paper analysis."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "outputs" / "grouped_eval_runs_officialsplit_retrained"
OUT_ROOT = ROOT / "outputs" / "paper_officialsplit_analysis_final"

METRICS = (
    "accuracy",
    "mae",
    "median",
    "acc_at_5",
    "acc_at_10",
    "fb_err",
    "opp_err",
    "large_err",
)
SNR_ORDER = ("10", "5", "0", "-5", "-10")

RUNS = (
    ("main", 42, "main_seed42", 152803),
    ("main", 43, "main_seed43", 152803),
    ("main", 44, "main_seed44", 152803),
    ("rawconcat_controlled", 42, "ablation_rawconcat_controlled_officialsplit_seed42", 147683),
    ("rawconcat_controlled", 43, "ablation_rawconcat_controlled_officialsplit_seed43", 147683),
    ("rawconcat_controlled", 44, "ablation_rawconcat_controlled_officialsplit_seed44", 147683),
    ("sdel", 43, "sdel_officialsplit_seed43", 925834),
    ("sdel", 44, "sdel_officialsplit_seed44", 925834),
    ("sdel", 45, "sdel_officialsplit_seed45", 925834),
    ("dprtf", 42, "dprtf_officialsplit_seed42", 876608),
    ("dprtf", 43, "dprtf_officialsplit_seed43", 876608),
    ("dprtf", 44, "dprtf_officialsplit_seed44", 876608),
    ("bil", 43, "bil_officialsplit_seed43", 865228),
    ("fnssl", 43, "fnssl_officialsplit_seed43", 658890),
)

ABLATIONS_SEED42 = (
    ("main", "main_seed42", 152803),
    ("w/o reliability", "ablation_norel_officialsplit_seed42", 147123),
    ("w/o content", "ablation_nocontent_officialsplit_seed42", 92067),
    ("merged cue", "ablation_mergedcue_officialsplit_seed42", 154459),
    ("raw concat controlled", "ablation_rawconcat_controlled_officialsplit_seed42", 147683),
)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def load_runs() -> tuple[list[dict], list[str]]:
    rows = []
    missing = []
    for model, seed, run, parameters in RUNS:
        result_path = EVAL_ROOT / run / "overall.json"
        snr_path = EVAL_ROOT / run / "by_snr.csv"
        if not result_path.is_file() or not snr_path.is_file():
            missing.append(run)
            continue
        result = read_json(result_path)
        rows.append(
            {
                "model": model,
                "seed": seed,
                "run": run,
                "parameters": parameters,
                "count": int(result["count"]),
                **{metric: float(result[metric]) for metric in METRICS},
            }
        )
    return rows, missing


def build_group_overall(run_rows: list[dict]) -> list[dict]:
    output = []
    for model in sorted({row["model"] for row in run_rows}):
        group = [row for row in run_rows if row["model"] == model]
        item = {
            "model": model,
            "n": len(group),
            "parameters": group[0]["parameters"],
            "count_per_run": group[0]["count"],
        }
        for metric in METRICS:
            values = [row[metric] for row in group]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = sample_std(values)
        output.append(item)
    return sorted(output, key=lambda row: (-row["accuracy_mean"], row["mae_mean"]))


def build_by_snr(run_rows: list[dict]) -> list[dict]:
    output = []
    for run_row in run_rows:
        by_snr = {
            row["snr"]: row
            for row in read_csv(EVAL_ROOT / run_row["run"] / "by_snr.csv")
        }
        for snr in SNR_ORDER:
            item = by_snr[snr]
            output.append(
                {
                    "model": run_row["model"],
                    "seed": run_row["seed"],
                    "run": run_row["run"],
                    "snr": snr,
                    "count": int(item["count"]),
                    **{metric: float(item[metric]) for metric in METRICS},
                }
            )
    return output


def build_group_by_snr(by_snr_rows: list[dict]) -> list[dict]:
    output = []
    for model in sorted({row["model"] for row in by_snr_rows}):
        for snr in SNR_ORDER:
            group = [
                row
                for row in by_snr_rows
                if row["model"] == model and row["snr"] == snr
            ]
            item = {"model": model, "snr": snr, "n": len(group), "count_per_run": group[0]["count"]}
            for metric in METRICS:
                values = [row[metric] for row in group]
                item[f"{metric}_mean"] = mean(values)
                item[f"{metric}_std"] = sample_std(values)
            output.append(item)
    return output


def build_low_snr(by_snr_rows: list[dict]) -> list[dict]:
    output = []
    keys = {(row["model"], row["seed"], row["run"]) for row in by_snr_rows}
    for model, seed, run in sorted(keys):
        group = [
            row
            for row in by_snr_rows
            if row["model"] == model and row["seed"] == seed and row["snr"] in {"-5", "-10"}
        ]
        count = sum(row["count"] for row in group)
        item = {"model": model, "seed": seed, "run": run, "count": count}
        for metric in METRICS:
            item[metric] = sum(row[metric] * row["count"] for row in group) / count
        output.append(item)
    return sorted(output, key=lambda row: (-row["accuracy"], row["mae"]))


def build_filtered_overall(by_snr_rows: list[dict]) -> list[dict]:
    """Combine the five paper SNRs, excluding clean and -15 dB."""
    output = []
    keys = {(row["model"], row["seed"], row["run"]) for row in by_snr_rows}
    parameters = {(model, seed, run): params for model, seed, run, params in RUNS}
    for model, seed, run in sorted(keys):
        group = [
            row
            for row in by_snr_rows
            if row["model"] == model and row["seed"] == seed and row["snr"] in SNR_ORDER
        ]
        count = sum(row["count"] for row in group)
        item = {
            "model": model,
            "seed": seed,
            "run": run,
            "parameters": parameters[(model, seed, run)],
            "count": count,
        }
        for metric in METRICS:
            item[metric] = sum(row[metric] * row["count"] for row in group) / count
        output.append(item)
    return sorted(output, key=lambda row: (-row["accuracy"], row["mae"]))


def build_low_snr_groups(low_snr_rows: list[dict]) -> list[dict]:
    output = []
    for model in sorted({row["model"] for row in low_snr_rows}):
        group = [row for row in low_snr_rows if row["model"] == model]
        item = {"model": model, "n": len(group), "count_per_run": group[0]["count"]}
        for metric in METRICS:
            values = [row[metric] for row in group]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = sample_std(values)
        output.append(item)
    return sorted(output, key=lambda row: (-row["accuracy_mean"], row["mae_mean"]))


def build_ablation() -> list[dict]:
    rows = []
    main = read_json(EVAL_ROOT / "main_seed42" / "overall.json")
    for model, run, parameters in ABLATIONS_SEED42:
        result = read_json(EVAL_ROOT / run / "overall.json")
        rows.append(
            {
                "model": model,
                "seed": 42,
                "parameters": parameters,
                "count": int(result["count"]),
                **{metric: float(result[metric]) for metric in METRICS},
                "delta_accuracy_pp_vs_main": 100.0 * (float(result["accuracy"]) - float(main["accuracy"])),
                "delta_mae_deg_vs_main": float(result["mae"]) - float(main["mae"]),
            }
        )
    return rows


def build_filtered_ablation() -> list[dict]:
    rows = []
    combined = {}
    for model, run, parameters in ABLATIONS_SEED42:
        by_snr = {
            row["snr"]: row for row in read_csv(EVAL_ROOT / run / "by_snr.csv")
        }
        count = sum(int(by_snr[snr]["count"]) for snr in SNR_ORDER)
        item = {
            "model": model,
            "seed": 42,
            "parameters": parameters,
            "count": count,
        }
        for metric in METRICS:
            item[metric] = sum(
                float(by_snr[snr][metric]) * int(by_snr[snr]["count"])
                for snr in SNR_ORDER
            ) / count
        combined[model] = item

    main = combined["main"]
    for model, _, _ in ABLATIONS_SEED42:
        item = combined[model]
        item["delta_accuracy_pp_vs_main"] = 100.0 * (item["accuracy"] - main["accuracy"])
        item["delta_mae_deg_vs_main"] = item["mae"] - main["mae"]
        rows.append(item)
    return rows


def build_raw_vs_main(run_rows: list[dict]) -> list[dict]:
    output = []
    for seed in (42, 43, 44):
        main = next(row for row in run_rows if row["model"] == "main" and row["seed"] == seed)
        raw = next(
            row
            for row in run_rows
            if row["model"] == "rawconcat_controlled" and row["seed"] == seed
        )
        output.append(
            {
                "seed": seed,
                "main_accuracy": main["accuracy"],
                "raw_accuracy": raw["accuracy"],
                "delta_accuracy_pp_raw_minus_main": 100.0 * (raw["accuracy"] - main["accuracy"]),
                "main_mae": main["mae"],
                "raw_mae": raw["mae"],
                "delta_mae_deg_raw_minus_main": raw["mae"] - main["mae"],
            }
        )
    return output


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_rows, missing = load_runs()
    if missing:
        raise FileNotFoundError("Missing evaluation outputs: " + ", ".join(missing))

    run_rows = sorted(run_rows, key=lambda row: (-row["accuracy"], row["mae"]))
    group_rows = build_group_overall(run_rows)
    fixed_seed43 = sorted(
        [row for row in run_rows if row["seed"] == 43],
        key=lambda row: (-row["accuracy"], row["mae"]),
    )
    by_snr_rows = build_by_snr(run_rows)
    group_by_snr_rows = build_group_by_snr(by_snr_rows)
    fixed_seed43_by_snr = [row for row in by_snr_rows if row["seed"] == 43]
    filtered_overall = build_filtered_overall(by_snr_rows)
    filtered_group_overall = build_group_overall(filtered_overall)
    filtered_fixed_seed43 = sorted(
        [row for row in filtered_overall if row["seed"] == 43],
        key=lambda row: (-row["accuracy"], row["mae"]),
    )
    low_snr_rows = build_low_snr(by_snr_rows)
    low_snr_group_rows = build_low_snr_groups(low_snr_rows)
    ablation_rows = build_ablation()
    filtered_ablation_rows = build_filtered_ablation()
    raw_vs_main_rows = build_raw_vs_main(run_rows)

    run_fields = ["model", "seed", "run", "parameters", "count", *METRICS]
    group_fields = ["model", "n", "parameters", "count_per_run"] + [
        name for metric in METRICS for name in (f"{metric}_mean", f"{metric}_std")
    ]
    by_snr_fields = ["model", "seed", "run", "snr", "count", *METRICS]
    group_by_snr_fields = ["model", "snr", "n", "count_per_run"] + [
        name for metric in METRICS for name in (f"{metric}_mean", f"{metric}_std")
    ]
    low_snr_fields = ["model", "seed", "run", "count", *METRICS]
    low_snr_group_fields = ["model", "n", "count_per_run"] + [
        name for metric in METRICS for name in (f"{metric}_mean", f"{metric}_std")
    ]

    write_csv(OUT_ROOT / "single_run_overall_ranking.csv", run_rows, run_fields)
    write_csv(OUT_ROOT / "group_overall_mean_std.csv", group_rows, group_fields)
    write_csv(OUT_ROOT / "fixed_seed43_overall.csv", fixed_seed43, run_fields)
    write_csv(OUT_ROOT / "single_run_noisy_only_ranking.csv", filtered_overall, run_fields)
    write_csv(OUT_ROOT / "group_noisy_only_mean_std.csv", filtered_group_overall, group_fields)
    write_csv(OUT_ROOT / "fixed_seed43_noisy_only.csv", filtered_fixed_seed43, run_fields)
    write_csv(OUT_ROOT / "all_runs_by_snr_no_clean_no_m15.csv", by_snr_rows, by_snr_fields)
    write_csv(OUT_ROOT / "group_by_snr_mean_std_no_clean_no_m15.csv", group_by_snr_rows, group_by_snr_fields)
    write_csv(OUT_ROOT / "fixed_seed43_by_snr_no_clean_no_m15.csv", fixed_seed43_by_snr, by_snr_fields)
    write_csv(OUT_ROOT / "low_snr_m5_m10_single_runs.csv", low_snr_rows, low_snr_fields)
    write_csv(OUT_ROOT / "low_snr_m5_m10_group_mean_std.csv", low_snr_group_rows, low_snr_group_fields)
    write_csv(
        OUT_ROOT / "ablation_seed42.csv",
        ablation_rows,
        ["model", "seed", "parameters", "count", *METRICS, "delta_accuracy_pp_vs_main", "delta_mae_deg_vs_main"],
    )
    write_csv(
        OUT_ROOT / "ablation_seed42_noisy_only.csv",
        filtered_ablation_rows,
        ["model", "seed", "parameters", "count", *METRICS, "delta_accuracy_pp_vs_main", "delta_mae_deg_vs_main"],
    )
    write_csv(
        OUT_ROOT / "rawconcat_vs_main_same_seed.csv",
        raw_vs_main_rows,
        list(raw_vs_main_rows[0].keys()),
    )

    print(f"Saved {OUT_ROOT}")
    for path in sorted(OUT_ROOT.glob("*.csv")):
        print(path)


if __name__ == "__main__":
    main()
