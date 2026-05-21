#!/usr/bin/env python3
"""对测试集样本级预测结果做 robustness 分析并生成图表。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def load_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wrap_deg(angle: float) -> float:
    return ((angle + 180.0) % 360.0) - 180.0


def region_of_angle(angle_deg: float) -> str:
    a = abs(wrap_deg(float(angle_deg)))
    if a <= 60.0:
        return "front"
    if a >= 120.0:
        return "back"
    return "side"


def merge_rows(pred_rows: List[dict], mixing_rows: List[dict]) -> List[dict]:
    mix_by_id = {row["file_id"]: row for row in mixing_rows}
    merged = []
    for row in pred_rows:
        file_id = row["file_id"]
        if file_id not in mix_by_id:
            continue
        mix = mix_by_id[file_id]
        true_label = int(row["true_label"])
        pred_label = int(row["pred_label"])
        ang_err = float(row["angular_error_deg"])
        merged.append({
            **row,
            **mix,
            "region": region_of_angle(float(row["true_deg"])),
            "angular_error_deg": ang_err,
            "front_back_halfplane_error": int(row["front_back_halfplane_error"]),
            "opposite_error": int(row["opposite_error"]),
            "large_error": int(row["large_error"]),
            "accuracy": int(true_label == pred_label),
            "acc_at_5deg": ang_err <= 5.0,
            "acc_at_10deg": ang_err <= 10.0,
            "snr_db": float(mix["snr_db"]),
            "rt60_s": float(mix["rt60_s"]),
        })
    return merged


def summarize_group(rows: List[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {
            "count": 0,
            "accuracy": None,
            "mae": None,
            "acc_at_5deg": None,
            "acc_at_10deg": None,
            "front_back_halfplane_error_rate": None,
            "opposite_error_rate": None,
            "large_error_rate": None,
        }
    return {
        "count": n,
        "accuracy": float(np.mean([r["accuracy"] for r in rows])),
        "mae": float(np.mean([r["angular_error_deg"] for r in rows])),
        "acc_at_5deg": float(np.mean([r["acc_at_5deg"] for r in rows])),
        "acc_at_10deg": float(np.mean([r["acc_at_10deg"] for r in rows])),
        "front_back_halfplane_error_rate": float(np.mean([r["front_back_halfplane_error"] for r in rows])),
        "opposite_error_rate": float(np.mean([r["opposite_error"] for r in rows])),
        "large_error_rate": float(np.mean([r["large_error"] for r in rows])),
    }


def bucketize(rows: List[dict], key_fn: Callable[[dict], str], order_fn: Callable[[Tuple[str, List[dict]]], object] | None = None) -> Dict[str, List[dict]]:
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        buckets[key_fn(row)].append(row)
    items = list(buckets.items())
    if order_fn is None:
        items.sort(key=lambda kv: kv[0])
    else:
        items.sort(key=order_fn)
    return dict(items)


def snr_bucket(snr: float) -> str:
    edges = [-10, -5, 0, 5, 10]
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= snr < hi:
            return f"[{lo},{hi})"
    return "[5,10]" if snr >= 5 else "[-10,-5)"


def rt60_bucket(rt60: float) -> str:
    edges = [0.20, 0.35, 0.50, 0.65, 0.80]
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= rt60 < hi:
            return f"[{lo:.2f},{hi:.2f})"
    return f"[{edges[-2]:.2f},{edges[-1]:.2f}]"


def snr_order(item: Tuple[str, List[dict]]) -> float:
    label = item[0]
    return float(label.split(",")[0].strip("["))


def rt60_order(item: Tuple[str, List[dict]]) -> float:
    label = item[0]
    return float(label.split(",")[0].strip("["))


def write_summary_csv(path: Path, grouped: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "count",
        "accuracy",
        "mae",
        "acc_at_5deg",
        "acc_at_10deg",
        "front_back_halfplane_error_rate",
        "opposite_error_rate",
        "large_error_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, stats in grouped.items():
            writer.writerow({"group": key, **stats})


def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "legend.frameon": False,
        "grid.alpha": 0.25,
        "lines.linewidth": 2.2,
        "lines.markersize": 6,
    })


def pretty_metric(metric: str) -> str:
    names = {
        "accuracy": "Accuracy",
        "mae": "MAE (deg)",
        "acc_at_5deg": "Acc@5°",
        "acc_at_10deg": "Acc@10°",
    }
    return names.get(metric, metric)


def plot_line(grouped: Dict[str, dict], metric: str, title: str, out_path: Path) -> None:
    labels = list(grouped.keys())
    values = [grouped[k][metric] for k in labels]
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(labels, values, marker="o", color=COLORS[0])
    ax.set_title(title)
    ax.set_ylabel(pretty_metric(metric))
    ax.set_xlabel("")
    if metric != "mae":
        ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_bar(grouped: Dict[str, dict], metric: str, title: str, out_path: Path) -> None:
    labels = list(grouped.keys())
    values = [grouped[k][metric] for k in labels]
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    bars = ax.bar(labels, values, color=[COLORS[0], COLORS[2], COLORS[1]])
    ax.set_title(title)
    ax.set_ylabel(pretty_metric(metric))
    if metric != "mae":
        ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y")
    for bar, value in zip(bars, values):
        if value is not None:
            label = f"{value:.2f}" if metric == "mae" else f"{value:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), label, ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_lines(model_summaries: Dict[str, Dict[str, dict]], metric: str, title: str, out_path: Path) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for idx, (label, grouped) in enumerate(model_summaries.items()):
        xlabels = list(grouped.keys())
        values = [grouped[k][metric] for k in xlabels]
        ax.plot(xlabels, values, marker="o", label=label, color=COLORS[idx % len(COLORS)])
    ax.set_title(title)
    ax.set_ylabel(pretty_metric(metric))
    if metric != "mae":
        ax.set_ylim(0.0, 1.0)
    ax.legend(ncol=1, loc="best")
    ax.grid(True, axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_analyses(merged: List[dict]) -> Dict[str, Dict[str, List[dict]]]:
    return {
        "snr": bucketize(merged, lambda r: snr_bucket(r["snr_db"]), order_fn=snr_order),
        "rt60": bucketize(merged, lambda r: rt60_bucket(r["rt60_s"]), order_fn=rt60_order),
        "region": bucketize(merged, lambda r: r["region"], order_fn=lambda kv: {"front": 0, "side": 1, "back": 2}[kv[0]]),
        "scene": bucketize(merged, lambda r: str(r["demand_scene"])),
        "subject": bucketize(merged, lambda r: str(r["subject_id"])),
    }


def summarize_analyses(analyses: Dict[str, Dict[str, List[dict]]]) -> Dict[str, Dict[str, dict]]:
    return {name: {k: summarize_group(v) for k, v in buckets.items()} for name, buckets in analyses.items()}


def run_single(predictions: Path, mixing_report: Path, out_dir: Path) -> None:
    pred_rows = load_csv(predictions)
    mixing_rows = load_csv(mixing_report)
    merged = merge_rows(pred_rows, mixing_rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall = summarize_group(merged)
    (out_dir / "overall_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    analyses = build_analyses(merged)
    summaries = summarize_analyses(analyses)
    for name, summary in summaries.items():
        write_summary_csv(out_dir / f"{name}_summary.csv", summary)

    plot_line(summaries["snr"], "mae", "MAE vs SNR", out_dir / "mae_vs_snr.png")
    plot_line(summaries["snr"], "accuracy", "Accuracy vs SNR", out_dir / "acc_vs_snr.png")
    plot_line(summaries["snr"], "acc_at_5deg", "Acc@5° vs SNR", out_dir / "acc5_vs_snr.png")
    plot_line(summaries["snr"], "acc_at_10deg", "Acc@10° vs SNR", out_dir / "acc10_vs_snr.png")
    plot_line(summaries["rt60"], "mae", "MAE vs RT60", out_dir / "mae_vs_rt60.png")
    plot_line(summaries["rt60"], "accuracy", "Accuracy vs RT60", out_dir / "acc_vs_rt60.png")
    plot_line(summaries["rt60"], "acc_at_5deg", "Acc@5° vs RT60", out_dir / "acc5_vs_rt60.png")
    plot_line(summaries["rt60"], "acc_at_10deg", "Acc@10° vs RT60", out_dir / "acc10_vs_rt60.png")
    plot_bar(summaries["region"], "mae", "MAE by Region", out_dir / "mae_by_region.png")
    plot_bar(summaries["region"], "acc_at_10deg", "Acc@10° by Region", out_dir / "acc10_by_region.png")


def parse_compare_entries(entries: List[str]) -> Dict[str, Path]:
    parsed: Dict[str, Path] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid compare entry: {entry}. Use label=path/to/predictions.csv")
        label, path = entry.split("=", 1)
        parsed[label] = Path(path)
    return parsed


def run_compare(compare_entries: Dict[str, Path], mixing_report: Path, out_dir: Path) -> None:
    mixing_rows = load_csv(mixing_report)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_summaries_by_view: Dict[str, Dict[str, Dict[str, dict]]] = {"snr": {}, "rt60": {}}
    overall = {}
    for label, pred_path in compare_entries.items():
        merged = merge_rows(load_csv(pred_path), mixing_rows)
        overall[label] = summarize_group(merged)
        analyses = build_analyses(merged)
        summaries = summarize_analyses(analyses)
        model_summaries_by_view["snr"][label] = summaries["snr"]
        model_summaries_by_view["rt60"][label] = summaries["rt60"]

    (out_dir / "overall_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    plot_comparison_lines(model_summaries_by_view["snr"], "mae", "MAE vs SNR", out_dir / "compare_mae_vs_snr.png")
    plot_comparison_lines(model_summaries_by_view["snr"], "accuracy", "Accuracy vs SNR", out_dir / "compare_acc_vs_snr.png")
    plot_comparison_lines(model_summaries_by_view["snr"], "acc_at_5deg", "Acc@5° vs SNR", out_dir / "compare_acc5_vs_snr.png")
    plot_comparison_lines(model_summaries_by_view["snr"], "acc_at_10deg", "Acc@10° vs SNR", out_dir / "compare_acc10_vs_snr.png")
    plot_comparison_lines(model_summaries_by_view["rt60"], "mae", "MAE vs RT60", out_dir / "compare_mae_vs_rt60.png")
    plot_comparison_lines(model_summaries_by_view["rt60"], "accuracy", "Accuracy vs RT60", out_dir / "compare_acc_vs_rt60.png")
    plot_comparison_lines(model_summaries_by_view["rt60"], "acc_at_5deg", "Acc@5° vs RT60", out_dir / "compare_acc5_vs_rt60.png")
    plot_comparison_lines(model_summaries_by_view["rt60"], "acc_at_10deg", "Acc@10° vs RT60", out_dir / "compare_acc10_vs_rt60.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze robustness from predictions.csv and mixing_report.csv")
    parser.add_argument("--predictions", type=str, default="")
    parser.add_argument("--compare", type=str, nargs="*", default=[])
    parser.add_argument("--mixing-report", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    mixing_report = Path(args.mixing_report)

    if args.compare:
        run_compare(parse_compare_entries(args.compare), mixing_report, out_dir)
        return

    if not args.predictions:
        raise ValueError("Either --predictions or --compare must be provided.")
    run_single(Path(args.predictions), mixing_report, out_dir)


if __name__ == "__main__":
    main()
