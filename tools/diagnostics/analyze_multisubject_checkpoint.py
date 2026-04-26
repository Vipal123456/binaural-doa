#!/usr/bin/env python3
"""Detailed error analysis for robust50h multisubject checkpoints.

This script evaluates a checkpoint on the robust50h unseen-subject test split,
rebuilds a per-segment error table, and writes plots/summaries by:

- subject
- room profile
- scene
- SNR
- RT60
- true angle

It can also compare against an existing baseline per-segment CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.static_dataset import build_static_datasets
from models.binaural_doa_net import build_model
from utils.angle import angular_error, bins_to_angles
from utils.checkpoint import load_checkpoint
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze robust50h multisubject checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--config", required=True, help="Config path")
    parser.add_argument("--output_dir", required=True, help="Directory for analysis outputs")
    parser.add_argument(
        "--reference_csv",
        default="outputs/analysis_multisubject_robust50h_v5_test_best/per_segment_errors.csv",
        help="Reference per-segment CSV used to recover environment metadata",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Override eval batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Eval dataloader workers")
    return parser.parse_args()


def load_reference_rows(path: Path) -> Dict[Tuple[str, str], dict]:
    rows: Dict[Tuple[str, str], dict] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["file_id"], row["start_sec"])
            rows[key] = row
    return rows


@torch.no_grad()
def run_eval(cfg, checkpoint_path: Path, reference_rows: Dict[Tuple[str, str], dict], batch_size: int, num_workers: int):
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    _, _, test_ds = build_static_datasets(cfg)
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows = []
    global_idx = 0
    num_classes = cfg.model.num_classes
    azimuth_range = tuple(cfg.model.azimuth_range)

    for batch in loader:
        batch_size_now = len(batch["azimuth_label"])
        batch_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        out = model(batch_on_device)
        logits = out["logits"].float().cpu()
        probs = F.softmax(logits, dim=-1)
        pred_bins = probs.argmax(dim=-1).numpy()
        top3 = torch.topk(probs, k=min(3, probs.shape[-1]), dim=-1).indices.cpu().numpy()

        true_deg = batch["azimuth_deg"]
        if isinstance(true_deg, torch.Tensor):
            true_deg = true_deg.cpu().numpy()
        else:
            true_deg = np.asarray(true_deg, dtype=np.float32)

        true_bins = batch["azimuth_label"]
        if isinstance(true_bins, torch.Tensor):
            true_bins = true_bins.cpu().numpy()
        else:
            true_bins = np.asarray(true_bins)

        pred_deg = bins_to_angles(pred_bins, num_classes, azimuth_range)
        err_deg = angular_error(pred_deg, true_deg)

        for local_idx in range(batch_size_now):
            seg = test_ds.segments[global_idx + local_idx]
            file_id = Path(seg["audio_path"]).stem.replace("binaural", "")
            start_sec = f"{float(seg['start_sec']):.1f}"
            ref = reference_rows.get((file_id, start_sec))
            if ref is None:
                raise KeyError(f"Missing reference metadata for {(file_id, start_sec)}")

            rows.append({
                "file_id": file_id,
                "subject_id": ref["subject_id"],
                "start_sec": start_sec,
                "true_deg": float(true_deg[local_idx]),
                "pred_deg": float(pred_deg[local_idx]),
                "error_deg": float(err_deg[local_idx]),
                "correct": int(pred_bins[local_idx] == true_bins[local_idx]),
                "top3_hit": int(int(true_bins[local_idx]) in set(top3[local_idx].tolist())),
                "room_profile": ref["room_profile"],
                "scene": ref["scene"],
                "snr_db": float(ref["snr_db"]),
                "rt60_s": float(ref["rt60_s"]),
                "source_distance_m": float(ref["source_distance_m"]),
                "target_azimuth_deg": float(ref["target_azimuth_deg"]),
            })

        global_idx += batch_size_now

    return rows


def metric_summary(rows: List[dict]) -> dict:
    if not rows:
        return {}
    errors = np.array([r["error_deg"] for r in rows], dtype=np.float32)
    return {
        "count": int(len(rows)),
        "mae": float(errors.mean()),
        "median": float(np.median(errors)),
        "acc": float(np.mean([r["correct"] for r in rows])),
        "top3_acc": float(np.mean([r["top3_hit"] for r in rows])),
        "lt5": float(np.mean(errors < 5.0)),
        "lt10": float(np.mean(errors < 10.0)),
        "lt20": float(np.mean(errors < 20.0)),
        "lt30": float(np.mean(errors < 30.0)),
        "ge90": float(np.mean(errors >= 90.0)),
        "ge150": float(np.mean(errors >= 150.0)),
        "std": float(errors.std()),
    }


def group_rows(rows: List[dict], key_fn) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(key_fn(row))].append(row)
    return dict(grouped)


def bin_numeric(rows: List[dict], field: str, edges: Iterable[float]) -> Dict[str, List[dict]]:
    edges = list(edges)
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        value = float(row[field])
        label = None
        for lo, hi in zip(edges[:-1], edges[1:]):
            if lo <= value < hi or (value == edges[-1] and hi == edges[-1]):
                label = f"[{lo:.2f}, {hi:.2f})"
                break
        if label is None:
            label = f"out_of_range({value:.2f})"
        grouped[label].append(row)
    return dict(grouped)


def angle_bin_label(angle: float) -> str:
    start = int(math.floor((angle + 180.0) / 10.0) * 10 - 180)
    end = start + 10
    return f"[{start}, {end})"


def summarize_groups(grouped: Dict[str, List[dict]]) -> Dict[str, dict]:
    return {key: metric_summary(value) for key, value in grouped.items()}


def write_csv(rows: List[dict], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sorted_group_items(group_summary: Dict[str, dict], sort_by: str = "mae") -> List[Tuple[str, dict]]:
    return sorted(group_summary.items(), key=lambda kv: kv[1][sort_by], reverse=True)


def plot_group_metric(group_summary: Dict[str, dict], output_path: Path, title: str, metric: str, xlabel: str, topn: int | None = None) -> None:
    items = sorted_group_items(group_summary, sort_by=metric)
    if topn is not None:
        items = items[:topn]
    labels = [k for k, _ in items]
    values = [v[metric] for _, v in items]

    fig_w = max(10, min(20, 0.55 * len(labels)))
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    ax.bar(range(len(labels)), values, color="#2a6f97")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_group_curve(group_summary: Dict[str, dict], output_path: Path, title: str, x_values: List[float], labels: List[str], metrics: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#d62828", "#1d3557", "#2a9d8f", "#6a4c93"]
    for metric, color in zip(metrics, colors):
        y = [group_summary[label][metric] for label in labels]
        ax.plot(x_values, y, marker="o", linewidth=2, label=metric, color=color)
    ax.set_title(title)
    ax.set_xlabel("Bin center")
    ax.set_ylabel("Metric")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_angle_mae(group_summary: Dict[str, dict], output_path: Path) -> None:
    labels = sorted(group_summary.keys(), key=lambda s: int(s.split(",")[0][1:]))
    centers = []
    maes = []
    accs = []
    for label in labels:
        start, end = label.strip("[]").strip(")").split(",")
        start = float(start)
        end = float(end)
        centers.append((start + end) / 2.0)
        maes.append(group_summary[label]["mae"])
        accs.append(group_summary[label]["acc"])

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.plot(centers, maes, color="#d62828", marker="o", linewidth=2, label="MAE")
    ax1.set_xlabel("True angle bin center (deg)")
    ax1.set_ylabel("MAE (deg)", color="#d62828")
    ax1.tick_params(axis="y", labelcolor="#d62828")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(centers, accs, color="#1d3557", marker="s", linewidth=2, label="Accuracy")
    ax2.set_ylabel("Accuracy", color="#1d3557")
    ax2.tick_params(axis="y", labelcolor="#1d3557")

    fig.suptitle("Error by True Angle")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_compare_summary(reference_rows: Dict[Tuple[str, str], dict], enhanced_rows: List[dict]) -> dict:
    baseline_rows = []
    for ref in reference_rows.values():
        baseline_rows.append({
            "error_deg": float(ref["error_deg"]),
            "correct": int(ref["correct"]),
            "top3_hit": int(ref["top3_hit"]),
            "subject_id": ref["subject_id"],
            "room_profile": ref["room_profile"],
            "scene": ref["scene"],
            "snr_db": float(ref["snr_db"]),
            "rt60_s": float(ref["rt60_s"]),
            "target_azimuth_deg": float(ref["target_azimuth_deg"]),
        })

    baseline = metric_summary(baseline_rows)
    enhanced = metric_summary(enhanced_rows)
    compare = {}
    for key in ["acc", "top3_acc", "mae", "median", "lt5", "lt10", "lt20", "lt30", "ge90", "ge150", "std"]:
        compare[key] = {
            "baseline": baseline[key],
            "enhanced": enhanced[key],
            "delta": enhanced[key] - baseline[key],
        }
    return compare


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, [])
    cfg.train.num_workers = args.num_workers
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_rows = load_reference_rows(Path(args.reference_csv))
    rows = run_eval(
        cfg=cfg,
        checkpoint_path=Path(args.checkpoint),
        reference_rows=reference_rows,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
    )

    per_segment_csv = output_dir / "per_segment_errors.csv"
    write_csv(rows, per_segment_csv)

    by_subject = summarize_groups(group_rows(rows, lambda r: r["subject_id"]))
    by_room = summarize_groups(group_rows(rows, lambda r: r["room_profile"]))
    by_scene = summarize_groups(group_rows(rows, lambda r: r["scene"]))
    by_snr = summarize_groups(bin_numeric(rows, "snr_db", [-10, -5, 0, 5, 10.0001]))
    by_rt60 = summarize_groups(bin_numeric(rows, "rt60_s", [0.2, 0.35, 0.5, 0.65, 0.8001]))
    by_angle = summarize_groups(group_rows(rows, lambda r: angle_bin_label(float(r["target_azimuth_deg"]))))

    summary = {
        "overall": metric_summary(rows),
        "by_subject": by_subject,
        "by_room_profile": by_room,
        "by_scene": by_scene,
        "by_snr_bin": by_snr,
        "by_rt60_bin": by_rt60,
        "by_true_angle_bin": by_angle,
        "reference_csv": args.reference_csv,
        "checkpoint": args.checkpoint,
        "config": args.config,
    }
    save_json(summary, output_dir / "summary.json")
    save_json(build_compare_summary(reference_rows, rows), output_dir / "baseline_vs_enhanced_compare.json")

    plot_group_metric(by_subject, output_dir / "mae_by_subject.png", "MAE by Subject", "mae", "subject")
    plot_group_metric(by_room, output_dir / "mae_by_room_profile.png", "MAE by Room Profile", "mae", "room_profile")
    plot_group_metric(by_scene, output_dir / "mae_by_scene.png", "MAE by Scene", "mae", "scene")
    plot_angle_mae(by_angle, output_dir / "error_by_true_angle.png")

    snr_labels = list(by_snr.keys())
    snr_centers = [-7.5, -2.5, 2.5, 7.5]
    plot_group_curve(by_snr, output_dir / "metrics_by_snr.png", "Metrics by SNR", snr_centers[: len(snr_labels)], snr_labels, ["mae", "lt10"])

    rt60_labels = list(by_rt60.keys())
    rt60_centers = [0.275, 0.425, 0.575, 0.725]
    plot_group_curve(by_rt60, output_dir / "metrics_by_rt60.png", "Metrics by RT60", rt60_centers[: len(rt60_labels)], rt60_labels, ["mae", "lt10"])

    print(f"Saved {per_segment_csv}")
    print(f"Saved {output_dir / 'summary.json'}")
    print(f"Saved {output_dir / 'baseline_vs_enhanced_compare.json'}")


if __name__ == "__main__":
    main()
