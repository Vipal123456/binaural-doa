#!/usr/bin/env python3
"""Analyze moving-DOA checkpoints, focusing on static-trajectory errors."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.moving_dataset import build_moving_datasets
from models.binaural_doa_net import build_model
from utils.checkpoint import load_checkpoint
from utils.config import load_config


def wrap_deg(angle: np.ndarray) -> np.ndarray:
    return ((angle + 180.0) % 360.0) - 180.0


def circular_error(pred_angle: np.ndarray, true_angle: np.ndarray) -> np.ndarray:
    diff = np.abs(wrap_deg(pred_angle - true_angle))
    return np.minimum(diff, 360.0 - diff)


def labels_to_angles(labels: np.ndarray, num_classes: int = 72, azimuth_range=(-180.0, 180.0)) -> np.ndarray:
    labels = np.asarray(labels)
    width = (azimuth_range[1] - azimuth_range[0]) / num_classes
    return azimuth_range[0] + (labels.astype(np.float64) + 0.5) * width


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze moving checkpoint static-trajectory subset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


def snr_bin(snr: float) -> str:
    edges = [-10.0, -5.0, 0.0, 5.0, 10.0001]
    labels = ["[-10,-5)", "[-5,0)", "[0,5)", "[5,10]"]
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if lo <= snr < hi or (i == len(labels) - 1 and snr <= 10.0):
            return labels[i]
    return f"out({snr:.2f})"


def rt60_bin(rt60: float) -> str:
    edges = [0.2, 0.35, 0.5, 0.65, 0.8001]
    labels = ["[0.20,0.35)", "[0.35,0.50)", "[0.50,0.65)", "[0.65,0.80]"]
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if lo <= rt60 < hi or (i == len(labels) - 1 and rt60 <= 0.8):
            return labels[i]
    return f"out({rt60:.3f})"


def angle_bin(angle: float) -> str:
    start = int(math.floor((float(angle) + 180.0) / 10.0) * 10 - 180)
    end = start + 10
    return f"[{start},{end})"


def front_back_label(angle: float) -> str:
    wrapped = ((float(angle) + 180.0) % 360.0) - 180.0
    return "front" if abs(wrapped) <= 90.0 else "back"


def metric_summary(rows: List[dict]) -> dict:
    if not rows:
        return {}
    errors = np.asarray([r["error_deg"] for r in rows], dtype=np.float64)
    return {
        "count": int(len(rows)),
        "mae": float(errors.mean()),
        "median": float(np.median(errors)),
        "acc_at_5deg": float(np.mean(errors <= 5.0)),
        "acc_at_10deg": float(np.mean(errors <= 10.0)),
        "frame_accuracy": float(np.mean([r["pred_label"] == r["true_label"] for r in rows])),
        "front_back_error_rate": float(np.mean([r["front_back_error"] for r in rows])),
        "large_error_rate": float(np.mean(errors > 90.0)),
        "opposite_error_rate": float(np.mean(errors > 150.0)),
    }


def summarize_group(rows: List[dict], key_fn) -> Dict[str, dict]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row))].append(row)
    return {k: metric_summary(v) for k, v in sorted(groups.items())}


@torch.no_grad()
def run_eval(cfg, checkpoint_path: Path, batch_size: int, num_workers: int) -> List[dict]:
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    _, _, test_ds = build_moving_datasets(cfg)
    meta_by_file_id = {}
    for file_id, _, meta_path in test_ds.records:
        meta_by_file_id[str(file_id)] = json.loads(meta_path.read_text(encoding="utf-8"))
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

    rows: List[dict] = []
    for batch in loader:
        batch_cpu = batch
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(batch_dev)
        logits = out["doa_logits"].float().cpu().numpy()
        pred_labels = logits.argmax(axis=-1)
        pred_angles = labels_to_angles(pred_labels, cfg.model.num_classes, tuple(cfg.model.azimuth_range))
        true_labels = batch_cpu["doa_labels"].cpu().numpy()
        true_angles = batch_cpu["doa_angles"].cpu().numpy()
        target_labels = batch_cpu["target_labels"].cpu().numpy()
        target_angles = batch_cpu["target_angles"].cpu().numpy()
        rendered_labels = batch_cpu["rendered_labels"].cpu().numpy()
        rendered_angles = batch_cpu["rendered_angles"].cpu().numpy()

        batch_size_now, steps = pred_labels.shape
        for b in range(batch_size_now):
            file_id = str(batch_cpu["file_id"][b])
            meta = meta_by_file_id[file_id]
            for t in range(steps):
                true_angle = float(true_angles[b, t])
                pred_angle = float(pred_angles[b, t])
                err = float(circular_error(np.array([pred_angle]), np.array([true_angle]))[0])
                rows.append({
                    "file_id": file_id,
                    "frame_idx": int(t),
                    "trajectory_type": str(batch_cpu["trajectory_type"][b]),
                    "speed_bin": str(batch_cpu["speed_bin"][b]),
                    "subject_id": str(meta.get("subject_id", "unknown")),
                    "room_profile": str(meta.get("room_profile", "unknown")),
                    "snr_db": float(batch_cpu["snr"][b]),
                    "rt60": float(batch_cpu["rt60"][b]),
                    "true_label": int(true_labels[b, t]),
                    "pred_label": int(pred_labels[b, t]),
                    "target_label": int(target_labels[b, t]),
                    "rendered_label": int(rendered_labels[b, t]),
                    "true_angle": true_angle,
                    "pred_angle": pred_angle,
                    "target_angle": float(target_angles[b, t]),
                    "rendered_angle": float(rendered_angles[b, t]),
                    "error_deg": err,
                    "front_back": front_back_label(true_angle),
                    "front_back_error": int(front_back_label(true_angle) != front_back_label(pred_angle)),
                    "angle_bin_10deg": angle_bin(true_angle),
                })
    return rows


def write_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config("configs/default.yaml", ["--config", args.config, "--train.num_workers", str(args.num_workers)])
    batch_size = args.batch_size or int(cfg.train.batch_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = run_eval(cfg, Path(args.checkpoint), batch_size=batch_size, num_workers=args.num_workers)
    static_rows = [r for r in rows if r["trajectory_type"] == "static"]
    static_rows_sorted = sorted(static_rows, key=lambda r: (r["error_deg"], r["file_id"], r["frame_idx"]), reverse=True)

    write_csv(static_rows, output_dir / "static_only_rows.csv")
    write_csv(static_rows_sorted[:500], output_dir / "worst_static_cases_top500.csv")

    summary = {
        "overall_static": metric_summary(static_rows),
        "by_subject": summarize_group(static_rows, lambda r: r["subject_id"]),
        "by_angle_bin_10deg": summarize_group(static_rows, lambda r: r["angle_bin_10deg"]),
        "by_front_back": summarize_group(static_rows, lambda r: r["front_back"]),
        "by_snr_bin": summarize_group(static_rows, lambda r: snr_bin(r["snr_db"])),
        "by_rt60_bin": summarize_group(static_rows, lambda r: rt60_bin(r["rt60"])),
        "by_room_profile": summarize_group(static_rows, lambda r: r["room_profile"]),
    }
    write_json(summary, output_dir / "summary.json")

    print(json.dumps({
        "output_dir": str(output_dir),
        "overall_static": summary["overall_static"],
        "top_subjects": sorted(summary["by_subject"].items(), key=lambda kv: kv[1]["mae"], reverse=True)[:5],
        "top_angle_bins": sorted(summary["by_angle_bin_10deg"].items(), key=lambda kv: kv[1]["mae"], reverse=True)[:8],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
