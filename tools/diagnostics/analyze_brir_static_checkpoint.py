#!/usr/bin/env python3
"""Analyze a static BRIR-like DOA checkpoint with dataset metadata groups."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List

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
    parser = argparse.ArgumentParser(description="Analyze static BRIR checkpoint by metadata groups")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def wrap_angle(angle: float) -> float:
    return ((float(angle) + 180.0) % 360.0) - 180.0


def angle_bin_label(angle: float, width: int = 30) -> str:
    wrapped = wrap_angle(angle)
    start = int(math.floor((wrapped + 180.0) / width) * width - 180)
    end = start + width
    return f"[{start}, {end})"


def front_side_back(angle: float) -> str:
    wrapped = wrap_angle(angle)
    if -60.0 <= wrapped <= 60.0:
        return "front"
    if 120.0 <= abs(wrapped) <= 180.0:
        return "back"
    return "side"


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


def group_rows(rows: List[dict], key_fn: Callable[[dict], str]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(key_fn(row))].append(row)
    return dict(grouped)


def metric_summary(rows: List[dict]) -> dict:
    errors = np.array([r["error_deg"] for r in rows], dtype=np.float32)
    correct = np.array([r["correct"] for r in rows], dtype=np.float32)
    return {
        "count": int(len(rows)),
        "mae": float(errors.mean()),
        "median": float(np.median(errors)),
        "accuracy": float(correct.mean()),
        "acc_at_5": float((errors <= 5.0).mean()),
        "acc_at_10": float((errors <= 10.0).mean()),
        "acc_at_20": float((errors <= 20.0).mean()),
        "large_error_rate": float((errors > 90.0).mean()),
        "opposite_error_rate": float((errors > 150.0).mean()),
        "std": float(errors.std()),
    }


def summarize_groups(grouped: Dict[str, List[dict]]) -> Dict[str, dict]:
    return {key: metric_summary(value) for key, value in sorted(grouped.items())}


def write_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_group_csv(summary: Dict[str, dict], path: Path) -> None:
    rows = [{"group": key, **value} for key, value in summary.items()]
    write_csv(rows, path)


def read_json_metadata(metadata_path: str) -> dict:
    json_path = Path(metadata_path).with_suffix(".json")
    if json_path.is_file():
        return json.loads(json_path.read_text(encoding="utf-8"))
    return {}


@torch.no_grad()
def evaluate_rows(cfg, checkpoint: Path, batch_size: int, num_workers: int) -> List[dict]:
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
    ckpt = load_checkpoint(str(checkpoint), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows: List[dict] = []
    global_idx = 0
    num_classes = cfg.model.num_classes
    azimuth_range = tuple(cfg.model.azimuth_range)

    for batch in loader:
        batch_size_now = len(batch["azimuth_label"])
        batch_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch_on_device)["logits"].float().cpu()
        probs = F.softmax(logits, dim=-1)
        pred_bins = probs.argmax(dim=-1).numpy()

        true_deg = batch["azimuth_deg"].cpu().numpy()
        true_bins = batch["azimuth_label"].cpu().numpy()
        pred_deg = bins_to_angles(pred_bins, num_classes, azimuth_range)
        err_deg = angular_error(pred_deg, true_deg)

        for local_idx in range(batch_size_now):
            seg = test_ds.segments[global_idx + local_idx]
            meta = read_json_metadata(seg["metadata_path"])
            target_az = float(meta.get("target_azimuth", true_deg[local_idx]))
            rendered_az = float(meta.get("rendered_azimuth", true_deg[local_idx]))
            rows.append(
                {
                    "file_id": seg["file_id"],
                    "start_sec": float(seg["start_sec"]),
                    "subject_id": str(meta.get("subject_id", "")),
                    "true_deg": float(true_deg[local_idx]),
                    "target_azimuth": target_az,
                    "rendered_azimuth": rendered_az,
                    "pred_deg": float(pred_deg[local_idx]),
                    "true_bin": int(true_bins[local_idx]),
                    "pred_bin": int(pred_bins[local_idx]),
                    "error_deg": float(err_deg[local_idx]),
                    "correct": int(pred_bins[local_idx] == true_bins[local_idx]),
                    "large_error": int(err_deg[local_idx] > 90.0),
                    "opposite_error": int(err_deg[local_idx] > 150.0),
                    "angle_bin_30deg": angle_bin_label(target_az, width=30),
                    "region": front_side_back(target_az),
                    "room_profile": str(meta.get("room_profile", "unknown")),
                    "target_rt60": float(meta.get("target_rt60", np.nan)),
                    "estimated_rt60": float(meta.get("estimated_rt60", np.nan)),
                    "snr_db": float(meta.get("snr_db", np.nan)),
                    "source_distance": float(meta.get("source_distance", np.nan)),
                    "num_paths": int(meta.get("num_paths", -1)),
                    "max_order": int(meta.get("max_order", -1)),
                }
            )
        global_idx += batch_size_now
    return rows


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, [])
    cfg.train.num_workers = args.num_workers
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = evaluate_rows(
        cfg=cfg,
        checkpoint=Path(args.checkpoint),
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
    )

    summaries = {
        "overall": metric_summary(rows),
        "by_angle_bin_30deg": summarize_groups(group_rows(rows, lambda r: r["angle_bin_30deg"])),
        "by_region": summarize_groups(group_rows(rows, lambda r: r["region"])),
        "by_room_profile": summarize_groups(group_rows(rows, lambda r: r["room_profile"])),
        "by_snr_bin": summarize_groups(bin_numeric(rows, "snr_db", [-10, -5, 0, 5, 10.0001])),
        "by_target_rt60_bin": summarize_groups(bin_numeric(rows, "target_rt60", [0.2, 0.35, 0.5, 0.65, 0.8001])),
        "by_estimated_rt60_bin": summarize_groups(bin_numeric(rows, "estimated_rt60", [0.0, 0.08, 0.12, 0.16, 0.25, 1.0])),
    }

    write_csv(rows, output_dir / "per_segment_errors.csv")
    for key, summary in summaries.items():
        if key == "overall":
            continue
        write_group_csv(summary, output_dir / f"{key}.csv")
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Saved {output_dir / 'summary.json'}")
    print(f"Saved {output_dir / 'per_segment_errors.csv'}")


if __name__ == "__main__":
    main()
