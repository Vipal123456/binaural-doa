#!/usr/bin/env python3
"""Evaluate a KEMAR static checkpoint and summarize grouped test metrics.

This tool is designed for the flat-layout KEMAR + SofaMyRoom dataset:

    root/
      metadata.csv
      binaural/*.wav

It loads a resolved config + checkpoint, runs inference on the test split, and
exports:

    - overall.json
    - by_snr.csv
    - by_scene.csv
    - per_sample.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.static_dataset import build_static_datasets
from models.binaural_doa_net import build_model
from utils.angle import angular_error, bins_to_angles, wrap_angles
from utils.checkpoint import load_checkpoint
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate grouped KEMAR static metrics")
    parser.add_argument("--config", required=True, help="Resolved config yaml path")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--test_root", type=str, default=None, help="Override cfg.dataset.test_root")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="Override cfg.train.device, e.g. cpu or cuda:0")
    parser.add_argument("--log_interval", type=int, default=20, help="Print progress every N batches")
    return parser.parse_args()


def normalize_snr_label(value: str) -> str:
    text = str(value).strip()
    if text.lower() == "clean":
        return "clean"
    numeric = float(text)
    if abs(numeric - round(numeric)) < 1e-6:
        return str(int(round(numeric)))
    return f"{numeric:.2f}"


def front_back_label(angle_deg: np.ndarray) -> np.ndarray:
    wrapped = wrap_angles(np.asarray(angle_deg, dtype=np.float64))
    return (np.abs(wrapped) > 90.0).astype(np.int64)


def summarize(rows: List[dict]) -> dict:
    errors = np.asarray([r["error_deg"] for r in rows], dtype=np.float64)
    correct = np.asarray([r["correct"] for r in rows], dtype=np.float64)
    return {
        "count": int(len(rows)),
        "accuracy": float(correct.mean()),
        "mae": float(errors.mean()),
        "median": float(np.median(errors)),
        "acc_at_5": float((errors <= 5.0).mean()),
        "acc_at_10": float((errors <= 10.0).mean()),
        "fb_err": float(np.mean([r["fb_err"] for r in rows])),
        "opp_err": float(np.mean([r["opp_err"] for r in rows])),
        "large_err": float(np.mean([r["large_err"] for r in rows])),
    }


def load_metadata_map(test_root: Path) -> Dict[str, dict]:
    metadata_csv = test_root / "metadata.csv"
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"metadata.csv not found under {test_root}")
    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        return {str(row["file_id"]): row for row in csv.DictReader(f)}


@torch.no_grad()
def collect_rows(
    cfg,
    checkpoint_path: Path,
    batch_size: int,
    num_workers: int,
    device_override: str | None = None,
    log_interval: int = 20,
) -> List[dict]:
    _, _, test_ds = build_static_datasets(cfg)
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    requested_device = device_override or cfg.train.device
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    print(
        f"[grouped-eval] test_segments={len(test_ds)} batch_size={batch_size} "
        f"num_workers={num_workers} device={device}",
        flush=True,
    )
    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    meta_map = load_metadata_map(Path(cfg.dataset.test_root))
    rows: List[dict] = []
    global_idx = 0
    num_classes = cfg.model.num_classes
    azimuth_range = tuple(cfg.model.azimuth_range)

    for batch_idx, batch in enumerate(loader, start=1):
        batch_size_now = len(batch["azimuth_label"])
        batch_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch_dev)["logits"].float().cpu().numpy()
        pred_bins = logits.argmax(axis=-1)
        true_bins = batch["azimuth_label"].cpu().numpy()
        true_deg = batch["azimuth_deg"].cpu().numpy()
        pred_deg = bins_to_angles(pred_bins, num_classes, azimuth_range)
        err_deg = angular_error(pred_deg, true_deg)

        pred_fb = front_back_label(pred_deg)
        true_fb = front_back_label(true_deg)

        for local_idx in range(batch_size_now):
            seg = test_ds.segments[global_idx + local_idx]
            file_id = str(seg["file_id"])
            meta = meta_map[file_id]
            rows.append(
                {
                    "file_id": file_id,
                    "true_deg": float(true_deg[local_idx]),
                    "pred_deg": float(pred_deg[local_idx]),
                    "true_bin": int(true_bins[local_idx]),
                    "pred_bin": int(pred_bins[local_idx]),
                    "error_deg": float(err_deg[local_idx]),
                    "correct": int(pred_bins[local_idx] == true_bins[local_idx]),
                    "fb_err": int(pred_fb[local_idx] != true_fb[local_idx]),
                    "opp_err": int(err_deg[local_idx] > 150.0),
                    "large_err": int(err_deg[local_idx] >= 45.0),
                    "snr": normalize_snr_label(meta["snr_db"]),
                    "scene": str(meta["noise_scene"]),
                }
            )
        global_idx += batch_size_now
        if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == len(loader)):
            print(
                f"[grouped-eval] processed {global_idx}/{len(test_ds)} segments "
                f"({batch_idx}/{len(loader)} batches)",
                flush=True,
            )
    return rows


def write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, [])
    if args.test_root is not None:
        cfg.dataset.test_root = str(args.test_root)
    cfg.train.num_workers = int(args.num_workers)
    if args.batch_size is not None:
        cfg.train.batch_size = int(args.batch_size)
    if args.device is not None:
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

    snr_groups: Dict[str, List[dict]] = defaultdict(list)
    scene_groups: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        snr_groups[row["snr"]].append(row)
        scene_groups[row["scene"]].append(row)

    overall = summarize(rows)

    snr_order = ["clean", "10", "5", "0", "-5", "-10", "-15"]
    by_snr_rows = []
    for key in snr_order:
        if key in snr_groups:
            by_snr_rows.append({"snr": key, **summarize(snr_groups[key])})

    by_scene_rows = []
    for key in sorted(scene_groups):
        by_scene_rows.append({"scene": key, **summarize(scene_groups[key])})

    write_csv(
        output_dir / "per_sample.csv",
        ["file_id", "true_deg", "pred_deg", "true_bin", "pred_bin", "error_deg", "correct", "fb_err", "opp_err", "large_err", "snr", "scene"],
        rows,
    )
    write_csv(
        output_dir / "by_snr.csv",
        ["snr", "count", "accuracy", "mae", "median", "acc_at_5", "acc_at_10", "fb_err", "opp_err", "large_err"],
        by_snr_rows,
    )
    write_csv(
        output_dir / "by_scene.csv",
        ["scene", "count", "accuracy", "mae", "median", "acc_at_5", "acc_at_10", "fb_err", "opp_err", "large_err"],
        by_scene_rows,
    )
    (output_dir / "overall.json").write_text(
        json.dumps(overall, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Saved {output_dir}")


if __name__ == "__main__":
    main()
