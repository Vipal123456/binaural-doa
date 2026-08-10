#!/usr/bin/env python3
"""Evaluate a flat-layout static checkpoint and summarize grouped metrics.

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
    parser.add_argument(
        "--cue_stat_mode",
        type=str,
        default=None,
        help="Override model.cue_stat_mode (used by controlled oracle evaluations)",
    )
    parser.add_argument(
        "--cue_target_bias_mode",
        type=str,
        default=None,
        help="Override the target-aware CPSD injection for controlled diagnostics",
    )
    parser.add_argument(
        "--component_supervision",
        action="store_true",
        help="Load aligned target/interferer component spectra from the dataset root",
    )
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


def snr_sort_key(label: str) -> tuple[int, float | str]:
    """Sort clean first, numeric SNRs high-to-low, then unknown labels."""
    if label == "clean":
        return (0, 0.0)
    try:
        return (1, -float(label))
    except ValueError:
        return (2, label)


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
        pred_deg = bins_to_angles(
            pred_bins,
            num_classes,
            azimuth_range,
            getattr(cfg.model, "class_angles_deg", None),
        )
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
                    "snr": normalize_snr_label(
                        meta.get(
                            "target_sir_db",
                            meta.get("snr_db", meta.get("target_snr_db")),
                        )
                    ),
                    "rt60_s": float(meta.get("rt60_s", "nan")),
                    "distance_m": float(meta.get("distance_m", "nan")),
                    "subject_id": str(meta.get("subject_id", "unknown")),
                    "room_id": str(meta.get("room_id", "unknown")),
                    "scene": str(meta.get("demand_scene", meta.get("noise_scene", "unknown"))),
                    "condition": str(meta.get("condition", "")),
                    "diffuse_snr_db": str(meta.get("target_diffuse_snr_db", "")),
                    "angular_separation_deg": float(meta.get("angular_separation_deg", "nan")),
                    "paired_test_key": str(meta.get("paired_test_key", "")),
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
    if args.cue_stat_mode is not None:
        cfg.model.cue_stat_mode = str(args.cue_stat_mode)
    if args.cue_target_bias_mode is not None:
        cfg.model.cue_target_bias_mode = str(args.cue_target_bias_mode)
    if args.component_supervision:
        cfg.dataset.component_supervision_enabled = True

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

    by_snr_rows = []
    for key in sorted(snr_groups, key=snr_sort_key):
        by_snr_rows.append({"snr": key, **summarize(snr_groups[key])})

    by_scene_rows = []
    for key in sorted(scene_groups):
        by_scene_rows.append({"scene": key, **summarize(scene_groups[key])})

    # The directional DNS test protocol is the union of two controlled sweeps.
    # Keep the fixed variable explicit so 5 dB samples from different RT60s are
    # not accidentally interpreted as one by-SIR condition.
    sir_sweep_groups: Dict[str, List[dict]] = defaultdict(list)
    rt60_sweep_groups: Dict[float, List[dict]] = defaultdict(list)
    for row in rows:
        if np.isfinite(row["rt60_s"]) and abs(row["rt60_s"] - 0.6) < 1e-6:
            sir_sweep_groups[row["snr"]].append(row)
        try:
            sir_value = float(row["snr"])
        except ValueError:
            sir_value = float("nan")
        if np.isfinite(sir_value) and abs(sir_value - 5.0) < 1e-6:
            rt60_sweep_groups[row["rt60_s"]].append(row)

    by_sir_sweep_rows = [
        {"sir_db": key, **summarize(sir_sweep_groups[key])}
        for key in sorted(sir_sweep_groups, key=snr_sort_key)
    ]
    by_rt60_sweep_rows = [
        {"rt60_s": key, **summarize(rt60_sweep_groups[key])}
        for key in sorted(rt60_sweep_groups)
    ]

    write_csv(
        output_dir / "per_sample.csv",
        ["file_id", "true_deg", "pred_deg", "true_bin", "pred_bin", "error_deg", "correct", "fb_err", "opp_err", "large_err", "snr", "diffuse_snr_db", "rt60_s", "distance_m", "subject_id", "room_id", "scene", "condition", "angular_separation_deg", "paired_test_key"],
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
    write_csv(
        output_dir / "by_sir_sweep_rt60_0p6.csv",
        ["sir_db", "count", "accuracy", "mae", "median", "acc_at_5", "acc_at_10", "fb_err", "opp_err", "large_err"],
        by_sir_sweep_rows,
    )
    write_csv(
        output_dir / "by_rt60_sweep_sir_5.csv",
        ["rt60_s", "count", "accuracy", "mae", "median", "acc_at_5", "acc_at_10", "fb_err", "opp_err", "large_err"],
        by_rt60_sweep_rows,
    )
    (output_dir / "overall.json").write_text(
        json.dumps(overall, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Saved {output_dir}")


if __name__ == "__main__":
    main()
