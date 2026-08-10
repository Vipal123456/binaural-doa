#!/usr/bin/env python3
"""Evaluate a static DOA checkpoint on the KEMAR moving center-label test set."""

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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.feature_extractor import FeatureExtractor  # noqa: E402
from models.binaural_doa_net import build_model  # noqa: E402
from utils.angle import angular_error, bins_to_angles, wrap_angles  # noqa: E402
from utils.checkpoint import load_checkpoint  # noqa: E402
from utils.config import load_config  # noqa: E402

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None


class MovingCenterLabelDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        sample_rate: int,
        segment_seconds: float,
        num_classes: int,
        azimuth_range,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: str,
    ):
        if sf is None:
            raise ImportError("soundfile is required")
        self.root_dir = Path(root_dir)
        self.sample_rate = int(sample_rate)
        self.segment_seconds = float(segment_seconds)
        self.num_classes = int(num_classes)
        self.azimuth_range = tuple(azimuth_range)
        self.feature_extractor = FeatureExtractor(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
        )
        self.records = self._scan()

    def _wrap_center_to_range(self, angle_deg: float) -> float:
        return ((float(angle_deg) + 180.0) % 360.0) - 180.0

    def _angle_to_label(self, angle_deg: float) -> int:
        lo, hi = self.azimuth_range
        span = hi - lo
        bin_width = span / self.num_classes
        wrapped = self._wrap_center_to_range(angle_deg)
        idx = int((wrapped - lo) / bin_width)
        return min(max(idx, 0), self.num_classes - 1)

    def _scan(self) -> List[dict]:
        meta_path = self.root_dir / "metadata.csv"
        rows = list(csv.DictReader(meta_path.open("r", encoding="utf-8", newline="")))
        records: List[dict] = []
        for row in rows:
            wav_path = Path(str(row["wav_path"]))
            if not wav_path.is_absolute():
                wav_path = (self.root_dir / wav_path).resolve()
            center_deg = float(row["center_azimuth_deg"])
            records.append(
                {
                    "file_id": str(row["file_id"]),
                    "wav_path": wav_path,
                    "center_deg": center_deg,
                    "center_label": self._angle_to_label(center_deg),
                    "meta": row,
                }
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        audio, sr = sf.read(str(rec["wav_path"]), dtype="float32", always_2d=True)
        if sr != self.sample_rate:
            raise ValueError(f"Unexpected sample rate {sr} for {rec['wav_path']}; expected {self.sample_rate}")
        audio = audio.T
        target_len = int(round(self.segment_seconds * self.sample_rate))
        if audio.shape[1] < target_len:
            audio = np.pad(audio, ((0, 0), (0, target_len - audio.shape[1])), mode="constant")
        audio = audio[:2, :target_len]
        feats = self.feature_extractor.extract(torch.from_numpy(audio).float())
        return {
            "file_id": rec["file_id"],
            "log_mag_L": feats["log_mag_L"],
            "log_mag_R": feats["log_mag_R"],
            "spec_real_L": feats["spec_real_L"],
            "spec_imag_L": feats["spec_imag_L"],
            "spec_real_R": feats["spec_real_R"],
            "spec_imag_R": feats["spec_imag_R"],
            "ipd": feats["ipd"],
            "ild": feats["ild"],
            "ipd_sin": feats["ipd_sin"],
            "ipd_cos": feats["ipd_cos"],
            "coherence": feats["coherence"],
            "azimuth_label": rec["center_label"],
            "azimuth_deg": rec["center_deg"],
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log_interval", type=int, default=20)
    return p.parse_args()


def normalize_snr_label(value: str) -> str:
    x = float(value)
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.2f}"


def front_back_label(angle_deg: np.ndarray) -> np.ndarray:
    wrapped = wrap_angles(np.asarray(angle_deg, dtype=np.float64))
    return (np.abs(wrapped) > 90.0).astype(np.int64)


def angle_region(angle_deg: float) -> str:
    a = float(angle_deg) % 360.0
    if a < 45 or a >= 315:
        return "front"
    if a < 135:
        return "right"
    if a < 225:
        return "back"
    return "left"


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


def write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, [])
    if args.batch_size is not None:
        cfg.train.batch_size = int(args.batch_size)
    cfg.train.num_workers = int(args.num_workers)
    if args.device is not None:
        cfg.train.device = str(args.device)

    ds = MovingCenterLabelDataset(
        root_dir=str(args.test_root),
        sample_rate=cfg.dataset.sample_rate,
        segment_seconds=cfg.dataset.segment_seconds,
        num_classes=cfg.model.num_classes,
        azimuth_range=tuple(cfg.model.azimuth_range),
        n_fft=cfg.feature.n_fft,
        hop_length=cfg.feature.hop_length,
        win_length=cfg.feature.win_length,
        window=cfg.feature.window,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    requested_device = args.device or cfg.train.device
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    print(
        f"[static-on-moving] samples={len(ds)} batch_size={cfg.train.batch_size} "
        f"num_workers={cfg.train.num_workers} device={device}",
        flush=True,
    )

    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    meta_map: Dict[str, dict] = {}
    with (Path(args.test_root) / "metadata.csv").open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            meta_map[str(row["file_id"])] = row

    rows: List[dict] = []
    num_classes = cfg.model.num_classes
    azimuth_range = tuple(cfg.model.azimuth_range)

    for batch_idx, batch in enumerate(loader, start=1):
        file_ids = batch["file_id"]
        batch_size_now = len(file_ids)
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

        for i in range(batch_size_now):
            file_id = str(file_ids[i])
            meta = meta_map[file_id]
            center_deg = float(meta["center_azimuth_deg"])
            rows.append(
                {
                    "file_id": file_id,
                    "true_deg": float(true_deg[i]),
                    "pred_deg": float(pred_deg[i]),
                    "true_bin": int(true_bins[i]),
                    "pred_bin": int(pred_bins[i]),
                    "error_deg": float(err_deg[i]),
                    "correct": int(pred_bins[i] == true_bins[i]),
                    "fb_err": int(pred_fb[i] != true_fb[i]),
                    "opp_err": int(err_deg[i] > 150.0),
                    "large_err": int(err_deg[i] >= 45.0),
                    "trajectory": str(meta["trajectory_type"]),
                    "speed": str(int(round(float(meta["speed_deg_per_sec"])))),
                    "motion": str(meta["motion_condition_id"]),
                    "scene": str(meta["noise_scene"]),
                    "snr": normalize_snr_label(meta["snr_db"]),
                    "room_size": str(meta["room_size"]),
                    "angle_region": angle_region(center_deg),
                }
            )

        if args.log_interval > 0 and (batch_idx % args.log_interval == 0 or batch_idx == len(loader)):
            print(
                f"[static-on-moving] processed {len(rows)}/{len(ds)} samples "
                f"({batch_idx}/{len(loader)} batches)",
                flush=True,
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "overall.json").write_text(
        json.dumps(summarize(rows), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(
        output_dir / "per_sample.csv",
        ["file_id", "true_deg", "pred_deg", "true_bin", "pred_bin", "error_deg", "correct", "fb_err", "opp_err", "large_err", "trajectory", "speed", "motion", "scene", "snr", "room_size", "angle_region"],
        rows,
    )

    groups: Dict[str, Dict[str, List[dict]]] = {
        "by_speed": defaultdict(list),
        "by_trajectory": defaultdict(list),
        "by_motion": defaultdict(list),
        "by_snr": defaultdict(list),
        "by_scene": defaultdict(list),
        "by_roomsize": defaultdict(list),
        "by_region": defaultdict(list),
    }
    for row in rows:
        groups["by_speed"][row["speed"]].append(row)
        groups["by_trajectory"][row["trajectory"]].append(row)
        groups["by_motion"][row["motion"]].append(row)
        groups["by_snr"][row["snr"]].append(row)
        groups["by_scene"][row["scene"]].append(row)
        groups["by_roomsize"][row["room_size"]].append(row)
        groups["by_region"][row["angle_region"]].append(row)

    specs = [
        ("by_speed.csv", "speed", ["20", "40"], groups["by_speed"]),
        ("by_trajectory.csv", "trajectory", ["linear", "piecewise"], groups["by_trajectory"]),
        ("by_motion.csv", "motion", ["linear_20", "linear_40", "piecewise_20", "piecewise_40"], groups["by_motion"]),
        ("by_snr.csv", "snr", ["0", "-5", "-10"], groups["by_snr"]),
        ("by_scene.csv", "scene", ["TBUS", "NPARK"], groups["by_scene"]),
        ("by_roomsize.csv", "room_size", ["small", "large"], groups["by_roomsize"]),
        ("by_region.csv", "angle_region", ["front", "right", "back", "left"], groups["by_region"]),
    ]
    for filename, key_name, key_order, group_map in specs:
        out_rows = []
        for key in key_order:
            if key in group_map:
                out_rows.append({key_name: key, **summarize(group_map[key])})
        write_csv(
            output_dir / filename,
            [key_name, "count", "accuracy", "mae", "median", "acc_at_5", "acc_at_10", "fb_err", "opp_err", "large_err"],
            out_rows,
        )

    print(f"Saved {output_dir}", flush=True)


if __name__ == "__main__":
    main()
