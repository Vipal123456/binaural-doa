#!/usr/bin/env python3
"""Measure how well a learned target-dominance mask matches component oracles."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.static_dataset import build_static_datasets
from models.binaural_doa_net import build_model
from utils.checkpoint import load_checkpoint
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--histogram_bins", type=int, default=200)
    parser.add_argument("--calibration_bins", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=25)
    return parser.parse_args()


class StreamingMaskMetrics:
    def __init__(self, histogram_bins: int, calibration_bins: int) -> None:
        self.histogram_bins = histogram_bins
        self.calibration_bins = calibration_bins
        self.count = 0
        self.sum_pred = 0.0
        self.sum_target = 0.0
        self.sum_pred2 = 0.0
        self.sum_target2 = 0.0
        self.sum_cross = 0.0
        self.sum_abs = 0.0
        self.sum_brier = 0.0
        self.sum_bce = 0.0
        self.positive_hist = torch.zeros(histogram_bins, dtype=torch.float64)
        self.negative_hist = torch.zeros(histogram_bins, dtype=torch.float64)
        self.pred_hist = torch.zeros(histogram_bins, dtype=torch.float64)
        self.cal_count = torch.zeros(calibration_bins, dtype=torch.float64)
        self.cal_pred = torch.zeros(calibration_bins, dtype=torch.float64)
        self.cal_target = torch.zeros(calibration_bins, dtype=torch.float64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.detach().float().clamp(1.0e-6, 1.0 - 1.0e-6).reshape(-1)
        target = target.detach().float().clamp(0.0, 1.0).reshape(-1)
        count = prediction.numel()
        self.count += count
        self.sum_pred += prediction.sum().item()
        self.sum_target += target.sum().item()
        self.sum_pred2 += prediction.square().sum().item()
        self.sum_target2 += target.square().sum().item()
        self.sum_cross += (prediction * target).sum().item()
        self.sum_abs += (prediction - target).abs().sum().item()
        self.sum_brier += (prediction - target).square().sum().item()
        self.sum_bce += F.binary_cross_entropy(prediction, target, reduction="sum").item()

        hist_index = torch.clamp(
            (prediction * self.histogram_bins).long(), max=self.histogram_bins - 1
        )
        positive = target >= 0.5
        self.positive_hist += torch.bincount(
            hist_index[positive], minlength=self.histogram_bins
        ).double().cpu()
        self.negative_hist += torch.bincount(
            hist_index[~positive], minlength=self.histogram_bins
        ).double().cpu()
        self.pred_hist += torch.bincount(
            hist_index, minlength=self.histogram_bins
        ).double().cpu()

        cal_index = torch.clamp(
            (prediction * self.calibration_bins).long(), max=self.calibration_bins - 1
        )
        self.cal_count += torch.bincount(
            cal_index, minlength=self.calibration_bins
        ).double().cpu()
        self.cal_pred += torch.bincount(
            cal_index, weights=prediction, minlength=self.calibration_bins
        ).double().cpu()
        self.cal_target += torch.bincount(
            cal_index, weights=target, minlength=self.calibration_bins
        ).double().cpu()

    def summary(self) -> dict:
        if self.count == 0:
            return {"count": 0}
        n = float(self.count)
        mean_pred = self.sum_pred / n
        mean_target = self.sum_target / n
        var_pred = max(self.sum_pred2 / n - mean_pred**2, 0.0)
        var_target = max(self.sum_target2 / n - mean_target**2, 0.0)
        covariance = self.sum_cross / n - mean_pred * mean_target
        denominator = max(var_pred * var_target, 0.0) ** 0.5
        correlation = covariance / denominator if denominator > 0.0 else float("nan")

        negatives_below = torch.cumsum(self.negative_hist, dim=0) - self.negative_hist
        positives = self.positive_hist.sum().item()
        negatives = self.negative_hist.sum().item()
        auc_numerator = (
            self.positive_hist * (negatives_below + 0.5 * self.negative_hist)
        ).sum().item()
        auc = auc_numerator / (positives * negatives) if positives and negatives else float("nan")

        cdf = torch.cumsum(self.pred_hist, dim=0)
        quantiles = {}
        for probability in (0.05, 0.25, 0.5, 0.75, 0.95):
            index = int(torch.searchsorted(cdf, probability * n).clamp(max=self.histogram_bins - 1))
            quantiles[str(probability)] = (index + 0.5) / self.histogram_bins

        calibration = []
        for index in range(self.calibration_bins):
            count = self.cal_count[index].item()
            calibration.append(
                {
                    "bin": index,
                    "count": int(count),
                    "mean_prediction": self.cal_pred[index].item() / count if count else None,
                    "mean_target": self.cal_target[index].item() / count if count else None,
                }
            )
        return {
            "count": self.count,
            "prediction_mean": mean_pred,
            "prediction_std": var_pred**0.5,
            "target_mean": mean_target,
            "target_std": var_target**0.5,
            "mae": self.sum_abs / n,
            "brier": self.sum_brier / n,
            "bce": self.sum_bce / n,
            "pearson_correlation": correlation,
            "roc_auc_target_ge_0_5": auc,
            "prediction_quantiles_approx": quantiles,
            "calibration": calibration,
        }


def load_metadata(root: Path) -> dict[str, dict]:
    with (root / "metadata.csv").open("r", encoding="utf-8", newline="") as handle:
        return {str(row["file_id"]): row for row in csv.DictReader(handle)}


def complex_spectrum(batch: dict, prefix: str, side: str) -> torch.Tensor:
    return torch.complex(
        batch[f"{prefix}spec_real_{side}"], batch[f"{prefix}spec_imag_{side}"]
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, [])
    cfg.dataset.component_supervision_enabled = True
    cfg.train.device = args.device
    _, val_dataset, _ = build_static_datasets(cfg)
    loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    metadata = load_metadata(Path(cfg.dataset.val_root))
    metrics = defaultdict(
        lambda: StreamingMaskMetrics(args.histogram_bins, args.calibration_bins)
    )
    offset = 0
    for batch_index, batch in enumerate(loader, start=1):
        batch_size = len(batch["azimuth_label"])
        device_batch = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        output = model(device_batch)
        prediction = output.get("target_probability")
        if prediction is None:
            raise RuntimeError("The selected model does not expose target_probability")
        target_l = complex_spectrum(device_batch, "target_", "L")
        target_r = complex_spectrum(device_batch, "target_", "R")
        interferer_l = complex_spectrum(device_batch, "interferer_", "L")
        interferer_r = complex_spectrum(device_batch, "interferer_", "R")
        target_power = target_l.abs().square() + target_r.abs().square()
        interferer_power = interferer_l.abs().square() + interferer_r.abs().square()
        ideal = target_power / (target_power + interferer_power + 1.0e-8)

        metrics["overall"].update(prediction, ideal)
        sir_to_indices: dict[str, list[int]] = defaultdict(list)
        for local_index in range(batch_size):
            file_id = str(val_dataset.segments[offset + local_index]["file_id"])
            sir = str(metadata[file_id].get("target_sir_db", "unknown"))
            sir_to_indices[sir].append(local_index)
        for sir, indices in sir_to_indices.items():
            metrics[f"sir_{sir}"].update(prediction[indices], ideal[indices])
        offset += batch_size
        if batch_index % args.log_interval == 0 or batch_index == len(loader):
            print(
                f"[mask-diagnostic] processed {offset}/{len(val_dataset)} samples "
                f"({batch_index}/{len(loader)} batches)",
                flush=True,
            )

    result = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "validation_root": str(Path(cfg.dataset.val_root).resolve()),
        "samples": len(val_dataset),
        "metrics": {key: value.summary() for key, value in sorted(metrics.items())},
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[mask-diagnostic] saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
