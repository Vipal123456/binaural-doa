#!/usr/bin/env python3
"""在测试集上评估模型并可视化误差分布。"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.static_dataset import build_static_datasets
from models.binaural_doa_net import build_model
from utils.angle import angular_error, bins_to_angles
from utils.checkpoint import load_checkpoint
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze test error distribution")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config path")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/logs/error_analysis",
        help="Directory for plots and npz",
    )
    return parser.parse_args()


@torch.no_grad()
def collect_errors(cfg, checkpoint_path: str):
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    _, _, test_ds = build_static_datasets(cfg)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_errors = []
    all_true_degs = []
    all_pred_degs = []

    num_classes = cfg.model.num_classes
    azimuth_range = tuple(cfg.model.azimuth_range)

    for batch in test_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        out = model(batch)
        logits = out["logits"].float().cpu().numpy()
        pred_bins = logits.argmax(axis=-1)
        pred_degs = bins_to_angles(pred_bins, num_classes, azimuth_range)

        true_degs = batch["azimuth_deg"]
        if isinstance(true_degs, torch.Tensor):
            true_degs = true_degs.float().cpu().numpy()
        else:
            true_degs = np.asarray(true_degs, dtype=np.float32)

        err = angular_error(pred_degs, true_degs)

        all_errors.append(err)
        all_true_degs.append(true_degs)
        all_pred_degs.append(pred_degs)

    errors = np.concatenate(all_errors)
    true_degs = np.concatenate(all_true_degs)
    pred_degs = np.concatenate(all_pred_degs)

    return errors, true_degs, pred_degs, test_ds


def summarize(errors: np.ndarray) -> dict:
    summary = {
        "count": int(errors.size),
        "mae": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std": float(np.std(errors)),
        "p90": float(np.percentile(errors, 90)),
        "p95": float(np.percentile(errors, 95)),
        "p99": float(np.percentile(errors, 99)),
        "max": float(np.max(errors)),
        "lt_5": float(np.mean(errors < 5.0)),
        "lt_10": float(np.mean(errors < 10.0)),
        "lt_20": float(np.mean(errors < 20.0)),
        "lt_30": float(np.mean(errors < 30.0)),
    }
    return summary


def plot_error_hist_and_cdf(errors: np.ndarray, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bins = np.linspace(0, 180, 37)
    axes[0].hist(errors, bins=bins, color="#2a6f97", alpha=0.9, edgecolor="white")
    axes[0].set_title("Error Histogram")
    axes[0].set_xlabel("Angular Error (deg)")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.25)

    sorted_err = np.sort(errors)
    y = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    axes[1].plot(sorted_err, y, color="#f25f5c", linewidth=2)
    axes[1].set_title("Error CDF")
    axes[1].set_xlabel("Angular Error (deg)")
    axes[1].set_ylabel("Cumulative Probability")
    axes[1].set_xlim(0, 180)
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_error_buckets(errors: np.ndarray, output_path: Path, title: str) -> None:
    buckets = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 60), (60, 90), (90, 180)]
    labels = [f"[{a},{b})" for a, b in buckets]
    ratios = []
    for a, b in buckets:
        ratios.append(float(np.mean((errors >= a) & (errors < b))))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, np.array(ratios) * 100.0, color="#70a288")
    ax.set_title("Error Bucket Ratio")
    ax.set_xlabel("Error Range (deg)")
    ax.set_ylabel("Ratio (%)")
    ax.grid(axis="y", alpha=0.25)

    for bar, ratio in zip(bars, ratios):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.4, f"{ratio * 100.0:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(summary: dict, output_path: Path) -> None:
    lines = [
        f"count: {summary['count']}",
        f"mae: {summary['mae']:.4f}",
        f"median: {summary['median']:.4f}",
        f"std: {summary['std']:.4f}",
        f"p90: {summary['p90']:.4f}",
        f"p95: {summary['p95']:.4f}",
        f"p99: {summary['p99']:.4f}",
        f"max: {summary['max']:.4f}",
        f"lt_5: {summary['lt_5']:.4f}",
        f"lt_10: {summary['lt_10']:.4f}",
        f"lt_20: {summary['lt_20']:.4f}",
        f"lt_30: {summary['lt_30']:.4f}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, [])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    errors, true_degs, pred_degs, _ = collect_errors(cfg, args.checkpoint)
    summary = summarize(errors)

    stem = "test_error_distribution"
    npz_path = output_dir / f"{stem}.npz"
    txt_path = output_dir / f"{stem}_summary.txt"
    hist_cdf_path = output_dir / f"{stem}_hist_cdf.png"
    buckets_path = output_dir / f"{stem}_buckets.png"

    np.savez(npz_path, errors=errors, true_degs=true_degs, pred_degs=pred_degs)
    write_summary(summary, txt_path)

    title = f"Test Error Distribution | MAE={summary['mae']:.2f} deg | Median={summary['median']:.2f} deg"
    plot_error_hist_and_cdf(errors, hist_cdf_path, title)
    plot_error_buckets(errors, buckets_path, title)

    print(f"Saved: {npz_path}")
    print(f"Saved: {txt_path}")
    print(f"Saved: {hist_cdf_path}")
    print(f"Saved: {buckets_path}")


if __name__ == "__main__":
    main()
