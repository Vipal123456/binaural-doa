#!/usr/bin/env python3
"""Evaluate moving DOA sequence checkpoints."""

import argparse

import torch
from torch.utils.data import DataLoader

from dataset.moving_dataset import build_moving_datasets
from metrics_dynamic import DynamicDOAMetrics
from models.binaural_doa_net import build_model
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.logger import setup_logger


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", type=str, required=True)
    known, remaining = parser.parse_known_args()
    cfg = load_config("configs/default.yaml", remaining)
    logger = setup_logger("moving-eval", cfg.output.log_dir)
    _, _, test_ds = build_moving_datasets(cfg, logger=logger)
    loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(known.checkpoint, map_location=str(device))
    model.load_state_dict(ckpt["model"])
    model.eval()
    metrics = DynamicDOAMetrics(cfg.model.num_classes, tuple(cfg.model.azimuth_range))
    target_metrics = DynamicDOAMetrics(cfg.model.num_classes, tuple(cfg.model.azimuth_range))
    rendered_metrics = DynamicDOAMetrics(cfg.model.num_classes, tuple(cfg.model.azimuth_range))
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(batch)
        logits_np = out["doa_logits"].float().cpu().numpy()
        metrics.update(
            logits_np,
            batch["doa_labels"].cpu().numpy(),
            batch["doa_angles"].float().cpu().numpy(),
            group_values={
                "trajectory": batch.get("trajectory_type", []),
                "speed": batch.get("speed_bin", []),
            },
        )
        target_metrics.update(
            logits_np,
            batch["target_labels"].cpu().numpy(),
            batch["target_angles"].float().cpu().numpy(),
            group_values={
                "trajectory": batch.get("trajectory_type", []),
                "speed": batch.get("speed_bin", []),
            },
        )
        rendered_metrics.update(
            logits_np,
            batch["rendered_labels"].cpu().numpy(),
            batch["rendered_angles"].float().cpu().numpy(),
            group_values={
                "trajectory": batch.get("trajectory_type", []),
                "speed": batch.get("speed_bin", []),
            },
        )
    results = metrics.compute()
    logger.info("=== Moving DOA evaluation ===")
    for key, value in results.items():
        logger.info(f"  {key}: {value:.4f}")
    logger.info("=== Pred vs target trajectory ===")
    for key, value in target_metrics.compute().items():
        logger.info(f"  target_{key}: {value:.4f}")
    logger.info("=== Pred vs rendered HRTF angles ===")
    for key, value in rendered_metrics.compute().items():
        logger.info(f"  rendered_{key}: {value:.4f}")


if __name__ == "__main__":
    main()
