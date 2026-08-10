#!/usr/bin/env python3
"""Measure how much a trained fusion model relies on content and cue branches."""

import argparse
import json
import math
import os
import sys
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dataset.static_dataset import build_static_datasets
from metrics import DOAMetrics
from models.binaural_doa_net import build_model
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.seed import set_seed


MODES = ("full", "content_shuffle", "cue_shuffle", "content_zero", "cue_zero")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--gradient-batches", type=int, default=16)
    return parser.parse_args()


def mismatched_permutation(labels: torch.Tensor) -> torch.Tensor:
    """Choose a cyclic shift with the fewest unchanged labels."""
    batch_size = labels.numel()
    if batch_size < 2:
        return torch.arange(batch_size, device=labels.device)
    indices = torch.arange(batch_size, device=labels.device)
    candidates = [torch.roll(indices, shifts=shift) for shift in range(1, batch_size)]
    mismatch_counts = torch.stack([(labels[candidate] != labels).sum() for candidate in candidates])
    return candidates[int(mismatch_counts.argmax().item())]


def fusion_pre_hook(mode: str, content_dim: int, permutation: torch.Tensor):
    def hook(_module, inputs):
        fused = inputs[0]
        content = fused[..., :content_dim]
        cue = fused[..., content_dim:]
        if mode == "content_shuffle":
            content = content[permutation]
        elif mode == "cue_shuffle":
            cue = cue[permutation]
        elif mode == "content_zero":
            content = torch.zeros_like(content)
        elif mode == "cue_zero":
            cue = torch.zeros_like(cue)
        return (torch.cat([content, cue], dim=-1),)

    return hook


def move_batch(batch: Dict, device: torch.device) -> Dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def finite_metrics(values: Dict[str, float]) -> Dict[str, float | None]:
    return {
        key: float(value) if math.isfinite(float(value)) else None
        for key, value in values.items()
    }


def main() -> None:
    args = parse_args()
    cfg = load_config("configs/default.yaml", ["--config", args.config])
    set_seed(cfg.dataset.split_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds = build_static_datasets(cfg)
    dataset = val_ds if args.split == "val" else test_ds
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(cfg).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    content_dim = int(cfg.model.content_fusion_dim)
    fusion_dim = int(model.fusion_norm.normalized_shape[0])
    cue_dim = fusion_dim - content_dim
    if content_dim <= 0 or cue_dim <= 0:
        raise ValueError(f"Expected two non-empty branches, got {content_dim}+{cue_dim}")

    metrics = {
        mode: DOAMetrics(
            num_classes=cfg.model.num_classes,
            azimuth_range=tuple(cfg.model.azimuth_range),
            class_angles_deg=cfg.model.get("class_angles_deg", None),
        )
        for mode in MODES
    }

    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            batch = move_batch(raw_batch, device)
            labels = batch["azimuth_label"]
            permutation = mismatched_permutation(labels)
            labels_np = labels.cpu().numpy()
            true_degs = batch["azimuth_deg"].cpu().numpy()
            for mode in MODES:
                handle = None
                if mode != "full":
                    handle = model.fusion_norm.register_forward_pre_hook(
                        fusion_pre_hook(mode, content_dim, permutation)
                    )
                try:
                    logits = model(batch)["logits"]
                finally:
                    if handle is not None:
                        handle.remove()
                metrics[mode].update(logits.float().cpu().numpy(), labels_np, true_degs)
            if (batch_idx + 1) % 25 == 0:
                print(f"evaluated {batch_idx + 1}/{len(loader)} batches", flush=True)

    perturbation_metrics = {
        mode: finite_metrics(metric.compute()) for mode, metric in metrics.items()
    }
    partial_output = f"{args.output}.perturbations.json"
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(partial_output, "w", encoding="utf-8") as handle:
        json.dump(perturbation_metrics, handle, indent=2, sort_keys=True)

    gradient_sums = {
        "content": {"activation_rms": 0.0, "gradient_rms": 0.0, "grad_x_abs_per_dim": 0.0, "grad_x_abs_total": 0.0},
        "cue": {"activation_rms": 0.0, "gradient_rms": 0.0, "grad_x_abs_per_dim": 0.0, "grad_x_abs_total": 0.0},
    }
    gradient_count = 0
    recurrent = model.temporal_head.temporal_encoder
    recurrent_was_training = recurrent.training
    recurrent.train()
    for batch_idx, raw_batch in enumerate(loader):
        if batch_idx >= args.gradient_batches:
            break
        batch = move_batch(raw_batch, device)
        captured = {}

        def capture_hook(_module, inputs):
            captured["fused"] = inputs[0]
            captured["fused"].retain_grad()

        handle = model.fusion_norm.register_forward_pre_hook(capture_hook)
        model.zero_grad(set_to_none=True)
        logits = model(batch)["logits"]
        loss = F.cross_entropy(logits, batch["azimuth_label"])
        loss.backward()
        handle.remove()

        fused = captured["fused"].detach()
        gradient = captured["fused"].grad.detach()
        for name, start, end in (
            ("content", 0, content_dim),
            ("cue", content_dim, fusion_dim),
        ):
            value_part = fused[..., start:end]
            grad_part = gradient[..., start:end]
            grad_x = (value_part * grad_part).abs()
            gradient_sums[name]["activation_rms"] += value_part.square().mean().sqrt().item()
            gradient_sums[name]["gradient_rms"] += grad_part.square().mean().sqrt().item()
            gradient_sums[name]["grad_x_abs_per_dim"] += grad_x.mean().item()
            gradient_sums[name]["grad_x_abs_total"] += grad_x.sum(dim=-1).mean().item()
        gradient_count += 1
    recurrent.train(recurrent_was_training)

    gradient_stats = {
        name: {key: value / max(gradient_count, 1) for key, value in stats.items()}
        for name, stats in gradient_sums.items()
    }

    weight = recurrent.weight_ih_l0.detach()
    content_weight = weight[:, :content_dim]
    cue_weight = weight[:, content_dim:]
    input_weight_stats = {
        "content_frobenius": content_weight.norm().item(),
        "cue_frobenius": cue_weight.norm().item(),
        "content_rms_per_weight": content_weight.square().mean().sqrt().item(),
        "cue_rms_per_weight": cue_weight.square().mean().sqrt().item(),
        "content_column_norm_mean": content_weight.norm(dim=0).mean().item(),
        "cue_column_norm_mean": cue_weight.norm(dim=0).mean().item(),
    }

    results = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "samples": int(sum(len(part) for part in metrics["full"]._true_bins)),
        "content_dim": content_dim,
        "cue_dim": cue_dim,
        "perturbation_metrics": perturbation_metrics,
        "gradient_batches": gradient_count,
        "gradient_stats": gradient_stats,
        "first_recurrent_input_weight_stats": input_weight_stats,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
