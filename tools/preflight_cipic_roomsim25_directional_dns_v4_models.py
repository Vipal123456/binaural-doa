#!/usr/bin/env python3
"""Build each queued model and run one real dataset sample through it on CPU."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data._utils.collate import default_collate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.static_dataset import build_static_datasets
from models.binaural_doa_net import build_model
from tools.generate_cipic_roomsim25 import SPLIT_SUBJECTS
from utils.config import load_config


EXPECTED_LENGTHS = (120_000, 12_000, 64_800)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("configs", type=Path, nargs="+")
    return parser.parse_args()


def validate_dprtf_template(path: Path) -> None:
    payload = loadmat(path, variable_names=["subject_ids", "azimuth_deg", "sample_rate"])
    subjects = {str(value) for value in np.asarray(payload["subject_ids"]).reshape(-1)}
    if subjects != set(SPLIT_SUBJECTS["train"]):
        raise RuntimeError(
            f"DP-RTF template subjects do not match train split: {sorted(subjects)}"
        )
    angles = np.asarray(payload["azimuth_deg"], dtype=np.float64).reshape(-1)
    if len(angles) != 25:
        raise RuntimeError(f"DP-RTF template has {len(angles)} directions, expected 25")
    if int(np.asarray(payload["sample_rate"]).reshape(-1)[0]) != 16_000:
        raise RuntimeError("DP-RTF template sample rate is not 16 kHz")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    reports: List[dict] = []
    errors: List[str] = []

    for config_path in args.configs:
        config_path = config_path.resolve()
        item = {"config": str(config_path), "passed": False}
        try:
            cfg = load_config("configs/default.yaml", ["--config", str(config_path)])
            cfg.dataset.root_dir = str(dataset_root / "train")
            cfg.dataset.train_root = str(dataset_root / "train")
            cfg.dataset.val_root = str(dataset_root / "val")
            cfg.dataset.test_root = str(dataset_root / "test")
            cfg.train.device = "cpu"
            cfg.train.amp = False
            if int(cfg.model.num_classes) != 25 or len(cfg.model.class_angles_deg) != 25:
                raise RuntimeError("queued model is not configured for the 25 CIPIC directions")
            if cfg.model.type == "dprtf_doa_cls":
                validate_dprtf_template((ROOT / cfg.model.dprtf_template_path).resolve())

            train_ds, val_ds, test_ds = build_static_datasets(cfg)
            lengths = (len(train_ds), len(val_ds), len(test_ds))
            if lengths != EXPECTED_LENGTHS:
                raise RuntimeError(f"dataset lengths {lengths} != {EXPECTED_LENGTHS}")
            batch = default_collate([train_ds[0]])
            model = build_model(cfg).cpu().eval()
            with torch.no_grad():
                output = model(batch)
            logits = output.get("logits")
            if logits is None or tuple(logits.shape) != (1, 25):
                raise RuntimeError(f"invalid logits shape: {None if logits is None else tuple(logits.shape)}")
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("model produced non-finite logits")
            parameters = sum(value.numel() for value in model.parameters() if value.requires_grad)
            item.update(
                {
                    "passed": True,
                    "model_type": cfg.model.type,
                    "dataset_lengths": lengths,
                    "logits_shape": list(logits.shape),
                    "trainable_parameters": parameters,
                }
            )
            print(
                f"[preflight] {config_path.name}: passed, params={parameters:,}, logits={tuple(logits.shape)}",
                flush=True,
            )
            del model, output, logits, batch, train_ds, val_ds, test_ds
            gc.collect()
        except Exception as exc:
            detail = f"{config_path}: {type(exc).__name__}: {exc}"
            item["error"] = detail
            errors.append(detail)
            print(f"[preflight] FAILED: {detail}", flush=True)
        reports.append(item)

    report = {
        "passed": not errors,
        "dataset_root": str(dataset_root),
        "elapsed_seconds": time.time() - started,
        "errors": errors,
        "models": reports,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
