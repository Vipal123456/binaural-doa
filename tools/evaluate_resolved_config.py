#!/usr/bin/env python3
"""Evaluate a checkpoint from a fully resolved YAML config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.static_dataset import build_static_datasets
from engine.evaluator import Evaluator
from models.binaural_doa_net import build_model
from utils.checkpoint import load_checkpoint
from utils.config import Config
from utils.logger import setup_logger
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True, help="Path to resolved_config.yaml")
    parser.add_argument("--test_root", required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    return parser.parse_args()


def main() -> dict:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    cfg.dataset.test_root = args.test_root
    cfg.output.log_dir = args.log_dir
    cfg.train.device = args.device
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers

    set_seed(cfg.dataset.split_seed)
    logger = setup_logger("DOA-eval", cfg.output.log_dir)

    _, _, test_ds = build_static_datasets(cfg)
    logger.info(f"测试集: {len(test_ds)} 个片段")
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    model = build_model(cfg)
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info(f"已加载检查点: {args.checkpoint}（轮次 {ckpt.get('epoch', '?')}）")

    evaluator = Evaluator(model, test_loader, cfg, logger)
    results = evaluator.evaluate(save_cm=True)
    logger.info("完成。")
    return results


if __name__ == "__main__":
    main()
