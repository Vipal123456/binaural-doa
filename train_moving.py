#!/usr/bin/env python3
"""Train moving single-speaker binaural DOA sequence model."""

import argparse

import torch
from torch.utils.data import DataLoader

from dataset.moving_dataset import build_moving_datasets
from engine.dynamic_trainer import DynamicTrainer
from models.binaural_doa_net import build_model
from utils.config import load_config
from utils.logger import setup_logger
from utils.seed import set_seed


def load_pretrained_modules(model, checkpoint_path: str, prefixes, logger) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    src_state = ckpt.get("model", ckpt)
    dst_state = model.state_dict()
    allowed = tuple(f"{p}." for p in prefixes)
    matched = {}
    skipped = []
    for key, value in src_state.items():
        if not key.startswith(allowed):
            continue
        if key in dst_state and tuple(dst_state[key].shape) == tuple(value.shape):
            matched[key] = value
        else:
            skipped.append(key)
    dst_state.update(matched)
    model.load_state_dict(dst_state)
    logger.info(
        f"Loaded moving pretrained modules from {checkpoint_path}: "
        f"matched={len(matched)} skipped={len(skipped)} prefixes={list(prefixes)}"
    )


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--resume", type=str, default=None)
    args, remaining = pre.parse_known_args()
    cfg = load_config("configs/default.yaml", remaining)
    set_seed(cfg.dataset.split_seed)
    logger = setup_logger("moving-doa", cfg.output.log_dir)
    logger.info("Moving DOA sequence training start")
    logger.info(f"Config: {cfg.to_dict()}")
    cfg.save_yaml(f"{cfg.output.log_dir}/resolved_config.yaml")

    train_ds, val_ds, _ = build_moving_datasets(cfg, logger=logger)
    logger.info(f"train={len(train_ds)} val={len(val_ds)}")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    model = build_model(cfg)
    pretrain_path = getattr(cfg.train, "pretrained_checkpoint", "")
    if pretrain_path and not args.resume:
        prefixes = getattr(
            cfg.train,
            "pretrained_prefixes",
            ["encoder", "content_fusion", "cue_encoder", "fusion_norm"],
        )
        load_pretrained_modules(model, pretrain_path, prefixes, logger)
    logger.info(f"model params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    trainer = DynamicTrainer(model, train_loader, val_loader, cfg, logger)
    if args.resume:
        trainer.resume(args.resume)
    trainer.fit()


if __name__ == "__main__":
    main()
