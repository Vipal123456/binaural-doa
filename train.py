#!/usr/bin/env python3
"""训练双耳 DOA-Net。

用法:
    python train.py                                # 使用 configs/default.yaml
    python train.py --config configs/my.yaml       # 自定义配置
    python train.py --train.lr 0.0005 --train.epochs 50  # 命令行参数覆盖
    python train.py --resume outputs/checkpoints/latest.pth  # 恢复训练
"""

import argparse

from torch.utils.data import DataLoader

from utils.config import load_config
from utils.seed import set_seed
from utils.logger import setup_logger
from models.binaural_doa_net import build_model
from engine.trainer import Trainer
from dataset.static_dataset import build_static_datasets


def main():
    # ---- 单独解析 --resume 参数（它不是配置文件中的字段） ----
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--resume", type=str, default=None,
                            help="要恢复训练的检查点路径")
    pre_args, remaining = pre_parser.parse_known_args()

    # ---- 加载配置并应用命令行覆盖 ----
    cfg = load_config("configs/default.yaml", remaining)

    # ---- 随机种子 ----
    set_seed(cfg.dataset.split_seed)

    # ---- 日志记录器 ----
    logger = setup_logger("DOA-net", cfg.output.log_dir)
    logger.info("双耳 DOA-Net 训练开始")
    logger.info(f"Config: {cfg.to_dict()}")

    # ---- 保存解析后的配置 ----
    cfg.save_yaml(f"{cfg.output.log_dir}/resolved_config.yaml")

    # ---- 数据 ----
    logger.info("正在构建数据集...")
    train_ds, val_ds, test_ds = build_static_datasets(cfg, logger=logger)
    logger.info(f"训练集: {len(train_ds)} 个片段 | 验证集: {len(val_ds)} | 测试集: {len(test_ds)}")

    t = cfg.train
    train_loader = DataLoader(
        train_ds,
        batch_size=t.batch_size,
        shuffle=True,
        num_workers=t.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=t.batch_size,
        shuffle=False,
        num_workers=t.num_workers,
        pin_memory=True,
    )

    # ---- 模型 ----
    model = build_model(cfg)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型构建完成 -- {num_params:,} 个可训练参数")

    # ---- 训练器 ----
    trainer = Trainer(model, train_loader, val_loader, cfg, logger)

    if pre_args.resume is not None:
        trainer.resume(pre_args.resume)

    # ---- 开始训练 ----
    trainer.fit()


if __name__ == "__main__":
    main()
