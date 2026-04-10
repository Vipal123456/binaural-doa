#!/usr/bin/env python3
"""在测试集上评估已训练的双耳 DOA-Net。

用法:
    python evaluate.py --checkpoint outputs/checkpoints/best.pth
    python evaluate.py --checkpoint outputs/checkpoints/best.pth --config configs/default.yaml
"""

import argparse

from torch.utils.data import DataLoader

from utils.config import load_config
from utils.seed import set_seed
from utils.logger import setup_logger
from utils.checkpoint import load_checkpoint
from models.binaural_doa_net import build_model
from engine.evaluator import Evaluator
from dataset.static_dataset import build_static_datasets


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--checkpoint", type=str, required=True,
                            help="已训练检查点的路径（.pth 文件）")
    pre_args, remaining = pre_parser.parse_known_args()

    cfg = load_config("configs/default.yaml", remaining)
    set_seed(cfg.dataset.split_seed)
    logger = setup_logger("DOA-eval", cfg.output.log_dir)

    # ---- 数据 ----
    _, _, test_ds = build_static_datasets(cfg)
    logger.info(f"测试集: {len(test_ds)} 个片段")

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    # ---- 模型 ----
    model = build_model(cfg)
    ckpt = load_checkpoint(pre_args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info(f"已加载检查点: {pre_args.checkpoint}（轮次 {ckpt.get('epoch', '?')}）")

    # ---- 评估 ----
    evaluator = Evaluator(model, test_loader, cfg, logger)
    results = evaluator.evaluate(save_cm=True)

    logger.info("完成。")
    return results


if __name__ == "__main__":
    main()
