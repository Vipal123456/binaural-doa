#!/usr/bin/env python3
"""测试训练好的DOA模型在测试集上的性能。"""

import argparse
import csv
import logging
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import build_model
from metrics import DOAMetrics
from utils.angle import angular_error, bins_to_angles, wrap_angles
from utils.config import load_config
from utils.checkpoint import load_checkpoint
from dataset.static_dataset import build_static_datasets


def setup_logger():
    """设置日志记录器。"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def _compute_output_dir(cfg, checkpoint_path: str) -> Path:
    log_dir = Path(cfg.output.log_dir)
    if str(log_dir):
        return log_dir
    return Path(checkpoint_path).resolve().parent


def _export_predictions_csv(rows, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_id",
        "true_label",
        "pred_label",
        "true_deg",
        "pred_deg",
        "angular_error_deg",
        "front_back_halfplane_error",
        "opposite_error",
        "large_error",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_model_state_compat(model, state_dict, logger):
    """兼容历史 checkpoint 中 temporal head 命名。"""
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError as e:
        msg = str(e)
        needs_temporal_rename = (
            "temporal_head.temporal_encoder" in msg
            and ("temporal_head.gru" in msg or "temporal_head.lstm" in msg)
        )
        if not needs_temporal_rename:
            raise

    renamed = {}
    for k, v in state_dict.items():
        if k.startswith("temporal_head.gru."):
            renamed[k.replace("temporal_head.gru.", "temporal_head.temporal_encoder.", 1)] = v
        elif k.startswith("temporal_head.lstm."):
            renamed[k.replace("temporal_head.lstm.", "temporal_head.temporal_encoder.", 1)] = v
        else:
            renamed[k] = v

    logger.info("检测到历史 temporal head 命名，已自动兼容旧 checkpoint 键名。")
    model.load_state_dict(renamed)


def test_model(model, test_loader, cfg, device, logger):
    """在测试集上评估模型。

    参数
    ----------
    model : nn.Module
        训练好的模型。
    test_loader : DataLoader
        测试数据加载器。
    cfg : Config
        配置对象。
    device : torch.device
        计算设备。
    logger : logging.Logger
        日志记录器。

    返回
    -------
    dict
        包含各种评估指标的字典。
    """
    model.eval()
    metrics = DOAMetrics(
        num_classes=cfg.model.num_classes,
        azimuth_range=tuple(cfg.model.azimuth_range)
    )
    metrics.reset()

    logger.info(f"开始在测试集上评估... (共 {len(test_loader)} 个batch)")

    prediction_rows = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            file_ids = batch["file_id"]
            # 将数据移到设备
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            # 前向传播
            out = model(batch)
            logits = out["logits"]  # [B, num_classes]
            labels = batch["azimuth_label"]  # [B]
            azimuths_deg = batch["azimuth_deg"]  # [B]

            # 更新指标
            logits_np = logits.float().cpu().numpy()
            labels_np = labels.cpu().numpy()
            azimuths_deg_np = azimuths_deg.cpu().numpy() if isinstance(azimuths_deg, torch.Tensor) else azimuths_deg
            metrics.update(logits_np, labels_np, azimuths_deg_np)

            pred_bins_np = logits_np.argmax(axis=-1)
            pred_degs_np = bins_to_angles(pred_bins_np, cfg.model.num_classes, tuple(cfg.model.azimuth_range))
            ang_err_np = angular_error(pred_degs_np, azimuths_deg_np)
            true_fb = (abs(wrap_angles(azimuths_deg_np)) > 90.0).astype(int)
            pred_fb = (abs(wrap_angles(pred_degs_np)) > 90.0).astype(int)
            circular_bin_diff = abs(pred_bins_np - labels_np)
            circular_bin_diff = circular_bin_diff.clip(min=0)
            circular_bin_diff = np.minimum(circular_bin_diff, cfg.model.num_classes - circular_bin_diff)
            half_bins = cfg.model.num_classes // 2

            for i, file_id in enumerate(file_ids):
                prediction_rows.append({
                    "file_id": str(file_id),
                    "true_label": int(labels_np[i]),
                    "pred_label": int(pred_bins_np[i]),
                    "true_deg": float(azimuths_deg_np[i]),
                    "pred_deg": float(pred_degs_np[i]),
                    "angular_error_deg": float(ang_err_np[i]),
                    "front_back_halfplane_error": int(pred_fb[i] != true_fb[i]),
                    "opposite_error": int(abs(int(circular_bin_diff[i]) - half_bins) <= 1),
                    "large_error": int(float(ang_err_np[i]) >= 45.0),
                })

            if (batch_idx + 1) % 20 == 0:
                logger.info(f"  处理进度: {batch_idx + 1}/{len(test_loader)}")

    # 计算指标
    results = metrics.compute()

    return results, prediction_rows


def main():
    parser = argparse.ArgumentParser(description="测试DOA模型")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_static.yaml",
        help="训练配置文件路径"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best_model.pth",
        help="模型checkpoint路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="计算设备 (cuda/cpu)"
    )
    parser.add_argument(
        "--predictions-csv",
        type=str,
        default="",
        help="可选：样本级预测结果输出路径"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="可选：覆盖测试时 DataLoader 的 num_workers；-1 表示沿用配置"
    )
    args = parser.parse_args()

    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("DOA模型测试")
    logger.info("=" * 60)

    # 加载配置
    logger.info(f"加载配置: {args.config}")
    cfg = load_config(args.config, [])

    # 构建数据集
    logger.info("构建数据集...")
    dataset_type = cfg.dataset.get("dataset_type", "static")
    if dataset_type == "static":
        _, _, test_ds = build_static_datasets(cfg)
    else:
        logger.error(f"不支持的数据集类型: {dataset_type}")
        sys.exit(1)

    logger.info(f"测试集: {len(test_ds)} 个样本")

    # 创建数据加载器
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=(cfg.train.num_workers if args.num_workers < 0 else args.num_workers),
        pin_memory=True
    )

    # 构建模型
    logger.info("构建模型...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)

    # 加载checkpoint
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint文件不存在: {args.checkpoint}")
        sys.exit(1)

    logger.info(f"加载checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint, map_location=str(device))
    _load_model_state_compat(model, ckpt["model"], logger)

    epoch = ckpt.get("epoch", "未知")
    best_mae = ckpt.get("best_mae", "未知")
    logger.info(f"  Epoch: {epoch}, Best MAE: {best_mae}")

    # 测试模型
    logger.info("\n" + "=" * 60)
    logger.info("开始测试...")
    logger.info("=" * 60)

    results, prediction_rows = test_model(model, test_loader, cfg, device, logger)

    # 打印结果
    logger.info("\n" + "=" * 60)
    logger.info("测试结果")
    logger.info("=" * 60)
    logger.info(f"  准确率 (Accuracy):           {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    logger.info(f"  F1-score:                    {results['f1_score']:.4f} ({results['f1_score']*100:.2f}%)")
    logger.info(f"  平均角度误差 (MAE):          {results['mean_angular_error']:.2f}°")
    logger.info(f"  中位数角度误差 (Median AE):  {results['median_angular_error']:.2f}°")
    logger.info(f"  标准差:                      {results['std_angular_error']:.2f}°")
    logger.info(f"  最大误差:                    {results['max_angular_error']:.2f}°")
    logger.info("=" * 60)

    # 分析误差分布
    logger.info("\n误差分布分析:")
    logger.info(f"  Acc@5°:      {results.get('acc_at_5deg', 0):.4f} ({results.get('acc_at_5deg', 0)*100:.2f}%)")
    logger.info(f"  Acc@10°:     {results.get('acc_at_10deg', 0):.4f} ({results.get('acc_at_10deg', 0)*100:.2f}%)")
    logger.info(f"  Acc@20°:     {results.get('acc_at_20deg', 0):.4f} ({results.get('acc_at_20deg', 0)*100:.2f}%)")
    logger.info(f"  Acc@30°:     {results.get('acc_at_30deg', 0):.4f} ({results.get('acc_at_30deg', 0)*100:.2f}%)")
    logger.info(f"  Half-plane Err: {results.get('front_back_halfplane_error_rate', 0):.4f}")
    logger.info(f"  Opposite Err:   {results.get('opposite_error_rate', 0):.4f}")
    logger.info(f"  Within-1-bin Acc: {results.get('within_1bin_acc', 0):.4f}")
    logger.info("=" * 60)

    output_dir = _compute_output_dir(cfg, args.checkpoint)
    predictions_csv = Path(args.predictions_csv) if args.predictions_csv else output_dir / "predictions.csv"
    _export_predictions_csv(prediction_rows, predictions_csv)
    logger.info(f"已保存样本级预测结果: {predictions_csv}")


if __name__ == "__main__":
    main()
