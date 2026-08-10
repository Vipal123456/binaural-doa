#!/usr/bin/env python3
"""按 SNR 分组评估模型鲁棒性。

用法:
    python tools/analyze_snr_robustness.py \
        --checkpoint outputs/checkpoints_static_hybridbrir_gate2_50h_v1_v7_dualcue/best.pth \
        --config configs/train_librispeech_multisubject_static_hybridbrir_gate2_50h_v1_v7_dualcue.yaml \
        --report_root data/librispeech_cipic_multisubject_static_hybridbrir_gate2_50h_v1
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.static_dataset import build_static_datasets
from engine.evaluator import Evaluator
from models.binaural_doa_net import build_model
from utils.angle import bins_to_angles, angular_error
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.logger import setup_logger
from utils.seed import set_seed


def load_snr_map(report_root: Path) -> dict:
    """从 mixing_report.csv 加载 file_id -> snr_db 映射."""
    snr_map = {}
    for split_dir in ["train_subjects", "val_subjects", "test_subjects_unseen"]:
        report_path = report_root / split_dir / "mixing_report.csv"
        if not report_path.is_file():
            continue
        with open(report_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                snr_map[row["file_id"]] = float(row["snr_db"])
    return snr_map


@torch.no_grad()
def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--checkpoint", type=str, required=True)
    pre_parser.add_argument("--report_root", type=str, required=True,
                            help="数据集根目录（包含 mixing_report.csv）")
    pre_args, remaining = pre_parser.parse_known_args()

    cfg = load_config("configs/default.yaml", remaining)
    set_seed(cfg.dataset.split_seed)
    logger = setup_logger("SNR-analysis", cfg.output.log_dir)

    snr_map = load_snr_map(Path(pre_args.report_root))
    logger.info(f"加载 SNR 映射: {len(snr_map)} 条记录")

    _, _, test_ds = build_static_datasets(cfg)
    logger.info(f"测试集: {len(test_ds)} 个片段")

    test_loader = DataLoader(
        test_ds, batch_size=cfg.train.batch_size,
        shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True,
    )

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(pre_args.checkpoint, map_location=str(device))
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(f"加载 checkpoint epoch={ckpt.get('epoch','?')}")

    num_classes = cfg.model.num_classes
    az_range = tuple(cfg.model.azimuth_range)

    # 收集 per-segment 结果
    pred_bins_all = []
    true_labels_all = []
    snrs_all = []

    for batch in test_loader:
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items()}
        out = model(batch_gpu)
        logits = out["logits"].float().cpu().numpy()
        pred_bins = logits.argmax(axis=-1)

        for i, fid in enumerate(batch["file_id"]):
            snr = snr_map.get(fid, float("nan"))
            pred_bins_all.append(pred_bins[i])
            true_labels_all.append(batch["azimuth_label"][i].item())
            snrs_all.append(snr)

    pred_bins_all = np.array(pred_bins_all)
    true_labels_all = np.array(true_labels_all)
    snrs_all = np.array(snrs_all)

    # 转换为角度
    pred_angles = bins_to_angles(pred_bins_all, num_classes, az_range)
    true_angles = bins_to_angles(true_labels_all, num_classes, az_range)
    ang_errs = angular_error(pred_angles, true_angles)

    # ---- SNR bins ----
    snr_bins = [(-10, -5), (-5, 0), (0, 5), (5, 10)]
    logger.info("=" * 70)
    logger.info(f"{'SNR Range':<16} {'N':>6} {'MAE':>8} {'Acc@5°':>8} {'Acc@10°':>8} {'FB Err':>8}")
    logger.info("-" * 70)

    for lo, hi in snr_bins:
        mask = (snrs_all >= lo) & (snrs_all < hi)
        if mask.sum() == 0:
            continue
        mae = ang_errs[mask].mean()
        acc5 = (ang_errs[mask] <= 5).mean()
        acc10 = (ang_errs[mask] <= 10).mean()
        # front/back error: |angle|>90 is back
        pred_fb = np.abs(pred_angles[mask]) > 90
        true_fb = np.abs(true_angles[mask]) > 90
        fb_err = (pred_fb != true_fb).mean()
        logger.info(f"SNR [{lo:3d},{hi:2d}) dB  {mask.sum():>6d}  {mae:>7.2f}° {acc5:>7.1%}  {acc10:>7.1%}  {fb_err:>7.1%}")

    # ---- Overall ----
    logger.info("-" * 70)
    logger.info(f"{'Overall':<16} {len(ang_errs):>6d}  {ang_errs.mean():>7.2f}° {(ang_errs<=5).mean():>7.1%}  {(ang_errs<=10).mean():>7.1%}")

    # ---- Per noise scene ----
    # Also group by DEMAND scene if available
    logger.info("")
    logger.info("=" * 70)
    logger.info("Per DEMAND scene:")
    scene_map = {}
    for split_dir in ["train_subjects", "val_subjects", "test_subjects_unseen"]:
        report_path = Path(pre_args.report_root) / split_dir / "mixing_report.csv"
        if not report_path.is_file():
            continue
        with open(report_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                scene_map[row["file_id"]] = row.get("demand_scene", "unknown")

    scenes_all = np.array([scene_map.get(fid, "unknown") for fid in
                           [test_ds.segments[i]["file_id"] for i in range(len(test_ds))]])

    logger.info(f"{'Scene':<16} {'N':>6} {'MAE':>8} {'Acc@5°':>8} {'Acc@10°':>8}")
    logger.info("-" * 70)
    for scene in sorted(set(scenes_all)):
        mask = scenes_all == scene
        if mask.sum() == 0:
            continue
        mae = ang_errs[mask].mean()
        acc5 = (ang_errs[mask] <= 5).mean()
        acc10 = (ang_errs[mask] <= 10).mean()
        logger.info(f"{scene:<16} {mask.sum():>6d}  {mae:>7.2f}° {acc5:>7.1%}  {acc10:>7.1%}")

    logger.info("完成。")


if __name__ == "__main__":
    main()
