#!/usr/bin/env python3
"""调试回归预测，查看回归头输出的分布。"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.config import load_config
from utils.checkpoint import load_checkpoint
from models.binaural_doa_net import build_model
from dataset.static_dataset import build_static_datasets


def main():
    # 加载配置
    cfg = load_config('configs/train_static_regression.yaml', [])

    # 构建数据集
    _, _, test_ds = build_static_datasets(cfg)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    # 加载模型
    model = build_model(cfg)
    ckpt = load_checkpoint('outputs/checkpoints_regression/best.pth', map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()

    # 收集预测
    all_pred_angles = []
    all_true_angles = []
    all_pred_bins = []
    all_true_bins = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 10:  # 只看前10个batch
                break

            out = model(batch)

            # 回归预测 (弧度)
            pred_angle_rad = out['angle']
            pred_angle_deg = torch.rad2deg(pred_angle_rad).numpy()

            # 分类预测
            pred_bin = out['logits'].argmax(dim=-1).numpy()

            # 真实值
            true_angle_deg = batch['azimuth_deg'].numpy()
            true_bin = batch['azimuth_label'].numpy()

            all_pred_angles.extend(pred_angle_deg)
            all_true_angles.extend(true_angle_deg)
            all_pred_bins.extend(pred_bin)
            all_true_bins.extend(true_bin)

    all_pred_angles = np.array(all_pred_angles)
    all_true_angles = np.array(all_true_angles)
    all_pred_bins = np.array(all_pred_bins)

    # 分类bin对应的角度
    def bin_to_angle(bin_idx):
        return -180 + (bin_idx + 0.5) * (360 / 72)

    pred_bin_angles = np.array([bin_to_angle(b) for b in all_pred_bins])

    print("=" * 60)
    print("回归预测分析（前320个样本）")
    print("=" * 60)

    print("\n【回归预测统计】")
    print(f"  均值: {all_pred_angles.mean():.2f}°")
    print(f"  标准差: {all_pred_angles.std():.2f}°")
    print(f"  最小值: {all_pred_angles.min():.2f}°")
    print(f"  最大值: {all_pred_angles.max():.2f}°")
    print(f"  中位数: {np.median(all_pred_angles):.2f}°")

    print("\n【真实角度统计】")
    print(f"  均值: {all_true_angles.mean():.2f}°")
    print(f"  标准差: {all_true_angles.std():.2f}°")
    print(f"  最小值: {all_true_angles.min():.2f}°")
    print(f"  最大值: {all_true_angles.max():.2f}°")

    # 计算误差
    def angular_error(pred, true):
        diff = pred - true
        diff = (diff + 180) % 360 - 180
        return np.abs(diff)

    reg_error = angular_error(all_pred_angles, all_true_angles)
    cls_error = angular_error(pred_bin_angles, all_true_angles)

    print("\n【回归预测误差】")
    print(f"  MAE: {reg_error.mean():.2f}°")
    print(f"  Median AE: {np.median(reg_error):.2f}°")
    print(f"  误差<5°: {(reg_error < 5).mean() * 100:.1f}%")
    print(f"  误差<10°: {(reg_error < 10).mean() * 100:.1f}%")

    print("\n【分类bin预测误差】")
    print(f"  MAE: {cls_error.mean():.2f}°")
    print(f"  Median AE: {np.median(cls_error):.2f}°")
    print(f"  误差<5°: {(cls_error < 5).mean() * 100:.1f}%")
    print(f"  误差<10°: {(cls_error < 10).mean() * 100:.1f}%")

    print("\n【前10个样本对比】")
    print(f"{'真实角度':>10} | {'回归预测':>10} | {'回归误差':>10} | {'分类bin':>10} | {'分类误差':>10}")
    print("-" * 60)
    for i in range(10):
        print(f"{all_true_angles[i]:>9.2f}° | {all_pred_angles[i]:>9.2f}° | "
              f"{reg_error[i]:>9.2f}° | {pred_bin_angles[i]:>9.2f}° | {cls_error[i]:>9.2f}°")

    # 检查回归预测是否卡在某个值
    unique_preds = np.unique(all_pred_angles)
    print(f"\n【回归预测唯一值数量】: {len(unique_preds)} / {len(all_pred_angles)}")
    if len(unique_preds) < 10:
        print(f"  警告：回归预测几乎不变化！唯一值: {unique_preds}")


if __name__ == '__main__':
    main()
