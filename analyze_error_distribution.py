#!/usr/bin/env python3
"""
分析V2最佳模型的详细误差分布
运行: python analyze_error_distribution.py
"""
import torch
import numpy as np
from pathlib import Path
from utils.config import load_config
from dataset.static_dataset import build_static_datasets
from models.binaural_doa_net import build_model
from metrics import DOAMetrics
from utils.angle import bins_to_angles, angular_error

def load_model_and_data(config_path, checkpoint_path):
    """加载模型和数据集"""
    # 加载配置
    cfg = load_config(config_path, [])

    # 构建数据集
    _, val_dataset, _ = build_static_datasets(cfg)

    # 构建模型
    model = build_model(cfg).cuda()

    # 加载权重
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt['model'])
    model.eval()

    return model, val_dataset, cfg

def analyze_predictions(model, dataset, cfg):
    """分析所有预测的误差分布"""
    metrics = DOAMetrics(
        num_classes=cfg.model.num_classes,
        azimuth_range=tuple(cfg.model.azimuth_range),
        top_k=3
    )

    all_errors = []
    all_pred_degs = []
    all_true_degs = []

    print("正在分析验证集预测...")
    with torch.no_grad():
        for i in range(len(dataset)):
            batch = dataset[i]

            # 准备输入
            inputs = {
                'magnitude': batch['magnitude'].unsqueeze(0).cuda(),
                'ipd': batch['ipd'].unsqueeze(0).cuda(),
                'ild': batch['ild'].unsqueeze(0).cuda()
            }

            # 预测
            out = model(inputs)
            logits = out['logits'].cpu().numpy()  # [1, 72]

            # 获取预测bin和真实bin
            pred_bin = logits.argmax()
            true_bin = batch['azimuth_label'].item()
            true_deg = batch['azimuth_deg'].item()

            # 转换为角度
            pred_deg = bins_to_angles(
                np.array([pred_bin]),
                cfg.model.num_classes,
                tuple(cfg.model.azimuth_range)
            )[0]

            # 计算角度误差
            error = angular_error(
                np.array([pred_deg]),
                np.array([true_deg])
            )[0]

            all_errors.append(error)
            all_pred_degs.append(pred_deg)
            all_true_degs.append(true_deg)

            # 更新metrics
            metrics.update(
                logits,
                np.array([true_bin]),
                np.array([true_deg]),
                None
            )

            if (i + 1) % 500 == 0:
                print(f"  已处理 {i+1}/{len(dataset)} 样本")

    all_errors = np.array(all_errors)
    all_pred_degs = np.array(all_pred_degs)
    all_true_degs = np.array(all_true_degs)

    # 计算完整指标
    results = metrics.compute()

    return all_errors, all_pred_degs, all_true_degs, results

def print_error_analysis(errors, pred_degs, true_degs, results):
    """打印详细的误差分析"""
    print("\n" + "="*80)
    print("V2最佳模型 (Epoch 23) 详细误差分析")
    print("="*80)

    # 基础统计
    print(f"\n📊 基础统计:")
    print(f"  样本总数:      {len(errors)}")
    print(f"  MAE (平均):    {results['mean_angular_error']:.2f}°")
    print(f"  Median (中位数): {results['median_angular_error']:.2f}°")
    print(f"  Std (标准差):  {results['std_angular_error']:.2f}°")
    print(f"  Max (最大误差): {results['max_angular_error']:.2f}°")
    print(f"  Min (最小误差): {errors.min():.2f}°")

    # 误差分布百分位数
    print(f"\n📈 误差分布 (百分位数):")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(errors, p)
        count = (errors <= val).sum()
        ratio = count / len(errors) * 100
        print(f"  {p:2d}th: {val:6.2f}° (≤此误差的样本: {count:4d}, {ratio:.1f}%)")

    # 误差区间分布
    print(f"\n📉 误差区间分布:")
    bins = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 60), (60, 90), (90, 180)]
    for low, high in bins:
        count = ((errors >= low) & (errors < high)).sum()
        ratio = count / len(errors) * 100
        print(f"  [{low:3d}°, {high:3d}°): {count:4d} 样本 ({ratio:5.2f}%)")

    # 大误差样本分析
    print(f"\n⚠️  大误差样本分析 (>90°):")
    large_errors_idx = errors > 90
    large_errors_count = large_errors_idx.sum()
    print(f"  误差>90°的样本数: {large_errors_count} ({large_errors_count/len(errors)*100:.2f}%)")

    if large_errors_count > 0:
        large_errors = errors[large_errors_idx]
        large_pred = pred_degs[large_errors_idx]
        large_true = true_degs[large_errors_idx]

        # 检查是否是前后混淆（180°误差）
        near_180_count = (large_errors > 170).sum()
        print(f"  误差>170°的样本数 (接近前后混淆): {near_180_count} ({near_180_count/len(errors)*100:.2f}%)")

        print(f"\n  前10个最大误差样本:")
        worst_idx = np.argsort(errors)[-10:][::-1]
        for rank, idx in enumerate(worst_idx, 1):
            print(f"    #{rank}: 误差={errors[idx]:.1f}°, 预测={pred_degs[idx]:6.1f}°, 真实={true_degs[idx]:6.1f}°")

    # 分类性能
    print(f"\n🎯 分类性能:")
    print(f"  Top-1 准确率: {results['accuracy']*100:.2f}%")
    print(f"  Top-3 准确率: {results['top_k_accuracy']*100:.2f}%")
    print(f"  <5°误差占比:  {results['error_lt_5']*100:.2f}%")
    print(f"  <10°误差占比: {results['error_lt_10']*100:.2f}%")
    print(f"  <20°误差占比: {results['error_lt_20']*100:.2f}%")
    print(f"  <30°误差占比: {results['error_lt_30']*100:.2f}%")

    print("\n" + "="*80)

    # 结论
    print("\n💡 结论:")
    median_mae_ratio = results['mean_angular_error'] / results['median_angular_error']
    print(f"  MAE/Median比值: {median_mae_ratio:.2f}")

    if median_mae_ratio > 4:
        print(f"  ⚠️  MAE是Median的{median_mae_ratio:.1f}倍，说明存在显著的离群大误差")
    if near_180_count > 0:
        print(f"  ⚠️  发现{near_180_count}个接近180°的误差，可能是前后混淆问题")
        print(f"      这可能是由于双耳线索在对称位置相似导致的")

    # MAE贡献分析
    total_mae = results['mean_angular_error'] * len(errors)
    large_contrib = large_errors.sum() if large_errors_count > 0 else 0
    large_contrib_pct = large_contrib / total_mae * 100 if total_mae > 0 else 0

    if large_errors_count > 0:
        print(f"  📌 大误差(>90°)样本仅占{large_errors_count/len(errors)*100:.2f}%，但贡献了{large_contrib_pct:.1f}%的总误差")
        print(f"  📌 如果消除这些大误差，MAE可降至约{(total_mae - large_contrib)/len(errors):.2f}°")

    print("="*80)

def main():
    config_path = '/disk2/bywang/DOA-net/configs/train_static_improved.yaml'
    checkpoint_path = '/disk2/bywang/DOA-net/outputs/checkpoints_v2/best.pth'

    print("加载模型和数据集...")
    model, dataset, cfg = load_model_and_data(config_path, checkpoint_path)
    print(f"✓ 验证集样本数: {len(dataset)}")

    print("\n开始分析...")
    errors, pred_degs, true_degs, results = analyze_predictions(model, dataset, cfg)

    print_error_analysis(errors, pred_degs, true_degs, results)

    # 保存详细结果
    np.savez(
        '/disk2/bywang/DOA-net/error_analysis.npz',
        errors=errors,
        pred_degs=pred_degs,
        true_degs=true_degs
    )
    print("\n✅ 详细数据已保存至: error_analysis.npz")

if __name__ == '__main__':
    main()
