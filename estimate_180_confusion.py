#!/usr/bin/env python3
"""
基于MAE和Median推算180°前后混淆的比例
不需要加载模型，纯数学推导，无外部依赖
"""

def estimate_confusion_from_stats(mae, median, total_samples=3750):
    """
    基于MAE和Median估算前后混淆的比例

    假设:
    - 50%的样本误差 < median (约5.47°)
    - 剩余50%中，部分是正常误差(10-60°)，部分是前后混淆(~180°)

    参数:
        mae: 平均绝对误差 (度)
        median: 中位数误差 (度)
        total_samples: 总样本数
    """

    print("="*80)
    print("前后混淆比例估算 (基于MAE和Median)")
    print("="*80)

    print(f"\n已知数据:")
    print(f"  MAE (平均误差):     {mae:.2f}°")
    print(f"  Median (中位数):    {median:.2f}°")
    print(f"  验证集样本数:       {total_samples}")
    print(f"  MAE/Median比值:     {mae/median:.2f}")

    # 场景1: 假设无前后混淆，误差服从正态分布
    print(f"\n场景1: 假设无前后混淆 (理论对比)")
    print(f"  正态分布下，MAE/Median ≈ 1.25-1.50")
    print(f"  ❌ 实际 MAE/Median = {mae/median:.2f}，远超正态分布")
    print(f"  → 结论: 存在显著的离群大误差")

    # 场景2: 估算前后混淆比例
    print(f"\n场景2: 估算前后混淆比例")

    # 假设误差分布:
    # - 50% 样本误差 < 5.47° (中位数以下)，平均约 3°
    # - 40% 样本误差在 5.47-30° (正常范围)，平均约 15°
    # - 10% 样本误差 > 30° (可能包含前后混淆)

    # 尝试不同的前后混淆比例
    for confusion_ratio in [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15]:
        n_confusion = int(total_samples * confusion_ratio)
        n_normal = total_samples - n_confusion

        # 假设正常样本的误差分布
        # 50% < median, 平均 3°
        # 50% > median, 平均 15°
        normal_below_median = int(n_normal * 0.5)
        normal_above_median = n_normal - normal_below_median

        avg_below = 3.0
        avg_above = 15.0

        # 前后混淆样本误差约 175-180°
        avg_confusion = 177.0

        # 计算加权MAE
        estimated_mae = (
            (normal_below_median * avg_below +
             normal_above_median * avg_above +
             n_confusion * avg_confusion) / total_samples
        )

        diff = abs(estimated_mae - mae)

        if diff < 1.0:  # 误差小于1°
            print(f"  ✓ 前后混淆比例 {confusion_ratio*100:.1f}% ({n_confusion}样本):")
            print(f"    - 估算MAE = {estimated_mae:.2f}° (误差 {diff:.2f}°)")
            print(f"    - 正常样本: {n_normal} ({normal_below_median}个<{median:.1f}°, {normal_above_median}个>{median:.1f}°)")
            print(f"    - 混淆样本: {n_confusion} (误差~{avg_confusion:.0f}°)")

    # 更精细的二分搜索
    print(f"\n场景3: 精确计算最佳拟合")

    best_ratio = None
    best_diff = float('inf')

    # 使用纯Python循环替代numpy.arange
    ratio = 0.001
    while ratio < 0.20:
        n_confusion = int(total_samples * ratio)
        n_normal = total_samples - n_confusion

        normal_below = int(n_normal * 0.5)
        normal_above = n_normal - normal_below

        estimated_mae = (
            (normal_below * 3.0 +
             normal_above * 15.0 +
             n_confusion * 177.0) / total_samples
        )

        diff = abs(estimated_mae - mae)
        if diff < best_diff:
            best_diff = diff
            best_ratio = ratio

        ratio += 0.001

    n_best = int(total_samples * best_ratio)
    n_normal_best = total_samples - n_best

    print(f"  最佳拟合前后混淆比例: {best_ratio*100:.2f}% ({n_best} 样本)")
    print(f"  此时估算MAE: {mae:.2f}° (拟合误差: {best_diff:.3f}°)")

    # 场景4: 考虑更复杂的误差分布
    print(f"\n场景4: 更详细的误差分布估算")

    # 假设误差分布（基于一般DOA系统经验）:
    error_bins = [
        (0, 5, 0.45, 2.5),      # 45% 样本误差 0-5°, 平均 2.5°
        (5, 10, 0.20, 7.5),     # 20% 样本误差 5-10°, 平均 7.5°
        (10, 20, 0.15, 15.0),   # 15% 样本误差 10-20°, 平均 15°
        (20, 40, 0.10, 30.0),   # 10% 样本误差 20-40°, 平均 30°
        (40, 90, 0.05, 60.0),   # 5% 样本误差 40-90°, 平均 60°
    ]

    # 尝试不同的前后混淆比例
    confusion_ratio = 0.01
    step = 0.01
    while confusion_ratio < 0.15:
        n_confusion = int(total_samples * confusion_ratio)

        # 正常误差分布
        total_normal_mae = 0
        total_normal_count = 0

        for low, high, bin_ratio, avg in error_bins:
            n = int(total_samples * bin_ratio * (1 - confusion_ratio))
            total_normal_mae += n * avg
            total_normal_count += n

        # 加上前后混淆
        total_mae_sum = total_normal_mae + n_confusion * 177.0
        estimated_mae = total_mae_sum / total_samples

        diff = abs(estimated_mae - mae)

        if diff < 0.5:
            print(f"  ✓ 前后混淆比例 {confusion_ratio*100:.1f}% ({n_confusion}样本):")
            print(f"    估算MAE = {estimated_mae:.2f}° (误差 {diff:.2f}°)")

            # 打印详细分布
            print(f"    详细误差分布:")
            for low, high, bin_ratio, avg in error_bins:
                n = int(total_samples * bin_ratio * (1 - confusion_ratio))
                pct = n / total_samples * 100
                print(f"      [{low:3d}°-{high:3d}°): {n:4d} 样本 ({pct:5.2f}%), 平均{avg:5.1f}°")

            print(f"      [170°-180°): {n_confusion:4d} 样本 ({confusion_ratio*100:5.2f}%), 平均177.0° (前后混淆)")

        confusion_ratio += step

    print("\n" + "="*80)
    print("💡 结论:")
    print("="*80)

    if best_ratio > 0.02:
        print(f"  ⚠️  估算有 {best_ratio*100:.1f}% ({n_best}个) 样本存在前后混淆 (误差~180°)")
        print(f"  ⚠️  这些样本仅占 {best_ratio*100:.1f}%，但贡献了约 {n_best*177/total_samples/mae*100:.1f}% 的总误差")

        # 计算消除混淆后的MAE
        mae_without_confusion = (mae * total_samples - n_best * 177) / (total_samples - n_best)
        print(f"\n  📌 如果解决前后混淆问题:")
        print(f"     MAE可从 {mae:.2f}° 降至 {mae_without_confusion:.2f}° (改进 {(mae-mae_without_confusion)/mae*100:.1f}%)")
        print(f"     这将使性能接近SOTA水平 (22-25°)")

    print("\n  💡 测试假设:")
    print(f"     运行: python analyze_error_distribution.py")
    print(f"     查看实际的误差分布和最大误差值")
    print("="*80)

def main():
    # V2最佳模型的统计数据
    mae = 27.93
    median = 5.47
    total_samples = 3750

    estimate_confusion_from_stats(mae, median, total_samples)

if __name__ == '__main__':
    main()
