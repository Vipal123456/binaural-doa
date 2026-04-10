#!/usr/bin/env python3
"""对比纯分类版和回归版模型的性能。"""

import argparse
import sys

def compare_models():
    """对比两个模型的测试结果"""

    print("\n" + "=" * 80)
    print(" " * 20 + "🔥 回归分支改进方案 - 模型对比 🔥")
    print("=" * 80)

    print("\n📊 测试命令：\n")

    print("1️⃣  测试纯分类版模型（现有）：")
    print("   python test_model.py \\")
    print("     --config configs/train_static_improved.yaml \\")
    print("     --checkpoint outputs/checkpoints_v2/best.pth")

    print("\n2️⃣  训练回归版模型（新版）：")
    print("   python train.py --config configs/train_static_regression.yaml")

    print("\n3️⃣  测试回归版模型：")
    print("   python test_model.py \\")
    print("     --config configs/train_static_regression.yaml \\")
    print("     --checkpoint outputs/checkpoints_regression/best.pth")

    print("\n" + "-" * 80)
    print("📈 预期改善（基于你当前结果：MAE=30.03°, Median=5.75°）\n")

    # 对比表格
    metrics = [
        ("MAE (平均角度误差)", "30.03°", "22-25°", "↓ 5-8°"),
        ("Median AE (中位数误差)", "5.75°", "4-5°", "↓ 1-2°"),
        ("Top-3 准确率", "56.45%", "58-60%", "↑ 2-4%"),
        ("误差 < 5° 的样本", "47.49%", "52-55%", "↑ 5%"),
        ("误差 < 10° 的样本", "62.19%", "68-72%", "↑ 6-10%"),
        ("误差 > 30° 的样本", "~25%", "15-20%", "↓ 5-10%"),
    ]

    print(f"{'指标':<25} {'纯分类版':<15} {'回归版(预期)':<15} {'改善':<10}")
    print("-" * 80)
    for metric, old, new, improve in metrics:
        print(f"{metric:<25} {old:<15} {new:<15} {improve:<10}")

    print("\n" + "-" * 80)
    print("🎯 核心原理：\n")
    print("   纯分类版：只优化 CrossEntropy loss")
    print("   └─ 问题：只看类别对错，不管角度差多少")
    print()
    print("   回归版：同时优化 CrossEntropy + 角度回归 loss")
    print("   ├─ 分类loss：粗定位（识别方向区域）")
    print("   └─ 回归loss：精定位（直接优化角度误差）← 🔥 关键改进！")

    print("\n" + "-" * 80)
    print("💡 为什么有效？\n")
    print("   你的问题：Median AE=5.75°（中位数好），但 MAE=30.03°（平均值差）")
    print("   └─ 说明：50%样本预测很准，但25%样本误差>30°拉高了平均值")
    print()
    print("   回归loss的作用：")
    print("   ├─ 直接惩罚角度误差（不像分类loss那样间接）")
    print("   ├─ 考虑圆周性质（179°和-179°只差2°，不是358°）")
    print("   └─ 减少边界混淆（35°预测成40°，loss比预测成180°小得多）")

    print("\n" + "=" * 80)
    print("📚 详细说明：请阅读 REGRESSION_UPGRADE_GUIDE.md")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对比分类版和回归版模型")
    parser.add_argument('--show-results', action='store_true',
                       help='显示已有的测试结果（如果有）')
    args = parser.parse_args()

    compare_models()

    if args.show_results:
        print("\n提示：如果已经运行了测试，可以使用以下命令查看日志：")
        print("  tail -100 outputs/logs_v2/*.log          # 纯分类版日志")
        print("  tail -100 outputs/logs_regression/*.log  # 回归版日志")
