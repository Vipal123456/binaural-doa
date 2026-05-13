#!/usr/bin/env python3
"""比较两个实验的评估日志，并输出改进幅度。"""

import argparse
import re
from pathlib import Path


METRIC_KEYS = [
    "accuracy",
    "f1_score",
    "mean_angular_error",
    "median_angular_error",
    "std_angular_error",
    "acc_at_5deg",
    "acc_at_10deg",
    "acc_at_20deg",
    "acc_at_30deg",
    "within_1bin_acc",
    "front_back_halfplane_error_rate",
    "opposite_error_rate",
]

LOWER_IS_BETTER = {
    "mean_angular_error",
    "median_angular_error",
    "std_angular_error",
    "front_back_halfplane_error_rate",
    "opposite_error_rate",
}


def parse_metrics(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    metrics = {}
    for k in METRIC_KEYS:
        # 只提取最后一次出现（通常是最终结果）
        matches = re.findall(rf"\b{k}:\s*([-+]?\d*\.?\d+)", text)
        if matches:
            metrics[k] = float(matches[-1])
    return metrics


def fmt(v: float) -> str:
    return f"{v:.4f}"


def improvement_text(key: str, base: float, new: float) -> str:
    delta = new - base
    if key in LOWER_IS_BETTER:
        rel = ((base - new) / base * 100.0) if base != 0 else 0.0
        return f"{delta:+.4f} ({rel:.2f}% 降低)"
    rel = (delta / base * 100.0) if base != 0 else 0.0
    return f"{delta:+.4f} ({rel:.2f}% 提升)"


def compare(base_log: Path, new_log: Path) -> None:
    base_metrics = parse_metrics(base_log)
    new_metrics = parse_metrics(new_log)

    print("\n" + "=" * 96)
    print("模型对比结果")
    print("=" * 96)
    print(f"Baseline: {base_log}")
    print(f"New     : {new_log}")
    print("-" * 96)
    print(f"{'Metric':<24} {'Baseline':>12} {'New':>12} {'Delta':>26}")
    print("-" * 96)

    for key in METRIC_KEYS:
        if key not in base_metrics or key not in new_metrics:
            print(f"{key:<24} {'N/A':>12} {'N/A':>12} {'缺失':>26}")
            continue
        b = base_metrics[key]
        n = new_metrics[key]
        print(f"{key:<24} {fmt(b):>12} {fmt(n):>12} {improvement_text(key, b, n):>26}")

    print("=" * 96 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="比较两个评估日志的指标")
    parser.add_argument(
        "--base_log",
        type=Path,
        default=Path("outputs/logs_multisubject_robust50h_v5_bias_gating_attnpool_nocsl_enhanced_fbaux_only_cohfix_test_best_workers4/train.log"),
        help="基线实验评估日志路径",
    )
    parser.add_argument(
        "--new_log",
        type=Path,
        default=Path("outputs/logs_multisubject_robust50h_sdel_doa_cls_fbaux_nw8_gpu1_test_best_workers8/train.log"),
        help="新实验评估日志路径",
    )
    args = parser.parse_args()

    if not args.base_log.exists():
        raise FileNotFoundError(f"Baseline log not found: {args.base_log}")
    if not args.new_log.exists():
        raise FileNotFoundError(f"New log not found: {args.new_log}")

    compare(args.base_log, args.new_log)
