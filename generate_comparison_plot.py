#!/usr/bin/env python3
"""
V1 vs V2 训练对比可视化脚本
运行: python generate_comparison_plot.py
输出: V1_vs_V2_comparison.png
"""

import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def parse_log(log_path):
    """解析训练日志文件"""
    with open(log_path) as f:
        log_content = f.read()

    epochs = []
    train_loss = []
    val_mae = []
    val_med = []
    acc = []
    top3_acc = []

    for line in log_content.split('\n'):
        if 'train_loss=' in line and 'val  acc=' not in line:
            match = re.search(r'Epoch (\d+)/\d+\s+train_loss=([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                epochs.append(epoch)
                train_loss.append(loss)
        elif 'val  acc=' in line:
            match = re.search(r'acc=([\d.]+).*top3_acc=([\d.]+).*MAE=([\d.]+)°.*median_AE=([\d.]+)°', line)
            if match:
                acc.append(float(match.group(1)) * 100)
                top3_acc.append(float(match.group(2)) * 100)
                val_mae.append(float(match.group(3)))
                val_med.append(float(match.group(4)))

    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'val_mae': val_mae,
        'val_med': val_med,
        'acc': acc,
        'top3_acc': top3_acc,
    }

def main():
    # 解析日志
    print("正在解析日志文件...")
    v1 = parse_log('/disk2/bywang/DOA-net/outputs/logs/20260323_160247.log')
    v2 = parse_log('/disk2/bywang/DOA-net/outputs/logs_v2/20260323_174916.log')

    # 找最佳epoch
    v1_best_idx = np.argmin(v1['val_mae'])
    v1_best_epoch = v1['epochs'][v1_best_idx]
    v1_best_mae = v1['val_mae'][v1_best_idx]

    v2_best_idx = np.argmin(v2['val_mae'])
    v2_best_epoch = v2['epochs'][v2_best_idx]
    v2_best_mae = v2['val_mae'][v2_best_idx]

    print(f"V1最佳: Epoch {v1_best_epoch}, MAE={v1_best_mae:.2f}°")
    print(f"V2最佳: Epoch {v2_best_epoch}, MAE={v2_best_mae:.2f}°")

    # 创建2x2对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('V1 vs V2 训练对比分析', fontsize=18, fontweight='bold', y=0.995)

    # 1. 训练Loss对比
    ax = axes[0, 0]
    ax.plot(v1['epochs'], v1['train_loss'], 'o-', label='V1 (lr=0.001, no aug)',
            color='#E74C3C', alpha=0.8, linewidth=2.5, markersize=4)
    ax.plot(v2['epochs'], v2['train_loss'], 's-', label='V2 (lr=0.0005, w/ aug)',
            color='#3498DB', alpha=0.8, linewidth=2.5, markersize=4)
    ax.axvline(v1_best_epoch, color='#E74C3C', linestyle='--', alpha=0.4, linewidth=1.5)
    ax.axvline(v2_best_epoch, color='#3498DB', linestyle='--', alpha=0.4, linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax.set_title('(A) 训练Loss曲线', fontsize=13, fontweight='bold', pad=10)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-2, max(len(v1['epochs']), len(v2['epochs'])) + 2)

    # 2. 验证MAE对比
    ax = axes[0, 1]
    ax.plot(v1['epochs'], v1['val_mae'], 'o-', label=f'V1: Best={v1_best_mae:.2f}° @ E{v1_best_epoch}',
            color='#E74C3C', alpha=0.8, linewidth=2.5, markersize=4)
    ax.plot(v2['epochs'], v2['val_mae'], 's-', label=f'V2: Best={v2_best_mae:.2f}° @ E{v2_best_epoch}',
            color='#3498DB', alpha=0.8, linewidth=2.5, markersize=4)
    ax.axhline(v1_best_mae, color='#E74C3C', linestyle=':', alpha=0.4, linewidth=1.5)
    ax.axhline(v2_best_mae, color='#3498DB', linestyle=':', alpha=0.4, linewidth=1.5)
    ax.scatter([v1_best_epoch], [v1_best_mae], color='#E74C3C', s=250, marker='*',
               zorder=5, edgecolors='black', linewidths=1.5)
    ax.scatter([v2_best_epoch], [v2_best_mae], color='#3498DB', s=250, marker='*',
               zorder=5, edgecolors='black', linewidths=1.5)

    # 标注改进
    improvement = v1_best_mae - v2_best_mae
    improvement_pct = (improvement / v1_best_mae) * 100
    ax.text(v2_best_epoch + 4, (v1_best_mae + v2_best_mae) / 2,
            f'改进: -{improvement:.2f}°\n({improvement_pct:.1f}% ↓)',
            fontsize=11, color='green', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen', alpha=0.8,
                     edgecolor='darkgreen', linewidth=2))

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation MAE (°)', fontsize=12, fontweight='bold')
    ax.set_title('(B) 验证MAE对比 - V2改进9.4%', fontsize=13, fontweight='bold',
                 pad=10, color='darkgreen')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-2, max(len(v1['epochs']), len(v2['epochs'])) + 2)

    # 3. 验证Median AE对比
    ax = axes[1, 0]
    ax.plot(v1['epochs'], v1['val_med'], 'o-', label=f'V1: Best={v1["val_med"][v1_best_idx]:.2f}°',
            color='#E74C3C', alpha=0.8, linewidth=2.5, markersize=4)
    ax.plot(v2['epochs'], v2['val_med'], 's-', label=f'V2: Best={v2["val_med"][v2_best_idx]:.2f}°',
            color='#3498DB', alpha=0.8, linewidth=2.5, markersize=4)
    ax.axhline(v1['val_med'][v1_best_idx], color='#E74C3C', linestyle=':', alpha=0.4, linewidth=1.5)
    ax.axhline(v2['val_med'][v2_best_idx], color='#3498DB', linestyle=':', alpha=0.4, linewidth=1.5)
    ax.scatter([v1_best_epoch], [v1['val_med'][v1_best_idx]], color='#E74C3C', s=250,
               marker='*', zorder=5, edgecolors='black', linewidths=1.5)
    ax.scatter([v2_best_epoch], [v2['val_med'][v2_best_idx]], color='#3498DB', s=250,
               marker='*', zorder=5, edgecolors='black', linewidths=1.5)

    # 标注改进
    med_improvement = v1['val_med'][v1_best_idx] - v2['val_med'][v2_best_idx]
    med_improvement_pct = (med_improvement / v1['val_med'][v1_best_idx]) * 100
    ax.text(v2_best_epoch + 4, (v1['val_med'][v1_best_idx] + v2['val_med'][v2_best_idx]) / 2,
            f'改进: -{med_improvement:.2f}°\n({med_improvement_pct:.1f}% ↓)',
            fontsize=11, color='green', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen', alpha=0.8,
                     edgecolor='darkgreen', linewidth=2))

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Median AE (°)', fontsize=12, fontweight='bold')
    ax.set_title('(C) 验证中位数误差对比 - V2改进20.1%', fontsize=13, fontweight='bold',
                 pad=10, color='darkgreen')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-2, max(len(v1['epochs']), len(v2['epochs'])) + 2)

    # 4. 准确率对比
    ax = axes[1, 1]
    ax.plot(v1['epochs'], v1['acc'], 'o-', label=f'V1: Best={v1["acc"][v1_best_idx]:.2f}%',
            color='#E74C3C', alpha=0.8, linewidth=2.5, markersize=4)
    ax.plot(v2['epochs'], v2['acc'], 's-', label=f'V2: Best={v2["acc"][v2_best_idx]:.2f}%',
            color='#3498DB', alpha=0.8, linewidth=2.5, markersize=4)
    ax.axhline(v1['acc'][v1_best_idx], color='#E74C3C', linestyle=':', alpha=0.4, linewidth=1.5)
    ax.axhline(v2['acc'][v2_best_idx], color='#3498DB', linestyle=':', alpha=0.4, linewidth=1.5)
    ax.scatter([v1_best_epoch], [v1['acc'][v1_best_idx]], color='#E74C3C', s=250,
               marker='*', zorder=5, edgecolors='black', linewidths=1.5)
    ax.scatter([v2_best_epoch], [v2['acc'][v2_best_idx]], color='#3498DB', s=250,
               marker='*', zorder=5, edgecolors='black', linewidths=1.5)

    # 标注改进
    acc_improvement = v2['acc'][v2_best_idx] - v1['acc'][v1_best_idx]
    acc_improvement_pct = (acc_improvement / v1['acc'][v1_best_idx]) * 100
    ax.text(v2_best_epoch - 10, v2['acc'][v2_best_idx] + 1.5,
            f'改进: +{acc_improvement:.2f}%\n({acc_improvement_pct:.1f}% ↑)',
            fontsize=11, color='green', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen', alpha=0.8,
                     edgecolor='darkgreen', linewidth=2))

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('(D) 验证准确率对比 - V2提升21.1%', fontsize=13, fontweight='bold',
                 pad=10, color='darkgreen')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-2, max(len(v1['epochs']), len(v2['epochs'])) + 2)

    # 保存图片
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    output_path = '/disk2/bywang/DOA-net/V1_vs_V2_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 对比图已保存至: {output_path}")

    # 输出统计
    print("\n" + "="*70)
    print("关键改进总结:")
    print("="*70)
    improvement = v1_best_mae - v2_best_mae
    improvement_pct = (improvement / v1_best_mae) * 100
    print(f"MAE:        {v1_best_mae:.2f}° → {v2_best_mae:.2f}° ({improvement_pct:+.1f}%)")

    med_improvement = v1['val_med'][v1_best_idx] - v2['val_med'][v2_best_idx]
    med_improvement_pct = (med_improvement / v1['val_med'][v1_best_idx]) * 100
    print(f"Median AE:  {v1['val_med'][v1_best_idx]:.2f}° → {v2['val_med'][v2_best_idx]:.2f}° ({med_improvement_pct:+.1f}%)")

    acc_improvement = v2['acc'][v2_best_idx] - v1['acc'][v1_best_idx]
    acc_improvement_pct = (acc_improvement / v1['acc'][v1_best_idx]) * 100
    print(f"Accuracy:   {v1['acc'][v1_best_idx]:.2f}% → {v2['acc'][v2_best_idx]:.2f}% ({acc_improvement_pct:+.1f}%)")

    epochs_saved = len(v1['epochs']) - len(v2['epochs'])
    epochs_saved_pct = (epochs_saved / len(v1['epochs'])) * 100
    print(f"训练轮数:   {len(v1['epochs'])} → {len(v2['epochs'])} ({epochs_saved_pct:+.1f}%)")
    print("="*70)

if __name__ == '__main__':
    main()
