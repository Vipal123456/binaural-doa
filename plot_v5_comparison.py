#!/usr/bin/env python3
"""v5 与历史版本对比图表生成"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 中文字体配置
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 数据整理
versions = ['v2\n(混合难度)', 'v3\n(回归分支)', 'v4\n(增强特征)', 'v5\n(创新主干)']
colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']

# 各版本指标数据
accuracy = [0.7067, 0.7280, 0.6952, 0.7344]
top3_accuracy = [0.8649, 0.8695, 0.8692, 0.8744]
mae = [9.6673, 9.6914, 9.8597, 8.7191]
median_ae = [2.2356, 2.4236, 2.2425, 2.1401]
error_lt5 = [0.7760, 0.7244, 0.7726, 0.8045]
error_lt10 = [0.8612, 0.8702, 0.8614, 0.8789]

# 创建图表
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('v2/v3/v4/v5 版本完整测试集对比分析', fontsize=16, fontweight='bold')

# 1. Accuracy 对比
ax = axes[0, 0]
bars = ax.bar(versions, accuracy, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax.set_ylim([0.65, 0.75])
ax.grid(axis='y', alpha=0.3, linestyle='--')
for i, (bar, val) in enumerate(zip(bars, accuracy)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.001, f'{val:.4f}', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    if i == 3:
        ax.text(bar.get_x() + bar.get_width()/2, val - 0.003, '★ 最佳', 
                ha='center', va='top', fontsize=8, color='red', fontweight='bold')

# 2. Top-3 Accuracy 对比
ax = axes[0, 1]
bars = ax.bar(versions, top3_accuracy, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Top-3 Accuracy', fontsize=11, fontweight='bold')
ax.set_ylim([0.85, 0.88])
ax.grid(axis='y', alpha=0.3, linestyle='--')
for i, (bar, val) in enumerate(zip(bars, top3_accuracy)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.001, f'{val:.4f}', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    if i == 3:
        ax.text(bar.get_x() + bar.get_width()/2, val - 0.002, '★ 最佳', 
                ha='center', va='top', fontsize=8, color='red', fontweight='bold')

# 3. MAE 对比（越小越好）
ax = axes[0, 2]
bars = ax.bar(versions, mae, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('MAE (°)', fontsize=11, fontweight='bold')
ax.set_ylim([8, 10.5])
ax.grid(axis='y', alpha=0.3, linestyle='--')
for i, (bar, val) in enumerate(zip(bars, mae)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.1, f'{val:.2f}°', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    if i == 3:
        ax.text(bar.get_x() + bar.get_width()/2, val - 0.15, '★ 最佳\n-10.04%', 
                ha='center', va='top', fontsize=8, color='red', fontweight='bold')

# 4. 中位误差对比（越小越好）
ax = axes[1, 0]
bars = ax.bar(versions, median_ae, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('中位误差 (°)', fontsize=11, fontweight='bold')
ax.set_ylim([1.8, 2.6])
ax.grid(axis='y', alpha=0.3, linestyle='--')
for i, (bar, val) in enumerate(zip(bars, median_ae)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.2f}°', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    if i == 3:
        ax.text(bar.get_x() + bar.get_width()/2, val - 0.08, '★ 最佳', 
                ha='center', va='top', fontsize=8, color='red', fontweight='bold')

# 5. Error < 5° 比例对比
ax = axes[1, 1]
bars = ax.bar(versions, error_lt5, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Error < 5° 比例', fontsize=11, fontweight='bold')
ax.set_ylim([0.7, 0.82])
ax.grid(axis='y', alpha=0.3, linestyle='--')
for i, (bar, val) in enumerate(zip(bars, error_lt5)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.002, f'{val:.1%}', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    if i == 3:
        ax.text(bar.get_x() + bar.get_width()/2, val - 0.005, '★ 最佳\n+11%', 
                ha='center', va='top', fontsize=8, color='red', fontweight='bold')

# 6. Error < 10° 比例对比
ax = axes[1, 2]
bars = ax.bar(versions, error_lt10, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Error < 10° 比例', fontsize=11, fontweight='bold')
ax.set_ylim([0.85, 0.895])
ax.grid(axis='y', alpha=0.3, linestyle='--')
for i, (bar, val) in enumerate(zip(bars, error_lt10)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.0015, f'{val:.1%}', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    if i == 3:
        ax.text(bar.get_x() + bar.get_width()/2, val - 0.004, '★ 最佳', 
                ha='center', va='top', fontsize=8, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/v5_comparison_metrics.png', dpi=150, bbox_inches='tight')
print("✓ 指标对比图已保存: outputs/v5_comparison_metrics.png")
plt.close()

# 创建雷达图
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# 标准化数据（都变为 0-1 范围，值越大越好）
categories = [
    'Accuracy',
    'Top-3\nAccuracy',
    'MAE\n(反向)',
    '中位误差\n(反向)',
    'Error<5%',
    'Error<10%'
]

# 标准化：反向指标（MAE、中位误差）需要反向处理
accuracy_norm = accuracy  # [0.7, 0.78]
top3_norm = top3_accuracy  # [0.86, 0.88]
mae_norm = [1 - (m - min(mae)) / (max(mae) - min(mae)) for m in mae]  # 反向
median_norm = [1 - (m - min(median_ae)) / (max(median_ae) - min(median_ae)) for m in median_ae]  # 反向
error_lt5_norm = error_lt5  # [0.72, 0.80]
error_lt10_norm = error_lt10  # [0.86, 0.88]

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True, linestyle='--', alpha=0.7)

# 绘制每个版本
for i, (version, color) in enumerate(zip(versions, colors)):
    values = [
        (accuracy[i] - min(accuracy)) / (max(accuracy) - min(accuracy)),
        (top3_accuracy[i] - min(top3_accuracy)) / (max(top3_accuracy) - min(top3_accuracy)),
        mae_norm[i],
        median_norm[i],
        (error_lt5[i] - min(error_lt5)) / (max(error_lt5) - min(error_lt5)),
        (error_lt10[i] - min(error_lt10)) / (max(error_lt10) - min(error_lt10))
    ]
    values += values[:1]
    
    linewidth = 2.5 if i == 3 else 1.5  # v5 线条加粗
    ax.plot(angles, values, 'o-', linewidth=linewidth, label=version, color=color, markersize=6)
    ax.fill(angles, values, alpha=0.15, color=color)

ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax.set_title('版本性能雷达对比\n(越接近外圈越好)', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('outputs/v5_comparison_radar.png', dpi=150, bbox_inches='tight')
print("✓ 雷达图已保存: outputs/v5_comparison_radar.png")
plt.close()

# 创建改进趋势图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 相对于 v2 的改进百分比
ax = axes[0]
v2_baseline = {
    'Accuracy': accuracy[0],
    'Top-3 Accuracy': top3_accuracy[0],
    'MAE': mae[0],
    'Error<5%': error_lt5[0]
}

improvements = {
    'v3': {
        'Accuracy': (accuracy[1] - accuracy[0]) / accuracy[0] * 100,
        'Top-3 Accuracy': (top3_accuracy[1] - top3_accuracy[0]) / top3_accuracy[0] * 100,
        'MAE': (mae[1] - mae[0]) / mae[0] * 100,  # 越小越好，改进为负数
        'Error<5%': (error_lt5[1] - error_lt5[0]) / error_lt5[0] * 100,
    },
    'v4': {
        'Accuracy': (accuracy[2] - accuracy[0]) / accuracy[0] * 100,
        'Top-3 Accuracy': (top3_accuracy[2] - top3_accuracy[0]) / top3_accuracy[0] * 100,
        'MAE': (mae[2] - mae[0]) / mae[0] * 100,
        'Error<5%': (error_lt5[2] - error_lt5[0]) / error_lt5[0] * 100,
    },
    'v5': {
        'Accuracy': (accuracy[3] - accuracy[0]) / accuracy[0] * 100,
        'Top-3 Accuracy': (top3_accuracy[3] - top3_accuracy[0]) / top3_accuracy[0] * 100,
        'MAE': (mae[3] - mae[0]) / mae[0] * 100,
        'Error<5%': (error_lt5[3] - error_lt5[0]) / error_lt5[0] * 100,
    }
}

x = np.arange(len(v2_baseline))
width = 0.25

for idx, (ver, imp) in enumerate(improvements.items()):
    values = list(imp.values())
    # MAE 和 Error<5% 符号调整（MAE越小越好，转换为正的改进）
    values[2] = -values[2]  # MAE 改为越小越好的正向指标
    
    color_map = {'v3': '#e74c3c', 'v4': '#f39c12', 'v5': '#2ecc71'}
    ax.bar(x + idx * width, values, width, label=ver, color=color_map[ver], alpha=0.85, edgecolor='black', linewidth=1)

ax.set_ylabel('相对 v2 的改进 (%)', fontsize=11, fontweight='bold')
ax.set_title('各版本相对 v2 基线的改进率', fontsize=12, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(v2_baseline.keys(), fontsize=10)
ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 添加数值标签
for idx, (ver, imp) in enumerate(improvements.items()):
    values = list(imp.values())
    values[2] = -values[2]
    for i, v in enumerate(values):
        x_pos = x[i] + idx * width
        va = 'bottom' if v >= 0 else 'top'
        y_pos = v + (0.3 if v >= 0 else -0.3)
        ax.text(x_pos, y_pos, f'{v:.1f}%', ha='center', va=va, fontsize=8, fontweight='bold')

# v2→v3→v4→v5 的演进趋势
ax = axes[1]
metrics_names = ['Accuracy', 'Top-3 Acc', 'MAE(↓)', 'Error<5%']
x_range = np.arange(len(metrics_names))

# 归一化到 [0, 100] 范围便于统一展示
acc_norm = [(a - min(accuracy)) / (max(accuracy) - min(accuracy)) * 100 for a in accuracy]
top3_norm = [(a - min(top3_accuracy)) / (max(top3_accuracy) - min(top3_accuracy)) * 100 for a in top3_accuracy]
mae_norm_100 = [(1 - (m - min(mae)) / (max(mae) - min(mae))) * 100 for m in mae]
error_norm = [(e - min(error_lt5)) / (max(error_lt5) - min(error_lt5)) * 100 for e in error_lt5]

v_versions = [versions[i].split('\n')[0] for i in range(4)]
for i, v_name in enumerate(v_versions):
    values = [acc_norm[i], top3_norm[i], mae_norm_100[i], error_norm[i]]
    ax.plot(x_range, values, 'o-', linewidth=2.5, markersize=8, label=v_name, color=colors[i])

ax.set_ylabel('归一化性能指标 (0-100)', fontsize=11, fontweight='bold')
ax.set_title('版本演进趋势', fontsize=12, fontweight='bold')
ax.set_xticks(x_range)
ax.set_xticklabels(metrics_names, fontsize=10)
ax.set_ylim([30, 105])
ax.legend(fontsize=10, loc='lower left')
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('outputs/v5_comparison_trends.png', dpi=150, bbox_inches='tight')
print("✓ 趋势图已保存: outputs/v5_comparison_trends.png")
plt.close()

print("\n" + "="*60)
print("✓ 所有对比可视化已生成完毕!")
print("="*60)
print("\n输出文件：")
print("  1. outputs/v5_comparison_metrics.png    - 指标对比条形图")
print("  2. outputs/v5_comparison_radar.png      - 性能雷达图")
print("  3. outputs/v5_comparison_trends.png     - 版本演进趋势")
