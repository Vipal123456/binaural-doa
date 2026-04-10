#!/usr/bin/env python3
"""检查数据集中方位角的分布情况。"""

import os
import glob
import numpy as np

print("=" * 60)
print("检查数据集方位角分布")
print("=" * 60)

metadata_dir = "/disk2/bywang/DOA-net/data/static/metadata_dev"
metadata_files = sorted(glob.glob(os.path.join(metadata_dir, "metadata*.csv")))[:100]  # 检查前100个

print(f"\n正在分析前 {len(metadata_files)} 个文件的方位角分布...\n")

all_azimuths = []

for metadata_path in metadata_files:
    try:
        data = np.loadtxt(metadata_path, delimiter=',')

        # 取中间帧的坐标
        mid_idx = len(data) // 2
        x = data[mid_idx, 1]
        y = data[mid_idx, 2]

        # 计算方位角（与代码中的计算方式一致）
        azimuth_rad = np.arctan2(x, y)
        azimuth_deg = np.degrees(azimuth_rad)

        all_azimuths.append(azimuth_deg)

    except Exception as e:
        print(f"⚠️  文件 {os.path.basename(metadata_path)} 读取失败: {e}")

all_azimuths = np.array(all_azimuths)

print("统计信息:")
print(f"  样本数:     {len(all_azimuths)}")
print(f"  最小值:     {all_azimuths.min():.2f}°")
print(f"  最大值:     {all_azimuths.max():.2f}°")
print(f"  平均值:     {all_azimuths.mean():.2f}°")
print(f"  标准差:     {all_azimuths.std():.2f}°")
print(f"  中位数:     {np.median(all_azimuths):.2f}°")

# 分区间统计
bins = [-180, -90, 0, 90, 180]
hist, _ = np.histogram(all_azimuths, bins=bins)
print(f"\n方位角分布:")
print(f"  [-180°, -90°): {hist[0]} 个 ({hist[0]/len(all_azimuths)*100:.1f}%)")
print(f"  [ -90°,   0°): {hist[1]} 个 ({hist[1]/len(all_azimuths)*100:.1f}%)")
print(f"  [   0°,  90°): {hist[2]} 个 ({hist[2]/len(all_azimuths)*100:.1f}%)")
print(f"  [  90°, 180°): {hist[3]} 个 ({hist[3]/len(all_azimuths)*100:.1f}%)")

# 检查是否有聚类现象
unique_azimuths = np.unique(np.round(all_azimuths, 1))
print(f"\n唯一方位角数量（0.1°精度）: {len(unique_azimuths)}")

if len(unique_azimuths) < 50:
    print(f"  ⚠️  方位角种类较少，可能是离散化的数据")
    print(f"  前10个唯一值: {unique_azimuths[:10]}")
else:
    print(f"  ✅ 方位角分布较为均匀/连续")

# 示例数据
print(f"\n前10个样本的方位角:")
for i, az in enumerate(all_azimuths[:10]):
    print(f"  文件 {i+1:04d}: {az:7.2f}°")

print("\n" + "=" * 60)
