#!/usr/bin/env python3
"""检查数据集中方位角的分布情况（纯Python实现）。"""

import os
import glob
import math

print("=" * 60)
print("检查数据集方位角分布")
print("=" * 60)

metadata_dir = "/disk2/bywang/DOA-net/data/static/metadata_dev"
metadata_files = sorted(glob.glob(os.path.join(metadata_dir, "metadata*.csv")))[:100]

print(f"\n正在分析前 {len(metadata_files)} 个文件的方位角分布...\n")

all_azimuths = []

for metadata_path in metadata_files:
    try:
        with open(metadata_path, 'r') as f:
            lines = f.readlines()

        # 取中间帧的坐标
        mid_idx = len(lines) // 2
        parts = lines[mid_idx].strip().split(',')

        x = float(parts[1])
        y = float(parts[2])

        # 计算方位角（与代码中的计算方式一致）
        azimuth_rad = math.atan2(x, y)
        azimuth_deg = math.degrees(azimuth_rad)

        all_azimuths.append(azimuth_deg)

    except Exception as e:
        print(f"⚠️  文件 {os.path.basename(metadata_path)} 读取失败: {e}")

print("统计信息:")
print(f"  样本数:     {len(all_azimuths)}")
print(f"  最小值:     {min(all_azimuths):.2f}°")
print(f"  最大值:     {max(all_azimuths):.2f}°")
print(f"  平均值:     {sum(all_azimuths)/len(all_azimuths):.2f}°")

sorted_az = sorted(all_azimuths)
median = sorted_az[len(sorted_az)//2]
print(f"  中位数:     {median:.2f}°")

# 计算标准差
mean = sum(all_azimuths)/len(all_azimuths)
variance = sum((x - mean)**2 for x in all_azimuths) / len(all_azimuths)
std = math.sqrt(variance)
print(f"  标准差:     {std:.2f}°")

# 分区间统计
count_1 = sum(1 for x in all_azimuths if -180 <= x < -90)
count_2 = sum(1 for x in all_azimuths if -90 <= x < 0)
count_3 = sum(1 for x in all_azimuths if 0 <= x < 90)
count_4 = sum(1 for x in all_azimuths if 90 <= x < 180)

print(f"\n方位角分布:")
print(f"  [-180°, -90°): {count_1} 个 ({count_1/len(all_azimuths)*100:.1f}%)")
print(f"  [ -90°,   0°): {count_2} 个 ({count_2/len(all_azimuths)*100:.1f}%)")
print(f"  [   0°,  90°): {count_3} 个 ({count_3/len(all_azimuths)*100:.1f}%)")
print(f"  [  90°, 180°): {count_4} 个 ({count_4/len(all_azimuths)*100:.1f}%)")

# 检查唯一值
unique_azimuths = sorted(set(round(x, 1) for x in all_azimuths))
print(f"\n唯一方位角数量（0.1°精度）: {len(unique_azimuths)}")

if len(unique_azimuths) < 50:
    print(f"  ⚠️  方位角种类较少，可能是离散化的数据")
    print(f"  前20个唯一值: {unique_azimuths[:20]}")
else:
    print(f"  ✅ 方位角分布较为均匀/连续")

# 示例数据
print(f"\n前10个样本的方位角:")
for i, az in enumerate(all_azimuths[:10]):
    file_num = i + 1
    print(f"  文件 {file_num:04d}: {az:7.2f}°")

print("\n" + "=" * 60)
print("\n结论:")
if std > 60:
    print("  ✅ 数据在整个360度范围内分布良好")
elif len(unique_azimuths) < 20:
    print("  ⚠️  数据可能集中在少数几个方位角")
else:
    print("  ✅ 数据分布正常")

print("=" * 60)
