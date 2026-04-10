#!/usr/bin/env python3
"""快速验证静态数据集是否能正确加载。"""

import sys
sys.path.insert(0, '/disk2/bywang/DOA-net')

from utils.config import load_config
from dataset.static_dataset import build_static_datasets

# 加载配置
cfg = load_config("configs/train_static.yaml", [])

print("=" * 60)
print("正在加载静态数据集...")
print("=" * 60)

# 构建数据集
try:
    train_ds, val_ds, test_ds = build_static_datasets(cfg)

    print(f"\n✅ 数据集加载成功！")
    print(f"   训练集: {len(train_ds)} 个片段")
    print(f"   验证集: {len(val_ds)} 个片段")
    print(f"   测试集: {len(test_ds)} 个片段")
    print(f"   总计:   {len(train_ds) + len(val_ds) + len(test_ds)} 个片段")

    # 测试读取一个样本
    print(f"\n正在测试读取第一个样本...")
    sample = train_ds[0]

    print(f"\n✅ 样本读取成功！")
    print(f"   log_mag_L shape: {sample['log_mag_L'].shape}")
    print(f"   log_mag_R shape: {sample['log_mag_R'].shape}")
    print(f"   ipd shape:       {sample['ipd'].shape}")
    print(f"   ild shape:       {sample['ild'].shape}")
    print(f"   方位角标签:      {sample['azimuth_label']}")
    print(f"   方位角(度):      {sample['azimuth_deg']:.2f}°")

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！数据集可以正常使用。")
    print("=" * 60)

except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
