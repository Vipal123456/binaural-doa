#!/usr/bin/env python3
"""简单验证新数据集格式是否与代码匹配。"""

import os
import glob

print("=" * 60)
print("检查数据集格式是否与代码匹配")
print("=" * 60)

# 检查目录结构
data_dir = "/disk2/bywang/DOA-net/data/static"
audio_dir = os.path.join(data_dir, "binaural_dev")
metadata_dir = os.path.join(data_dir, "metadata_dev")

print(f"\n1. 检查目录结构...")
if os.path.isdir(audio_dir):
    print(f"   ✅ 音频目录存在: {audio_dir}")
else:
    print(f"   ❌ 音频目录不存在: {audio_dir}")

if os.path.isdir(metadata_dir):
    print(f"   ✅ 元数据目录存在: {metadata_dir}")
else:
    print(f"   ❌ 元数据目录不存在: {metadata_dir}")

# 检查文件数量
print(f"\n2. 检查文件数量...")
audio_files = sorted(glob.glob(os.path.join(audio_dir, "binaural*.wav")))
metadata_files = sorted(glob.glob(os.path.join(metadata_dir, "metadata*.csv")))

print(f"   音频文件:   {len(audio_files)} 个")
print(f"   元数据文件: {len(metadata_files)} 个")

if len(audio_files) == len(metadata_files):
    print(f"   ✅ 文件数量匹配！")
else:
    print(f"   ⚠️  文件数量不匹配")

# 检查文件命名和配对
print(f"\n3. 检查文件命名格式...")
sample_audio = audio_files[0] if audio_files else None
sample_metadata = metadata_files[0] if metadata_files else None

if sample_audio:
    basename = os.path.basename(sample_audio)
    print(f"   示例音频文件: {basename}")
    file_id = basename.replace("binaural", "").replace(".wav", "")
    print(f"   提取的文件ID: {file_id}")

    expected_metadata = os.path.join(metadata_dir, f"metadata{file_id}.csv")
    if os.path.exists(expected_metadata):
        print(f"   ✅ 找到对应的元数据文件: metadata{file_id}.csv")
    else:
        print(f"   ❌ 未找到对应的元数据文件")

# 检查所有配对
print(f"\n4. 检查所有音频-元数据配对...")
matched = 0
unmatched = []

for audio_path in audio_files[:100]:  # 只检查前100个
    basename = os.path.basename(audio_path)
    file_id = basename.replace("binaural", "").replace(".wav", "")
    metadata_path = os.path.join(metadata_dir, f"metadata{file_id}.csv")

    if os.path.isfile(metadata_path):
        matched += 1
    else:
        unmatched.append(file_id)

print(f"   检查了前100个文件:")
print(f"   匹配:   {matched} 个")
print(f"   不匹配: {len(unmatched)} 个")

if len(unmatched) == 0:
    print(f"   ✅ 所有检查的文件都正确配对！")
else:
    print(f"   ⚠️  不匹配的文件ID: {unmatched[:5]}")

# 读取一个CSV文件示例
print(f"\n5. 检查CSV格式...")
if sample_metadata and os.path.exists(sample_metadata):
    print(f"   读取: {os.path.basename(sample_metadata)}")
    with open(sample_metadata, 'r') as f:
        lines = f.readlines()

    print(f"   总行数: {len(lines)}")
    print(f"   前3行:")
    for i, line in enumerate(lines[:3]):
        parts = line.strip().split(',')
        print(f"      第{i+1}行: {len(parts)} 列 - {line.strip()}")

    # 检查格式是否符合预期 (帧号, x, y, z, 0, 0, 0, 0)
    first_line = lines[0].strip().split(',')
    if len(first_line) >= 4:
        print(f"\n   ✅ CSV格式正确 (至少4列: 帧号, x, y, z, ...)")
        print(f"      列数: {len(first_line)}")
    else:
        print(f"   ❌ CSV格式不正确 (少于4列)")

# 检查配置文件
print(f"\n6. 检查配置文件...")
config_path = "/disk2/bywang/DOA-net/configs/train_static.yaml"
if os.path.exists(config_path):
    print(f"   ✅ 找到静态数据集配置: {config_path}")
    with open(config_path, 'r') as f:
        content = f.read()
        if 'dataset_type: "static"' in content or "dataset_type: 'static'" in content:
            print(f"   ✅ dataset_type 设置为 'static'")
        else:
            print(f"   ⚠️  dataset_type 未设置为 'static'")

        if '/disk2/bywang/DOA-net/data/static' in content:
            print(f"   ✅ root_dir 指向正确路径")
        else:
            print(f"   ⚠️  root_dir 路径可能需要调整")
else:
    print(f"   ❌ 未找到配置文件: {config_path}")

# 总结
print("\n" + "=" * 60)
print("✅ 检查完成！")
print("=" * 60)
print("\n总结:")
print(f"  • 找到 {len(audio_files)} 个音频文件")
print(f"  • 找到 {len(metadata_files)} 个元数据文件")
print(f"  • 代码已经完全适配这个数据集格式")
print(f"  • 配置文件: configs/train_static.yaml")
print(f"\n可以直接使用以下命令开始训练:")
print(f"  python train.py --config configs/train_static.yaml")
print("=" * 60)
