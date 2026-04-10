#!/usr/bin/env python3
"""快速验证回归版模型是否正常工作。"""

import sys
sys.path.insert(0, '/disk2/bywang/DOA-net')

import torch
from utils.config import load_config
from models import build_model

print("=" * 60)
print("测试回归版DOA模型")
print("=" * 60)

# 加载配置
print("\n1. 加载配置文件...")
cfg = load_config("configs/train_static_regression.yaml", [])
print(f"   ✅ use_regression = {cfg.model.use_regression}")
print(f"   ✅ regression_weight = {cfg.train.regression_weight}")

# 构建模型
print("\n2. 构建模型...")
model = build_model(cfg)
print(f"   ✅ 模型构建成功")

# 统计参数
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   总参数: {total_params:,}")
print(f"   可训练参数: {trainable_params:,}")

# 创建假数据测试forward
print("\n3. 测试模型forward...")
batch_size = 4
T = 200  # 时间帧数
F = 257  # 频率bins

fake_batch = {
    'log_mag_L': torch.randn(batch_size, T, F),
    'log_mag_R': torch.randn(batch_size, T, F),
    'ipd': torch.randn(batch_size, T, F),
    'ild': torch.randn(batch_size, T, F),
    'azimuth_label': torch.randint(0, 72, (batch_size,)),
    'azimuth_deg': torch.randn(batch_size) * 180,  # [-180, 180]
}

model.eval()
with torch.no_grad():
    output = model(fake_batch)

print(f"   ✅ Forward成功")
print(f"   输出键: {list(output.keys())}")
print(f"   logits shape: {output['logits'].shape}")

if 'angle' in output:
    print(f"   ✅ angle shape: {output['angle'].shape}")
    print(f"   angle 范围: [{output['angle'].min():.2f}, {output['angle'].max():.2f}] (弧度)")
    angle_deg = torch.rad2deg(output['angle'])
    print(f"   angle 范围: [{angle_deg.min():.2f}°, {angle_deg.max():.2f}°] (角度)")
else:
    print(f"   ❌ 未找到angle输出（预期应该有）")

# 测试loss
print("\n4. 测试多任务loss...")
from losses import MultiTaskDOALoss

criterion = MultiTaskDOALoss(
    num_classes=72,
    label_smoothing=0.15,
    regression_weight=0.5,
    use_angular_loss=True,
)

pred_logits = output['logits']
pred_angle = output['angle']
target_label = fake_batch['azimuth_label']
target_angle_rad = torch.deg2rad(fake_batch['azimuth_deg'])

loss_dict = criterion(pred_logits, pred_angle, target_label, target_angle_rad)

print(f"   ✅ Loss计算成功")
print(f"   Total loss: {loss_dict['total']:.4f}")
print(f"   Classification loss: {loss_dict['classification']:.4f}")
print(f"   Regression loss: {loss_dict['regression']:.4f}")

print("\n" + "=" * 60)
print("✅ 所有测试通过！回归版模型可以正常使用。")
print("=" * 60)
print("\n可以开始训练了：")
print("  python train.py --config configs/train_static_regression.yaml")
print("=" * 60)
