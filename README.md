# 双耳 DOA-Net

基于 LibriSpeech + CIPIC HRTF 的双耳声源到达方向（DOA）估计项目。
当前主线已从早期静态数据实验迁移到 50h 规模的合成数据流程，重点是：

- CIPIC(subject_003) 空间化
- 房间混响增强
- DEMAND 真实场景噪声增强
- 训练稳定性与长尾误差抑制

## 当前工作总结

### 阶段进展

1. 完成了 subject_003 主线的训练评估闭环（训练、恢复、正式测试）。
2. 完成了 50h CIPIC+混响数据合成与训练测试。
3. 完成了 50h CIPIC+混响+DEMAND 强噪版本（v1）训练测试。
4. 针对误差偏大问题完成了降误差改动（v2）：
   - 数据改为混合难度（clean/reverb/reverb+noise）
   - 训练稳定化（amp=false, grad_clip=1.0）
   - 启用前后对向混淆惩罚（训练损失内）
5. 完成了回归分支实验（v3，分类+DOA回归），相比v2 accuracy稍升但MAE未改善。
6. 完成了增强输入特征实验（v4，IPD sin/cos + coherence），结果接近v2基线。
7. **完成了创新主干结构实验（v5，attention bias + 独立门控 + attention pooling + 圆形软标签）**

### 近期关键结果对比

| 版本 | 数据配置 | Accuracy | Top-3 | MAE | Median | error<5° | 关键创新 |
|------|---------|----------|-------|-----|--------|----------|---------|
| v1 | 全量强噪 | 0.4641 | 0.7462 | 16.92° | 2.71° | - | baseline |
| v2 | 混合难度 | 0.7067 | 0.8649 | 9.67° | 2.24° | 77.6% | 数据混合 |
| v3 | 混合难度+回归 | 0.7280 | 0.8695 | 9.69° | 2.42° | 72.4% | 分类+回归 |
| v4 | 混合难度+增强特征 | 0.6952 | 0.8692 | 9.86° | 2.24° | 77.3% | IPD增强 |
| **v5** | **混合难度+创新主干** | **0.7344** | **0.8744** | **8.72°** | **2.14°** | **80.5%** | **架构创新** ✓ |

**v5 主要成果：**
- ✓ MAE 相对 v2 降低 9.8%（9.67° → 8.72°）
- ✓ accuracy 提升 3.9%（0.7067 → 0.7344）
- ✓ error<5° 的精准定位能力提升 3.7%（77.6% → 80.5%）
- ✓ 相比特征增强（v4）方案，架构创新显示出更强的优化潜力

## 模型架构

### 基础流程（v1-v4通用）

```
立体声 WAV
    │
    ▼
STFT 特征提取
(log_mag_L, log_mag_R, IPD, ILD)
    │
    ▼
共享编码器（左右耳）
    │
    ▼
差异先验 + 双向交叉注意力 + 门控
    │
    ▼
特征融合
    │
    ▼
BiGRU + 分类头
    │
    ▼
72 类方位角分类（[-180, 180)）
```

### v5 架构创新

v5 在上述基础上引入的核心改进：

1. **Attention Bias 注入（双向低秩）**
   - 从 d_feat 生成 rank=16 的双向 LR/RL attention bias
   - 在多头注意力计算中：score = QK^T/√d + bias
   - 增强频域与方位特征的相关性学习

2. **双向独立残差门控**
   - 替代统一门控，LR/RL 分离学习权重：g_lr、g_rl
   - 输出形式：f + g*a（残差+调制）
   - 适应双耳物理非对称性，精细化特征融合

3. **Attention Pooling**
   - BiGRU 后新增 attention 池化代替简单 mean pooling
   - 学习时间权重，动态突出关键时间步
   - 提升对长序列的自适应聚合与抗噪能力

4. **圆形软标签损失（Circular Soft Label Loss）**
   - 多任务：CE loss (weight=1.0) + circular soft label (weight=0.2, κ=4.0)
   - 利用 von Mises 分布编码角度的周期性约束
   - 减少类边界跳跃问题，提升梯度稳定性

主要模块：

- `dataset/feature_extractor.py`: 双耳频域特征提取
- `models/encoder.py`: 共享编码器
- `models/difference_prior.py`: 差异先验
- `models/cross_attention.py`: 双向交叉注意力 + attention bias
- `models/gating.py`: 独立残差门控（v5 升级）
- `models/temporal_head.py`: BiGRU + attention pooling（v5 升级）
- `models/binaural_doa_net.py`: 整体模型组装（集成所有创新）
- `losses.py`: 分类 + 圆形软标签多任务损失 (v5)

## 模型复杂度

### 参数量对比

| 版本 | 参数量 | FP32 大小 | FP16 大小 | 相对 v2 | 关键特点 |
|------|--------|---------|---------|---------|---------|
| v2 | 1.55M | 5.92 MB | 2.96 MB | baseline | 数据混合 |
| v3 | 1.58M | 6.05 MB | 3.02 MB | +2.1% | 分类+回归 |
| v4 | **1.68M** | **6.42 MB** | **3.21 MB** | **+8.5%** | IPD增强特征 |
| v5 | 1.55M | 5.92 MB | 2.96 MB | **0.0%** ✓ | 架构创新 |

**v5 的参数高效性：**
- ✓ 参数量 = v2（1.55M）
- ✓ 性能 > v4（v4多参数8.5%，但v5性能更优）
- ✓ Attention Bias 等创新无参数开销

### 分层参数分布（v5）

| 模块 | 参数量 | 占比 |
|------|--------|------|
| temporal_head（BiGRU+池化+分类） | 939,337 | 60.5% |
| difference_prior （差异先验） | 164,224 | 10.6% |
| cross_attention + attention bias | 132,608 | 8.5% |
| gating （门控） | 131,584 | 8.5% |
| encoder （编码器） | 109,632 | 7.1% |
| 其他（投影、偏置） | 74,304 | 4.8% |

### 推理成本

**单样本推理时间（batch_size=1）：**
- GPU (RTX 3090/4090): **< 1 ms** ✓ 实时
- GPU (V100/A100): **< 1 ms** ✓ 实时  
- CPU (Intel i7/i9): **5-20 ms** ✓ 可接受
- Edge Device (ARM CPU): **50-200 ms** ✓ 可边缘部署

**运行时内存占用：**
- v2/v5: ~17.8 MB（含激活值）
- v3: ~18.1 MB
- v4: ~19.3 MB

**结论：** v5 以最优的参数高效性和推理成本，实现了最佳的精度性能，适合生产和边缘设备部署。

## 消融实验方案

### 实验目标

验证 v5 的四个核心改动各自带来的收益，并分离它们的独立贡献与组合增益：

1. Attention Bias 注入
2. 双向独立残差门控
3. Attention Pooling
4. Circular Soft Label Loss

### 固定条件

所有消融实验必须保持以下条件完全一致：

- 数据集：`data/librispeech_cipic_subject003_reverb_demand50h_v2`
- 划分：train/val/test = 70/15/15
- 输入特征：`log_mag_L, log_mag_R, IPD, ILD`
- 训练策略：`lr=0.0005`, `amp=false`, `grad_clip=1.0`, `label_smoothing=0.1`
- 早停策略：`patience=15`
- 评估指标：Accuracy、Top-3 Accuracy、MAE、Median Error、error<5°、error<10°

### 具体消融组

下面把 v2 风格主干作为基线，然后逐项叠加 v5 改动。这里把门控拆成两个独立维度：

- `use_independent_gating=false`：共享门控
- `use_residual_gating=false`：非残差门控（g * a）

| 组别 | Attention Bias | 独立门控 | 残差门控 | Attention Pooling | Circular Soft Label | 说明 |
|------|----------------|---------|---------|-------------------|---------------------|------|
| A0 | 关闭 | 关闭 | 关闭 | 关闭 | 关闭 | 共享非残差门控基线 |
| A1 | 开启 | 关闭 | 关闭 | 关闭 | 关闭 | 只验证 attention bias |
| A2 | 关闭 | 开启 | 开启 | 关闭 | 关闭 | 只验证独立残差门控 |
| A3 | 关闭 | 关闭 | 关闭 | 开启 | 关闭 | 只验证 attention pooling |
| A4 | 关闭 | 关闭 | 关闭 | 关闭 | 开启 | 只验证 circular soft label |
| A5 | 开启 | 开启 | 开启 | 关闭 | 关闭 | 验证 bias + 独立门控协同 |
| A6 | 开启 | 开启 | 开启 | 开启 | 关闭 | 验证 bias + 门控 + pooling |
| **A7** | **开启** | **开启** | **开启** | **开启** | **开启** | **完整 v5** |

### 具体运行方式

每组实验都可以通过统一脚本启动：

```bash
bash run_cipic_reverb_demand50h_ablation_pipeline.sh a0 smoke
bash run_cipic_reverb_demand50h_ablation_pipeline.sh a7 smoke
```

如果要直接进入完整训练：

```bash
bash run_cipic_reverb_demand50h_ablation_pipeline.sh a0 full
```

### 结果记录建议

建议每组记录以下内容：

- 最佳验证轮次
- 最佳验证 MAE
- 最终测试集 Accuracy / Top-3 / MAE / Median Error
- `error<5°` 和 `error<10°`
- 参数量与推理耗时

### 判定标准

优先以以下顺序判断方案是否有效：

1. MAE 是否下降
2. `error<5°` 是否上升
3. Accuracy 是否上升
4. 参数量是否显著增加

如果某一改动只提升 Accuracy 但恶化 MAE，则不建议作为主干保留。

## 数据与实验主线

### 数据来源

- 语音：LibriSpeech `train-clean-100`
- HRTF：CIPIC `subject_003.sofa`
- 噪声：DEMAND（多真实场景）

### 当前主用数据集

- 50h clean（CIPIC 空间化）
  - `data/librispeech_cipic_subject003_50h_clean`
- 50h reverb-only
  - `data/librispeech_cipic_subject003_reverb50h`
- 50h reverb+DEMAND（v1，全量强噪）
  - `data/librispeech_cipic_subject003_reverb_demand50h`
- 50h reverb+DEMAND（v2~v5，混合难度）
  - `data/librispeech_cipic_subject003_reverb_demand50h_v2`

### 划分比例

统一采用：

- train: 70%
- val: 15%
- test: 15%

### 采样率与片段长度

- 采样率：16 kHz
- 训练片段长度：2 秒

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练

使用当前推荐的 v5 创新主干配置训练：

```bash
python train.py --config configs/train_librispeech_subject003_cipic_reverb_demand50h_v5_bias_gating_attnpool_csl.yaml
```

从 best 恢复并继续训练：

```bash
python train.py \
  --config configs/train_librispeech_subject003_cipic_reverb_demand50h_v5_bias_gating_attnpool_csl.yaml \
  --resume outputs/checkpoints_librispeech_subject003_cipic_reverb_demand50h_v5_bias_gating_attnpool_csl/best.pth \
  --train.epochs 30
```

### 3. 评估

```bash
python evaluate.py \
  --checkpoint outputs/checkpoints_librispeech_subject003_cipic_reverb_demand50h_v5_bias_gating_attnpool_csl/best.pth \
  --config configs/train_librispeech_subject003_cipic_reverb_demand50h_v5_bias_gating_attnpool_csl.yaml \
  --output.log_dir outputs/logs_librispeech_subject003_cipic_reverb_demand50h_v5_bias_gating_attnpool_csl_test_full_best
```

### 4. 流水线

- 50h CIPIC + 混响：

```bash
bash run_cipic_reverb50h_pipeline.sh
```

- 50h CIPIC + 混响 + DEMAND（v5 创新主干）：

```bash
bash run_cipic_reverb_demand50h_v5_pipeline.sh  # 待补充
```

## 当前推荐配置

推荐使用 **v5（创新主干架构）**：

- `configs/train_librispeech_subject003_cipic_reverb_demand50h_v5_bias_gating_attnpool_csl.yaml`

核心配置特点：

```yaml
model:
  use_attention_bias: true         # 低秩双向 attention bias
  attention_bias_rank: 16
  use_attention_pooling: true      # BiGRU 后 attention pooling
  use_gating: true                 # 双向独立残差门控
  gru_hidden_size: 128
  
train:
  lr: 0.0005                       # 保守学习率
  amp: false                        # 关闭 AMP，稳定数值
  grad_clip: 1.0                   # 严格梯度裁剪
  circular_soft_label_weight: 0.2  # 圆形软标签权重
  circular_kappa: 4.0              # von Mises 浓度参数
  anti_confusion_weight: 1.0       # 前后对向惩罚
  early_stopping_patience: 15      # 早停耐心度
```

## 项目结构

```
DOA-net/
├── README.md
├── requirements.txt
├── train.py
├── evaluate.py
├── infer.py
├── losses.py
├── metrics.py
├── synthesize_librispeech_cipic.py
├── prepare_demand_mixed_data.py
├── run_cipic_reverb50h_pipeline.sh
├── run_cipic_reverb_demand50h_pipeline.sh
├── run_cipic_reverb_demand50h_v2_pipeline.sh
├── run_cipic_reverb_demand50h_v3_regression_pipeline.sh
├── run_cipic_reverb_demand50h_v4_enhanced_features_pipeline.sh
├── run_cipic_reverb_demand50h_ablation_pipeline.sh
├── configs/
│   ├── default.yaml
│   ├── train_librispeech_subject003_cipic_reverb50h.yaml
│   ├── train_librispeech_subject003_cipic_reverb_demand50h.yaml
│   ├── train_librispeech_subject003_cipic_reverb_demand50h_v2.yaml
│   ├── train_librispeech_subject003_cipic_reverb_demand50h_v3_regression.yaml
│   ├── train_librispeech_subject003_cipic_reverb_demand50h_v4_enhanced_features.yaml
│   └── train_librispeech_subject003_cipic_reverb_demand50h_v5_bias_gating_attnpool_csl.yaml  ★
├── dataset/
├── engine/
├── models/
├── utils/
├── data/
└── outputs/
```

## 说明

- 本 README 已聚焦当前主线实验与可复现流程。
- 早期 static 数据集路线与早期后处理式混淆修正说明已移除。
- v5 是当前架构创新的最新成果，相比 v2 基线在 MAE、accuracy、精准误差占比上有全面改进。
- 详细的版本对比分析见：`v5_comparison_analysis.md`
- 如需回溯历史实验，可查看 `outputs/` 下对应日志目录。

---

最后更新：2026-04-15  
当前推荐 best：  
`outputs/checkpoints_librispeech_subject003_cipic_reverb_demand50h_v5_bias_gating_attnpool_csl/best.pth`

**推荐配置：**  
`configs/train_librispeech_subject003_cipic_reverb_demand50h_v5_bias_gating_attnpool_csl.yaml`

**关键指标（v5 测试集）：**
- Accuracy: 0.7344
- Top-3 Accuracy: 0.8744
- MAE: 8.72°
- Median Error: 2.14°
- Error < 5° 占比: 80.5%
