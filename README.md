# 双耳 DOA-Net

基于 LibriSpeech + CIPIC HRTF 的双耳声源到达方向（DOA）估计项目。
当前主线已从早期静态数据实验迁移到 50h 规模的合成数据流程，重点是：

- CIPIC(subject_003) 空间化
- CIPIC 多 subject / subject-disjoint 泛化
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
8. **完成了 robust50h 多 subject、subject-disjoint 数据集与 unseen-subject 评估主线**
   - 新数据集：`data/librispeech_cipic_multisubject_robust50h_v1`
   - 30 个 subject，按 `24 / 3 / 3` 做 train / val / test subject-disjoint 划分
   - 新增 `train_root / val_root / test_root` 显式 split-root 加载方式
   - v5 baseline 在 unseen-subject test 上达到：`MAE=15.90°`，`median=2.50°`，`error<10°=82.66%`
   - **enhanced binaural features** 版本在 unseen-subject test 上进一步达到：`MAE=13.77°`，`median=2.50°`，`error<10°=85.28%`
   - **front/back auxiliary only** 版本当前达到最佳 unseen-subject test：`MAE=11.97°`，`median=2.50°`，`error<10°=86.93%`

## 新增主线：robust50h 多 subject / unseen-subject 泛化

### 数据集设计

当前仓库已支持新的 robust50h 多 subject 数据主线：

- 数据集根目录：
  - `data/librispeech_cipic_multisubject_robust50h_v1`
- 语音：
  - `LibriSpeech/train-clean-100`
- HRTF：
  - `/disk2/bywang/data/HRTF/subject_*.sofa`
- 噪声：
  - `DEMAND`
- 采样率：
  - `16 kHz`
- 单条音频长度：
  - `10 s`
- 训练片段长度：
  - `2 s`
- 总规模：
  - `18,000` 条，约 `50 h`

### 真实几何设置

这版 robust50h 数据集不是“固定接收者位置”，而是：

- 接收者平面位置随机
- 接收者高度固定 `1.5 m`
- 头朝向固定
- 声源距离 `1.0 - 1.5 m`
- 声源与接收者都在水平面
- 不变量：
  - `metadata_azimuth == HRTF_azimuth == room_source_azimuth`

对应生成脚本：

- `prepare_robust_multisubject_dataset.py`

### subject-disjoint 划分

当前 robust50h v1 使用 30 个 subject：

- train subjects: `24`
- val subjects: `3`
- test subjects unseen: `3`

其中 test subject 在训练中完全不可见，用于评估 HRTF 泛化能力。

### robust50h v5 结果（已完成）

训练配置：

- `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl.yaml`

baseline 最佳验证结果：

- best epoch: `17`
- val MAE: `10.49°`

baseline unseen-subject test（best checkpoint）：

- Accuracy: `0.6198`
- Top-3 Accuracy: `0.9028`
- MAE: `15.90°`
- Median Error: `2.50°`
- Error < 5°: `71.20%`
- Error < 10°: `82.66%`

enhanced binaural features（`configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced.yaml`）：

- best epoch: `21`
- val MAE: `10.65°`
- unseen-subject test Accuracy: `0.6432`
- unseen-subject test Top-3 Accuracy: `0.8970`
- unseen-subject test MAE: `13.77°`
- unseen-subject test Median Error: `2.50°`
- unseen-subject test Error < 5°: `72.73%`
- unseen-subject test Error < 10°: `85.28%`

结论：

- `enhanced` 在验证集上没有超过 v5 baseline（`10.65°` vs `10.49°`）
- 但在 unseen-subject test 上显著更好，尤其体现在 MAE 和长尾误差改善
- 当前 robust50h 主线推荐 checkpoint 已进一步更新为 `fbaux_only` 版本

### 当前误差画像

对 unseen-subject test 的错误分析表明，当前长尾大错主要集中在：

- `large room`
- 高 RT60（尤其 `0.65 - 0.80 s`）
- 低 SNR（尤其 `[-10, -5] dB`）
- 某些未见 subject
- 前后对称方位角附近（如 `0° / 180°` 一带）

相关输出目录：

- `outputs/analysis_multisubject_robust50h_v5_test_best`

其中已包含：

- `per_segment_errors.csv`
- `true_vs_pred_heatmap.png`
- `front_back_zone_heatmap.png`
- `front_back_confusion_rate_by_angle.png`

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

### robust50h unseen-subject 主线结果对比

| 版本 | 配置 | Accuracy | Top-3 | MAE | Median | error<5° | error<10° | 说明 |
|------|------|----------|-------|-----|--------|----------|-----------|------|
| v5 baseline | `train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl.yaml` | 0.6198 | 0.9028 | 15.90° | 2.50° | 71.20% | 82.66% | 原 robust50h 主线 |
| v5 + enhanced features | `train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced.yaml` | 0.6432 | 0.8970 | 13.77° | 2.50° | 72.73% | 85.28% | enhanced 输入特征 |
| v5 + enhanced + fbaux + focus | `train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux.yaml` | 0.6497 | **0.9277** | 14.52° | 2.50° | 73.28% | 84.58% | 粗前后判别更强，但 MAE 退化 |
| v5 + enhanced + fbfocus only | `train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbfocus_only.yaml` | 0.6446 | 0.9008 | 13.45° | 2.50° | 73.81% | 84.76% | 单独前后轴加权，收益有限 |
| **v5 + enhanced + fbaux only** | `train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux_only.yaml` | **0.6489** | 0.8832 | **11.97°** | 2.50° | **74.21%** | **86.93%** | **当前推荐 best** |

当前主结论：

- `fbaux_only` 是当前最优版本，test MAE 从 `13.77°` 进一步降到 `11.97°`
- `fbfocus_only` 单独也有一定收益，但明显弱于 `fbaux_only`
- `fbaux + focus` 组合并没有叠加增益，反而会伤害最终 MAE
- 这说明主要收益来自 `front/back auxiliary head`，而不是前后轴样本加权

推荐顺序：

1. `fbaux_only`
2. `enhanced`
3. `fbfocus_only`
4. `fbaux + focus`

## 模型架构

### 当前主线流程（fbaux_only）

```
立体声 WAV
    │
    ▼
STFT 特征提取
(log_mag_L, log_mag_R, IPD, ILD,
 ipd_sin, ipd_cos, coherence)
    │
    ▼
共享编码器（左右耳）
    │
    ▼
IPD / ILD 投影
    │
    ▼
差异先验 d_feat
    │
    ├────────► 双向交叉注意力 + attention bias
    │                 │
    │                 ▼
    │            独立残差门控
    │                 │
    ▼                 ▼
         融合特征序列
              │
              ▼
      BiGRU + attention pooling
         ├────────► 72类 DOA 分类头
         └────────► front/back 辅助头
    │
    ▼
最终输出：72 类方位角分类（主任务）
```

### 当前主线的关键改动

在 v5 主干基础上，当前推荐主线 `fbaux_only` 叠加了两类关键设计：

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

5. **增强双耳输入特征（enhanced binaural features）**
   - 在 `IPD / ILD` 之外额外使用 `sin(IPD) / cos(IPD) / coherence`
   - 提升对 unseen-subject HRTF 的泛化表现

6. **front/back 辅助头（当前主收益项）**
   - 在时序池化后的共享表示上新增 `front/back` 二分类头
   - 主任务仍是 72 类 DOA 分类，辅助任务只用于训练阶段约束
   - 当前实验表明：该辅助头显著降低了最终 MAE，并减少了长尾大错

主要模块：

- `dataset/feature_extractor.py`: 双耳频域特征提取
- `models/encoder.py`: 共享编码器
- `models/difference_prior.py`: 差异先验
- `models/cross_attention.py`: 双向交叉注意力 + attention bias
- `models/gating.py`: 独立残差门控（v5 升级）
- `models/temporal_head.py`: BiGRU + attention pooling + front/back auxiliary head
- `models/binaural_doa_net.py`: 整体模型组装（集成所有创新）
- `losses.py`: 分类 + 圆形软标签 + 可选 front/back 辅助损失

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

### 已完成结果汇总（2026-04-16）

说明：

- A0~A7 已完成 smoke（3 epochs）多种子评估（seeds=42,43），并统计均值/方差。
- 汇总文件：`outputs/ablation_smoke_multiseed_summary.md`
- A4 已完成 full train + test，可用于更可靠结论。

#### A0~A7 smoke test（3 epochs, 2 seeds）

| 组别 | Accuracy (mean±std) | Top-3 (mean±std) | MAE (mean±std) | Median (mean±std) | error<5° (mean±std) | error<10° (mean±std) |
|------|----------------------|------------------|----------------|-------------------|----------------------|-----------------------|
| A0 | 0.5030 ± 0.0071 | 0.7997 ± 0.0015 | 13.7389 ± 0.1087 | 2.5000 ± 0.0000 | 0.6459 ± 0.0149 | 0.7916 ± 0.0044 |
| A1 | 0.4744 ± 0.0090 | 0.7767 ± 0.0103 | 13.8773 ± 0.2205 | 2.6010 ± 0.1087 | 0.6160 ± 0.0132 | 0.7724 ± 0.0083 |
| A2 | 0.5153 ± 0.0084 | 0.8079 ± 0.0085 | 14.2571 ± 0.6599 | 2.5000 ± 0.0000 | 0.6542 ± 0.0134 | 0.7887 ± 0.0081 |
| A3 | 0.5017 ± 0.0153 | 0.8030 ± 0.0059 | 13.6417 ± 0.2337 | 2.5000 ± 0.0000 | 0.6400 ± 0.0127 | 0.7916 ± 0.0011 |
| A4 | 0.5034 ± 0.0022 | 0.8034 ± 0.0067 | 12.8603 ± 0.0069 | 2.5000 ± 0.0000 | 0.6498 ± 0.0054 | 0.7945 ± 0.0018 |
| A5 | 0.5073 ± 0.0141 | 0.7945 ± 0.0092 | 13.7073 ± 0.2024 | 2.5000 ± 0.0000 | 0.6427 ± 0.0207 | 0.7875 ± 0.0190 |
| A6 | 0.5017 ± 0.0151 | 0.7920 ± 0.0265 | 13.9644 ± 1.0696 | 2.5000 ± 0.0000 | 0.6374 ± 0.0160 | 0.7760 ± 0.0246 |
| A7 | 0.5160 ± 0.0103 | 0.8018 ± 0.0123 | 13.5697 ± 0.2062 | 2.5000 ± 0.0000 | 0.6555 ± 0.0017 | 0.7916 ± 0.0103 |

多种子结论（smoke 阶段）：

- MAE 最优且最稳定的是 A4（12.8603 ± 0.0069，方差最小）。
- Accuracy 与 error<5° 均值最高的是 A7（0.5160 与 0.6555）。
- 波动最大的是 A6（MAE std=1.0696），提示其对随机种子敏感。

#### A4 full train + test（已完成）

- 最佳验证点：epoch 34，val MAE=8.47°（早停 patience=15）
- 测试集：
  - Accuracy: 0.7055
  - Top-3 Accuracy: 0.8687
  - MAE: 9.1611°
  - Median Error: 2.2383°
  - Error < 5°: 78.93%
  - Error < 10°: 86.79%

对比当前 v5 full（acc=0.7344, MAE=8.72°, error<5°=80.5%）：

- A4（仅 circular soft label）明显优于基础弱配置，但仍弱于完整 v5。
- 说明 circular soft label 是强收益项，但不足以单独替代完整结构。

### 基于结果的改动建议（下一步）

#### 1) 保留完整 v5 作为主线，A4 作为强单项对照

- A7 在多种子 smoke 下分类指标更好，A4 在 MAE 与稳定性更优。
- 训练主线建议保持 A7（完整结构），并持续以 A4 做单项强基线对照。

#### 2) smoke 规范固定为「3~5 epochs + >=2 seeds + 均值/方差」

- 当前脚本已支持：`SMOKE_EPOCHS=3~5` 与 `seeds_csv`。
- 统一使用示例：

```bash
SMOKE_EPOCHS=5 bash run_cipic_reverb_demand50h_ablation_pipeline.sh a7 smoke 42,43
```

#### 3) full 阶段优先做协同增益拆分

建议顺序：

1. A7 full（完整 v5）
2. A6 full（去掉 circular soft label）
3. A5 full（去掉 pooling，只保留 bias+门控）

以 A7-A6 量化 circular soft label 的组合增益，以 A6-A5 量化 attention pooling 的组合增益。

#### 4) 指标口径

- 主指标：`mean_angular_error`、`error_lt_5`、`error_lt_10`。
- 辅助指标：`accuracy`、`top_k_accuracy`。
- 已新增宏平均分类指标：`macro_precision`、`macro_recall`、`macro_f1`（见 `metrics.py`）。

## 数据与实验主线

## 泛化诊断实验（speaker/noise/subject）

已提供可直接执行的三组诊断脚本：

- speaker overlap vs disjoint
- clean-trained vs mixed-trained
- single-subject vs cross-subject

详细步骤与命令见：

- `docs/GENERALIZATION_DIAGNOSTICS.md`

核心脚本：

- `tools/diagnostics/prepare_librispeech_speaker_splits.py`
- `tools/diagnostics/prepare_diagnostic_datasets.sh`
- `tools/diagnostics/run_generalization_diagnostics.sh`

### 已完成诊断结果（2026-04-21）

诊断设置：`DIAG_EPOCHS=3`，seeds=`42,43`。

结果文件：

- `outputs/diagnostics/speaker_overlap_vs_disjoint.md`
- `outputs/diagnostics/clean_trained_vs_mixed_trained.md`
- `outputs/diagnostics/single_subject_vs_cross_subject.md`
- `docs/GENERALIZATION_RESULTS_REPORT.md`

关键结论（以 MAE 差值为主）：

1. 噪声/混响敏感性最高：
  - `30.8971° (clean-trained on robust test) - 12.3871° (mixed-trained on robust test) = +18.5100°`
2. HRTF 泛化敏感性次高：
  - `36.0037° (single-subject on unseen subject) - 19.6063° (cross-subject on unseen subject) = +16.3975°`
3. speaker overlap/disjoint 在本轮中未表现为主要瓶颈：
  - `32.2503° (disjoint) - 34.2873° (overlap) = -2.0370°`

诊断优先级：

1. 优先优化噪声/混响鲁棒性
2. 其次优化 cross-subject HRTF 泛化
3. 对 speaker 诊断做更严格配平后再复验

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
- 50h robust multisubject（subject-disjoint, unseen-subject）
  - `data/librispeech_cipic_multisubject_robust50h_v1`

### 当前主线配置说明

当前 robust50h unseen-subject 主线相关配置分为 4 条：

- baseline：
  - `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl.yaml`
- enhanced：
  - `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced.yaml`
- front/back 组合版：
  - `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux.yaml`
- 当前推荐主线 `fbaux_only`：
  - `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux_only.yaml`
- 对照消融 `fbfocus_only`：
  - `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbfocus_only.yaml`

相对 `enhanced`，`fbaux_only` 的配置改动是：

- `model.use_front_back_auxiliary: true`
- `train.front_back_aux_weight: 0.3`
- `train.front_back_focus_weight: 0.0`

也就是说，当前主线保留了：

- enhanced binaural features
- v5 attention bias / 独立残差门控 / attention pooling / circular soft label
- front/back auxiliary head

同时去掉了：

- front/back axis sample weighting

### 划分比例

统一采用：

- train: 70%
- val: 15%
- test: 15%

对于 robust50h 多 subject 主线，使用的是 **subject-disjoint 显式 split-root**：

- `train_root`
- `val_root`
- `test_root`

此时不会再在单个 root 内做随机 70/15/15 划分。

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

如果要训练多 subject / unseen-subject 泛化主线：

```bash
python train.py --config configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl.yaml
```

如果要做“只开启 enhanced binaural features”的单变量对照实验：

```bash
python train.py --config configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced.yaml
```

如果要运行当前推荐主线 `fbaux_only`：

```bash
python train.py --config configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux_only.yaml
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

评估 robust50h unseen-subject test：

```bash
python evaluate.py \
  --checkpoint outputs/checkpoints_multisubject_robust50h_v5_bias_gating_attnpool_csl_fast/best.pth \
  --config configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl.yaml \
  --output.log_dir outputs/logs_multisubject_robust50h_v5_bias_gating_attnpool_csl_fast_test_best
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

推荐使用 **v5 + enhanced + fbaux_only（当前 unseen-subject 主线）**：

- `configs/train_librispeech_subject003_cipic_reverb_demand50h_v5_bias_gating_attnpool_csl.yaml`
- `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux_only.yaml`

核心配置特点：

```yaml
model:
  use_attention_bias: true         # 低秩双向 attention bias
  attention_bias_rank: 16
  use_attention_pooling: true      # BiGRU 后 attention pooling
  use_gating: true                 # 双向独立残差门控
  use_enhanced_binaural_features: true
  use_front_back_auxiliary: true
  gru_hidden_size: 128
  
train:
  lr: 0.0005                       # 保守学习率
  amp: false                        # 关闭 AMP，稳定数值
  grad_clip: 1.0                   # 严格梯度裁剪
  circular_soft_label_weight: 0.2  # 圆形软标签权重
  circular_kappa: 4.0              # von Mises 浓度参数
  anti_confusion_weight: 1.0       # 前后对向惩罚
  front_back_aux_weight: 0.3       # front/back 辅助头权重
  front_back_focus_weight: 0.0     # 当前主线不使用 axis weighting
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
├── prepare_robust_multisubject_dataset.py
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
│   ├── train_librispeech_subject003_cipic_reverb_demand50h_v5_bias_gating_attnpool_csl.yaml
│   ├── train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl.yaml
│   ├── train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced.yaml
│   ├── train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux.yaml
│   ├── train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux_only.yaml  ★
│   └── train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbfocus_only.yaml
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

最后更新：2026-04-26  

当前推荐 best（subject_003 主线）：  
`outputs/checkpoints_librispeech_subject003_cipic_reverb_demand50h_v5_bias_gating_attnpool_csl/best.pth`

当前推荐 best（robust50h unseen-subject 主线）：  
`outputs/checkpoints_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux_only/best.pth`

**推荐配置：**

- `configs/train_librispeech_subject003_cipic_reverb_demand50h_v5_bias_gating_attnpool_csl.yaml`
- `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux_only.yaml`

**关键指标：**

- subject_003 v5 测试集：
  - Accuracy: `0.7344`
  - Top-3 Accuracy: `0.8744`
  - MAE: `8.72°`
  - Median Error: `2.14°`
  - Error < 5° 占比: `80.5%`

- robust50h v5 unseen-subject 测试集：
  - Accuracy: `0.6432`
  - Top-3 Accuracy: `0.8970`
  - MAE: `13.77°`
  - Median Error: `2.50°`
  - Error < 10° 占比: `85.28%`
