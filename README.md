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
   - 训练稳定化（amp=false, grad_clip=1.0, lr下降）
   - 启用前后对向混淆惩罚（训练损失内）

### 近期关键结果

- 50h 混响+DEMAND v1（恢复后 best 测试）
  - accuracy: 0.4641
  - top3: 0.7462
  - MAE: 16.9202°
  - median: 2.7107°

- 50h 混响+DEMAND v2（完整训练 best 正式测试）
  - accuracy: 0.7067
  - top3: 0.8649
  - MAE: 9.6673°
  - median: 2.2356°

v2 相比 v1 在该任务上有显著改善，主要体现在 MAE、Top-1、Top-3 和低误差样本占比。

## 模型架构

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

主要模块：

- `dataset/feature_extractor.py`: 双耳频域特征提取
- `models/encoder.py`: 共享编码器
- `models/difference_prior.py`: 差异先验
- `models/cross_attention.py`: 双向交叉注意力
- `models/gating.py`: 门控融合
- `models/temporal_head.py`: 时序建模与分类输出
- `models/binaural_doa_net.py`: 整体模型组装

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
- 50h reverb+DEMAND（v2，混合难度）
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

使用当前降误差 v2 配置训练：

```bash
python train.py --config configs/train_librispeech_subject003_cipic_reverb_demand50h_v2.yaml
```

从 best 恢复并使用安全参数：

```bash
python train.py \
  --config configs/train_librispeech_subject003_cipic_reverb_demand50h_v2.yaml \
  --resume outputs/checkpoints_librispeech_subject003_cipic_reverb_demand50h_v2/best.pth \
  --train.amp false \
  --train.grad_clip 1.0
```

### 3. 评估

```bash
python evaluate.py \
  --checkpoint outputs/checkpoints_librispeech_subject003_cipic_reverb_demand50h_v2/best.pth \
  --config configs/train_librispeech_subject003_cipic_reverb_demand50h_v2.yaml \
  --output.log_dir outputs/logs_librispeech_subject003_cipic_reverb_demand50h_v2_test_full_best
```

### 4. 一键流水线

- 50h CIPIC + 混响：

```bash
bash run_cipic_reverb50h_pipeline.sh
```

- 50h CIPIC + 混响 + DEMAND（v1）：

```bash
bash run_cipic_reverb_demand50h_pipeline.sh
```

- 50h CIPIC + 混响 + DEMAND（v2 降误差）：

```bash
bash run_cipic_reverb_demand50h_v2_pipeline.sh
```

## 当前推荐配置

推荐使用：

- `configs/train_librispeech_subject003_cipic_reverb_demand50h_v2.yaml`

该配置核心策略：

- 更保守学习率（`lr: 0.0005`）
- 关闭 AMP（`amp: false`）
- 更严格梯度裁剪（`grad_clip: 1.0`）
- 启用训练期前后对向惩罚（`anti_confusion_weight: 1.0`）

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
├── configs/
│   ├── default.yaml
│   ├── train_librispeech_subject003_cipic_reverb50h.yaml
│   ├── train_librispeech_subject003_cipic_reverb_demand50h.yaml
│   └── train_librispeech_subject003_cipic_reverb_demand50h_v2.yaml
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
- 如需回溯历史实验，可查看 `outputs/` 下对应日志目录。

---

最后更新：2026-04-10
当前推荐 best：
`outputs/checkpoints_librispeech_subject003_cipic_reverb_demand50h_v2/best.pth`
