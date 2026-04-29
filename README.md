# 双耳 DOA-Net

基于 `LibriSpeech + CIPIC HRTF + 房间混响 + DEMAND 噪声` 的双耳声源到达方向（DOA）估计项目。  
当前 README 以 **robust50h 多 subject、subject-disjoint unseen-subject 泛化实验** 为主线；早期 `subject_003` 单 subject 路线仅保留阶段性结果，不再作为文档主轴。

## 主线概览

当前主线任务是：

- 使用 `30` 个 CIPIC subject 构建多 subject 双耳数据集
- 按 `24 / 3 / 3` 做 `train / val / test` 的 subject-disjoint 划分
- 在 unseen-subject test 上评估 HRTF 泛化
- 重点关注：
  - `MAE`
  - `front_back_error_rate`
  - `opposite_error_rate`
  - `large_error_rate`
  - `front / back / side MAE`

当前推荐的**原生 DOA-Net 主线**是：

- `v5 + enhanced + fbaux_only + cohfix + no-csl`
- 配置：
  - `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_nocsl_enhanced_fbaux_only_cohfix.yaml`
- checkpoint：
  - `outputs/checkpoints_multisubject_robust50h_v5_bias_gating_attnpool_nocsl_enhanced_fbaux_only_cohfix/best.pth`

当前综合表现最强的**外部 backbone 对照**是：

- `SDEL-DOA-Cls + fbaux`
- 配置：
  - `configs/train_librispeech_multisubject_robust50h_sdel_doa_cls_fbaux.yaml`
- checkpoint：
  - `outputs/checkpoints_multisubject_robust50h_sdel_doa_cls_fbaux_nw8_gpu1/best.pth`

## 数据集主线

### robust50h 多 subject 数据集

- 根目录：
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

### 几何与合成设置

这版 robust50h 数据集采用：

- 接收者平面位置随机
- 接收者高度固定 `1.5 m`
- 头朝向固定
- 声源距离 `1.0 - 1.5 m`
- 声源与接收者均在水平面
- `metadata_azimuth == HRTF_azimuth == room_source_azimuth`

对应脚本：

- `prepare_robust_multisubject_dataset.py`

### subject-disjoint 划分

- train subjects: `24`
- val subjects: `3`
- test subjects unseen: `3`

test subjects 在训练阶段完全不可见，用于评估 unseen-subject HRTF 泛化。

## 历史演进：v2 到 v5

下面这组结果保留为项目演进记录，用来说明模型从早期混合难度数据到 v5 主干的改进轨迹。  
这一部分是**历史阶段结果**，不是当前文档主线。

| 版本 | 数据配置 | Accuracy | Top-3 | MAE | Median | error<5° | 关键变化 |
|---|---|---:|---:|---:|---:|---:|---|
| v2 | 混合难度 | 0.7067 | 0.8649 | 9.67° | 2.24° | 77.6% | 数据混合、训练稳定化 |
| v3 | 混合难度 + 回归分支 | 0.7280 | 0.8695 | 9.69° | 2.42° | 72.4% | 分类 + DOA 回归 |
| v4 | 混合难度 + 增强特征 | 0.6952 | 0.8692 | 9.86° | 2.24° | 77.3% | `sin/cos(IPD)` 等增强特征 |
| v5 | 混合难度 + 创新主干 | **0.7344** | **0.8744** | **8.72°** | **2.14°** | **80.5%** | attention bias + 独立门控 + attention pooling |

阶段结论：

- `v5` 是单 subject 阶段最强主干
- `v4` 说明增强双耳特征有价值，但仅靠输入增强不足以替代结构改动
- `v3` 的回归分支在当时没有稳定带来 MAE 改善

## robust50h unseen-subject 主线结果

### 主线结果对比

| 版本 | 配置 | Accuracy | Top-3 | MAE | Median | error<5° | error<10° | 说明 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| v5 baseline | `train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl.yaml` | 0.6198 | 0.9028 | 15.90° | 2.50° | 71.20% | 82.66% | 原 robust50h 主线 |
| v5 + enhanced | `train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced.yaml` | 0.6432 | 0.8970 | 13.77° | 2.50° | 72.73% | 85.28% | enhanced 输入特征 |
| v5 + enhanced + fbaux_only | `train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux_only.yaml` | 0.6489 | 0.8832 | 11.97° | 2.50° | 74.21% | 86.93% | 旧 coherence 条件下最佳 |
| v5 + enhanced + fbaux_only + cohfix | `train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux_only_cohfix.yaml` | **0.6750** | 0.9180 | 12.70° | 2.50° | 74.37% | 86.96% | 修正 coherence 后更稳 |
| **v5 + enhanced + fbaux_only + cohfix + no-csl** | `train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_nocsl_enhanced_fbaux_only_cohfix.yaml` | 0.6661 | **0.9317** | **11.92°** | 2.50° | **75.21%** | **87.20%** | **当前推荐原生主线** |
| v5 + enhanced + fbaux_only + cohfix + reg | `train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_reg_enhanced_fbaux_only_cohfix.yaml` | 0.6640 | 0.9360 | 12.40° | 3.54° | 63.83% | 86.09% | 多任务分类+回归，前向更稳但整体 MAE 未优于 no-csl |
| v5 + enhanced + fbaux_only + cohfix + csl-nols | `train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_nols_enhanced_fbaux_only_cohfix.yaml` | 0.6563 | 0.9096 | 15.14° | 2.50° | 74.18% | 84.03% | 去掉 label smoothing 明显退化 |

### 当前主结论

- `front/back auxiliary head` 是当前最有效的结构改动
- `coherence` 修正后，主线在 unseen-subject test 上更稳
- 在修正后的主线上：
  - `label smoothing` 应保留
  - `circular soft label` 会伤害 unseen-subject test 泛化
- 在原生 DOA-Net 主线上直接加入多任务回归分支：
  - 可以改善 `front_back_error_rate` 和 `opposite_error_rate`
  - 但没有进一步降低整体 `MAE`
- 因此当前推荐主线是：
  - `cohfix + no-csl + fbaux_only`

### fbaux_only 权重 sweep

| aux weight | 配置 | best val MAE | test Accuracy | test Top-3 | test MAE | test std | error<5° | error<10° | 说明 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `0.10` | `..._fbaux_w010.yaml` | 10.28° | **0.6563** | 0.8979 | 12.84° | 32.84 | 73.14% | 85.80% | 准确率较高，但 MAE 不如 0.30 |
| `0.15` | `..._fbaux_w015.yaml` | 9.50° | 0.6187 | 0.9009 | 14.97° | 35.73 | 69.30% | 82.87% | 验证集最优，但 test 泛化最差 |
| `0.20` | `..._fbaux_w020.yaml` | 10.27° | 0.6487 | **0.9229** | 13.71° | 35.11 | **74.94%** | 86.10% | Top-3 更高，但 MAE 不如 0.30 |
| **`0.30`** | `..._fbaux_only.yaml` | **9.09°** | 0.6489 | 0.8832 | **11.97°** | **31.51** | 74.21% | **86.93%** | 旧 coherence 条件下最佳设置 |

这轮 sweep 的结论是：

- 小权重更容易抬高 `accuracy / top-3`
- 但 `MAE` 和长尾误差仍然由 `0.30` 更占优
- 当前文档主线已经转到 `cohfix + no-csl`，这组 sweep 主要作为历史对照保留

## 外部 baseline：Moving-Binaural-SDEL 适配结果

为了和 `Moving-Binaural-SDEL` 的核心结构做公平比较，项目里实现了两条适配 baseline：

- `SDEL-DOA-Reg`
  - 输入：`MBMS proxy + ILD + cos(IPD) + sin(IPD)`
  - 输出：二维单位向量回归
- `SDEL-DOA-Cls`
  - 输入同上
  - 输出：72 类分类

### 结果对比

| 模型 | 输出形式 | Accuracy | Top-3 | MAE | Median | FB err | Opp err | Large err |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 当前主线 `cohfix + no-csl + fbaux_only` | 72类分类 | 0.6661 | 0.9317 | 11.92° | 2.50° | 0.0967 | 0.0089 | 0.0667 |
| `SDEL-DOA-Reg` | 向量回归 | 0.4112 | 0.8651 | **7.42°** | **2.25°** | **0.0428** | **0.0022** | **0.0316** |
| `SDEL-DOA-Cls` | 72类分类 | **0.6910** | **0.9558** | 12.77° | 2.50° | 0.0704 | 0.0118 | 0.0620 |
| **`SDEL-DOA-Cls + fbaux`** | 72类分类 | **0.7201** | **0.9727** | **9.02°** | 2.50° | **0.0422** | 0.0053 | **0.0381** |
| `DOA-Net mainline + reg` | 分类+角度回归 | 0.6640 | 0.9360 | 12.40° | 3.54° | 0.0892 | 0.0057 | 0.0679 |

这组结果说明：

- `SDEL` 风格 CRNN 主干本身很强
- 回归目标非常有利于降低 `MAE` 和前后大错
- `fbaux` 在外部 backbone 上同样显著有效，说明它具有较好的 backbone-agnostic 特性
- 当前综合最强的分类模型已经变成：
  - `SDEL-DOA-Cls + fbaux`
- 当前最值得继续验证的是：
  - `SDEL-DOA-Reg + fbaux`
  - `SDEL backbone + 分类/回归联合任务`

## 当前主线模型结构

### 流程

```text
立体声 WAV
  -> STFT 特征提取
  -> (log_mag_L, log_mag_R, IPD, ILD, sin/cos(IPD), coherence)
  -> 左右耳共享编码器
  -> IPD / ILD 投影
  -> 差异先验 d_feat
  -> 双向交叉注意力 + attention bias
  -> 独立残差门控
  -> 融合特征序列
  -> BiGRU + attention pooling
  -> 72类 DOA 分类头
  -> front/back 辅助头
```

### 当前主线关键点

1. `attention bias`
   - 用差异先验显式调制双向交叉注意力
2. `独立残差门控`
   - 左右两个方向分别学习 gate
3. `attention pooling`
   - 替代简单 mean pooling
4. `enhanced binaural features`
   - 使用 `sin/cos(IPD)` 与修正后的 `coherence`
5. `front/back auxiliary`
   - 显式学习前后半平面
6. `no-csl`
   - 当前主线关闭 `circular soft label`

相关实现：

- `dataset/feature_extractor.py`
- `models/encoder.py`
- `models/difference_prior.py`
- `models/cross_attention.py`
- `models/gating.py`
- `models/temporal_head.py`
- `models/binaural_doa_net.py`
- `losses.py`

## 当前数据与配置

### 当前主用数据集

- `data/librispeech_cipic_multisubject_robust50h_v1`

### 当前主用配置

- baseline：
  - `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl.yaml`
- enhanced：
  - `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced.yaml`
- 当前推荐主线：
  - `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_nocsl_enhanced_fbaux_only_cohfix.yaml`
- baseline 对照：
  - `configs/train_librispeech_multisubject_robust50h_sdel_doa_reg_baseline.yaml`
  - `configs/train_librispeech_multisubject_robust50h_sdel_doa_cls_baseline.yaml`
  - `configs/train_librispeech_multisubject_robust50h_sdel_doa_cls_fbaux.yaml`
  - `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_reg_enhanced_fbaux_only_cohfix.yaml`

### 主线配置变化

相对 `enhanced`，当前推荐主线的关键变化是：

- `model.use_front_back_auxiliary: true`
- `train.front_back_aux_weight: 0.3`
- `train.front_back_focus_weight: 0.0`
- `train.circular_soft_label_weight: 0.0`

也就是说它保留：

- enhanced binaural features
- v5 attention bias / 独立残差门控 / attention pooling
- front/back auxiliary head
- label smoothing

同时去掉：

- front/back axis sample weighting
- circular soft label

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练当前推荐主线

```bash
python train.py \
  --config configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_nocsl_enhanced_fbaux_only_cohfix.yaml
```

### 评估当前推荐主线

```bash
python evaluate.py \
  --checkpoint outputs/checkpoints_multisubject_robust50h_v5_bias_gating_attnpool_nocsl_enhanced_fbaux_only_cohfix/best.pth \
  --config configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_nocsl_enhanced_fbaux_only_cohfix.yaml \
  --output.log_dir outputs/logs_multisubject_robust50h_v5_bias_gating_attnpool_nocsl_enhanced_fbaux_only_cohfix_test_best
```

### 训练 SDEL baseline

回归版：

```bash
python train.py \
  --config configs/train_librispeech_multisubject_robust50h_sdel_doa_reg_baseline.yaml
```

分类版：

```bash
python train.py \
  --config configs/train_librispeech_multisubject_robust50h_sdel_doa_cls_baseline.yaml
```

分类版 + `fbaux`：

```bash
python train.py \
  --config configs/train_librispeech_multisubject_robust50h_sdel_doa_cls_fbaux.yaml
```

### 训练 DOA-Net 回归主线

```bash
python train.py \
  --config configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_reg_enhanced_fbaux_only_cohfix.yaml
```

## 项目结构

```text
DOA-net/
├── README.md
├── train.py
├── evaluate.py
├── infer.py
├── losses.py
├── metrics.py
├── prepare_robust_multisubject_dataset.py
├── configs/
│   ├── default.yaml
│   ├── train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl.yaml
│   ├── train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced.yaml
│   ├── train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux_only.yaml
│   ├── train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux_only_cohfix.yaml
│   ├── train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_nocsl_enhanced_fbaux_only_cohfix.yaml  ★
│   ├── train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_reg_enhanced_fbaux_only_cohfix.yaml
│   ├── train_librispeech_multisubject_robust50h_sdel_doa_reg_baseline.yaml
│   ├── train_librispeech_multisubject_robust50h_sdel_doa_cls_baseline.yaml
│   └── train_librispeech_multisubject_robust50h_sdel_doa_cls_fbaux.yaml
├── dataset/
├── engine/
├── models/
├── utils/
├── data/
└── outputs/
```

## 当前推荐结果

最后更新：`2026-04-29`

当前推荐 best（原生 DOA-Net 主线）：

- checkpoint：
  - `outputs/checkpoints_multisubject_robust50h_v5_bias_gating_attnpool_nocsl_enhanced_fbaux_only_cohfix/best.pth`
- config：
  - `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_nocsl_enhanced_fbaux_only_cohfix.yaml`

关键指标：

- Accuracy: `0.6661`
- Top-3 Accuracy: `0.9317`
- MAE: `11.92°`
- Median Error: `2.50°`
- Error < 10°: `87.20%`
- Front/back error rate: `0.0967`
- Opposite error rate: `0.0089`

当前 strongest classification 对照（外部 backbone）：

- checkpoint：
  - `outputs/checkpoints_multisubject_robust50h_sdel_doa_cls_fbaux_nw8_gpu1/best.pth`
- config：
  - `configs/train_librispeech_multisubject_robust50h_sdel_doa_cls_fbaux.yaml`

关键指标：

- Accuracy: `0.7201`
- Top-3 Accuracy: `0.9727`
- MAE: `9.02°`
- Front/back error rate: `0.0422`
- Opposite error rate: `0.0053`

当前文档只保留 `v2-v5` 的阶段性结果记录，不再展开 `subject_003` 单 subject 分析与早期消融细节。若需回看历史实验，可直接查阅 `outputs/` 下对应日志目录。
