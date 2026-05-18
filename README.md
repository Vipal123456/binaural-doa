# 双耳 DOA-Net

基于 `LibriSpeech + CIPIC HRTF + 房间混响 + DEMAND 噪声` 的双耳声源到达方向（DOA）估计项目。

当前仓库已经收束到 `robust50h 多 subject / subject-disjoint unseen-subject` 主线。早期 `subject003` 单 subject 数据、旧诊断流水线和部分历史绘图脚本已经从当前工作树移除，README 也只保留还存在的配置、脚本和输出。

## 当前主线

- 数据协议：`30` 个 CIPIC subjects，按 `24 / 3 / 3` 划分 `train / val / unseen-test`
- 主目标：在 unseen-subject test 上评估 HRTF 泛化
- 当前评估重点：
  - `accuracy`
  - `f1_score`
  - `mean_angular_error`
  - `acc_at_5deg`
  - `acc_at_10deg`
  - `front_back_halfplane_error_rate`
  - `opposite_error_rate`
  - `large_error_rate`
  - `front / back / side MAE`

当前推荐的原生 DOA-Net 主线：

- 配置：
  - `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_nocsl_enhanced_fbaux_only_cohfix.yaml`
- checkpoint：
  - `outputs/checkpoints_multisubject_robust50h_v5_bias_gating_attnpool_nocsl_enhanced_fbaux_only_cohfix/best.pth`
- test log：
  - `outputs/logs_multisubject_robust50h_v5_bias_gating_attnpool_nocsl_enhanced_fbaux_only_cohfix_test_best_workers4/train.log`

当前最强外部分类 baseline：

- 配置：
  - `configs/train_librispeech_multisubject_robust50h_sdel_doa_cls_fbaux.yaml`
- checkpoint：
  - `outputs/checkpoints_multisubject_robust50h_sdel_doa_cls_fbaux_nw8_gpu1/best.pth`
- test log：
  - `outputs/logs_multisubject_robust50h_sdel_doa_cls_fbaux_nw8_gpu1_test_best_workers8/train.log`

当前推荐的轻量 native `v7` 结果分成三条：

- `MAE` 更优的轻量主线：
  - 配置：
    - `configs/train_librispeech_multisubject_robust50h_v7_native_lite_encoderv2_balanced_nocsl_fbaux_cohfix.yaml`
  - checkpoint：
    - `outputs/checkpoints_multisubject_robust50h_v7_native_lite_encoderv2_balanced_nocsl_fbaux_cohfix/best.pth`
  - test 结果：
    - `accuracy = 0.6644`
    - `f1_score = 0.3856`
    - `mean_angular_error = 12.91°`
    - `acc_at_5deg = 0.7526`
    - `acc_at_10deg = 0.8550`

- `Acc / F1 / Acc@5° / Acc@10°` 更优的轻量 cue 独立流主线：
  - 配置：
    - `configs/train_librispeech_multisubject_robust50h_v7_litecueenc_concat_all_nocsl_fbaux_cohfix.yaml`
  - checkpoint：
    - `outputs/checkpoints_multisubject_robust50h_v7_litecueenc_concat_all_nocsl_fbaux_cohfix/best.pth`
  - test 结果：
    - `accuracy = 0.7133`
    - `f1_score = 0.4264`
    - `mean_angular_error = 13.89°`
    - `acc_at_5deg = 0.7800`
    - `acc_at_10deg = 0.8656`

- 当前更均衡、最值得继续推进的轻量候选主线：
  - 配置：
    - `configs/train_librispeech_multisubject_robust50h_v7_litecueenc_concat_all_cf80_cue24_gru80_nocsl_fbaux_cohfix.yaml`
  - checkpoint：
    - `outputs/checkpoints_multisubject_robust50h_v7_litecueenc_concat_all_cf80_cue24_gru80_nocsl_fbaux_cohfix/best.pth`
  - 三次 seed 复验：
    - `seed42: accuracy = 0.6664, f1_score = 0.3837, mean_angular_error = 12.09°, acc_at_5deg = 0.7446, acc_at_10deg = 0.8573`
    - `seed43: accuracy = 0.7022, f1_score = 0.4062, mean_angular_error = 12.90°, acc_at_5deg = 0.7709, acc_at_10deg = 0.8731`
    - `seed44: accuracy = 0.6630, f1_score = 0.3805, mean_angular_error = 12.75°, acc_at_5deg = 0.7340, acc_at_10deg = 0.8571`
  - 三次 seed 平均：
    - `accuracy = 0.6772`
    - `f1_score = 0.3901`
    - `mean_angular_error = 12.58°`
    - `acc_at_5deg = 0.7498`
    - `acc_at_10deg = 0.8625`

## 数据集

主用数据集根目录：

- `data/librispeech_cipic_multisubject_robust50h_v1`

关键设置：

- 语音：`LibriSpeech/train-clean-100`
- HRTF：`/disk2/bywang/data/HRTF/subject_*.sofa`
- 噪声：`DEMAND`
- 采样率：`16 kHz`
- 单条音频长度：`10 s`
- 训练片段长度：`2 s`
- 总规模：`18,000` 条，约 `50 h`

对应数据合成脚本：

- `prepare_robust_multisubject_dataset.py`

## 当前结果

### 核心结果

| 模型 | 配置 | Accuracy | F1-score | MAE | Acc@5° | Acc@10° | FB err | Opp err | Large err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原生主线 `v5 + enhanced + fbaux_only + cohfix + no-csl` | `train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_nocsl_enhanced_fbaux_only_cohfix.yaml` | 0.6661 | 0.3828 | 11.92° | 0.7429 | 0.8720 | 0.0967 | 0.0089 | 0.0667 |
| 轻量主线 `v7 + encoder v2 balanced + fbaux` | `train_librispeech_multisubject_robust50h_v7_native_lite_encoderv2_balanced_nocsl_fbaux_cohfix.yaml` | 0.6644 | 0.3856 | 12.91° | 0.7526 | 0.8550 | 0.0916 | 0.0064 | 0.0731 |
| 轻量 cue 主线 `v7 + lite cue encoder all + fbaux` | `train_librispeech_multisubject_robust50h_v7_litecueenc_concat_all_nocsl_fbaux_cohfix.yaml` | 0.7133 | 0.4264 | 13.89° | 0.7800 | 0.8656 | 0.1078 | 0.0088 | 0.0821 |
| 轻量候选主线 `v7 + lite cue + cf80 + cue24 + gru80 (seed42)` | `train_librispeech_multisubject_robust50h_v7_litecueenc_concat_all_cf80_cue24_gru80_nocsl_fbaux_cohfix.yaml` | 0.6664 | 0.3837 | 12.09° | 0.7446 | 0.8573 | 0.0828 | 0.0097 | 0.0697 |
| 轻量候选主线 `v7 + lite cue + cf80 + cue24 + gru80 (seed43)` | `train_librispeech_multisubject_robust50h_v7_litecueenc_concat_all_cf80_cue24_gru80_seed43_nocsl_fbaux_cohfix.yaml` | 0.7022 | 0.4062 | 12.90° | 0.7709 | 0.8731 | 0.0982 | 0.0106 | 0.0730 |
| 轻量候选主线 `v7 + lite cue + cf80 + cue24 + gru80 (seed44)` | `train_librispeech_multisubject_robust50h_v7_litecueenc_concat_all_cf80_cue24_gru80_seed44_nocsl_fbaux_cohfix.yaml` | 0.6630 | 0.3805 | 12.75° | 0.7340 | 0.8571 | 0.0879 | 0.0068 | 0.0666 |
| 轻量候选主线 `v7 + lite cue + cf80 + cue24 + gru80 (3-seed avg)` | `seed42 / seed43 / seed44` | 0.6772 | 0.3901 | 12.58° | 0.7498 | 0.8625 | 0.0896 | 0.0090 | 0.0698 |
| `DOA-Net pure-reg + fbaux` | `train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_pure_reg_enhanced_fbaux_only_cohfix.yaml` | 0.4457 | 0.3196 | 11.01° | 0.7438 | 0.8834 | 0.0816 | 0.0063 | 0.0578 |
| `DOA-Net cls + reg + fbaux` | `train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_reg_enhanced_fbaux_only_cohfix.yaml` | 0.6640 | 0.3772 | 12.40° | 0.7380 | 0.8609 | 0.0892 | 0.0057 | 0.0679 |
| `SDEL-DOA-Reg` | `train_librispeech_multisubject_robust50h_sdel_doa_reg_baseline.yaml` | 0.4112 | 0.3202 | **7.42°** | 0.7780 | 0.9246 | 0.0428 | 0.0022 | 0.0316 |
| `SDEL-DOA-Cls` | `train_librispeech_multisubject_robust50h_sdel_doa_cls_baseline.yaml` | 0.6910 | 0.3897 | 12.77° | 0.7514 | 0.8843 | 0.0704 | 0.0118 | 0.0620 |
| `SDEL-DOA-Cls + fbaux` | `train_librispeech_multisubject_robust50h_sdel_doa_cls_fbaux.yaml` | **0.7201** | **0.4127** | **9.02°** | **0.8032** | **0.9221** | **0.0422** | 0.0053 | **0.0381** |

### 当前结论

- `front/back auxiliary head` 仍然是最稳定、最有迁移性的改动。
- `coherence` 修正后，原生主线在 unseen-subject test 上更稳。
- 在原生 backbone 上：
  - `no-csl + fbaux_only + cohfix` 是当前最均衡的分类主线。
  - `v7 + encoder v2 balanced` 是当前更推荐的轻量 `MAE` 主线；相比原始 `v7 native_lite`，`F1 / MAE / Acc@5° / Acc@10°` 全部提升。
  - `v7 + lite cue encoder all` 是当前更强的轻量分类主线；`Accuracy / F1 / Acc@5° / Acc@10°` 全部高于 `encoder v2 balanced`，但 `MAE` 更差，且 `front/back` 与 `large error` 更高。
  - `v7 + lite cue + cf80 + cue24 + gru80` 是当前更均衡、最值得继续推进的轻量候选主线；它比 `encoder v2 balanced` 参数更少，并在三次 seed 下都保持了稳定、且有竞争力的 `Acc / F1 / MAE / Acc@10°`。
  - 在轻量 cue 独立流上，`coherence` 不是冗余项；去掉 `coherence` 的 `ild_phase` 版本没有超过 `encoder v2 balanced`。
  - 在 lite cue 独立流上，`temporal conv` 不是冗余项；改成 `MLP-only` 会明显削弱分类表现。
  - `absdiff` 更偏向帮助分类锐度，而不是帮助 `MAE`；去掉后 `Acc / F1 / Acc@5° / Acc@10°` 会掉，但 `MAE` 和尾部错误会更保守。
  - 纯回归进一步降低 `MAE` 和结构性大错，但分类指标明显下降。
  - 分类 + 回归联合没有超过当前原生分类主线。
- 在外部 backbone 上：
  - `SDEL-DOA-Cls + fbaux` 是当前最强分类结果。
  - `SDEL-DOA-Reg` 的 `MAE` 最低，仍值得继续做 `+ fbaux` 或联合任务验证。

## 当前代码结构

主模型链路：

```text
立体声 WAV
  -> STFT 特征提取
  -> (log_mag_L, log_mag_R, IPD, ILD, sin/cos(IPD), coherence)
  -> 共享编码器 / 或 native_lite_v7 内容流
  -> 双耳交互与时序建模
  -> DOA 分类 / 回归头
  -> 可选 front/back 辅助头
```

关键实现文件：

- 特征：
  - `dataset/feature_extractor.py`
  - `dataset/static_dataset.py`
- 原生主线：
  - `models/binaural_doa_net.py`
  - `models/encoder.py`
  - `models/difference_prior.py`
  - `models/cross_attention.py`
  - `models/gating.py`
  - `models/temporal_head.py`
- 轻量支线：
  - `models/native_lite_v7.py`
- 外部 baseline：
  - `models/sdel_crnn_baseline.py`
- 训练与评估：
  - `train.py`
  - `evaluate.py`
  - `engine/trainer.py`
  - `engine/evaluator.py`
  - `losses.py`
  - `metrics.py`

## EncoderV2 结构

当前轻量 `v7` 推荐配置使用的是：

- `models/encoder.py` 中的 `BinauralEncoderV2Balanced`
- 对应配置：
  - `configs/train_librispeech_multisubject_robust50h_v7_native_lite_encoderv2_balanced_nocsl_fbaux_cohfix.yaml`

它的设计目标不是单纯“多堆几层卷积”，而是让前端在沿频率轴压缩之前，先完成一轮更充分的局部时频建模，从而提升抗噪和抗混响能力，同时保持轻量。

### 输入输出

- 输入：
  - 单耳内容谱图 `x ∈ [B, 1, T, F]`
  - 左右耳共用同一个 encoder 实例（共享权重）
- 输出：
  - 单耳时序特征 `h ∈ [B, T, D]`
  - 默认 `D = 96`

在 `native_lite_v7` 中：

- 左耳 `log_mag_L` 经过 encoder 得到 `F_L`
- 右耳 `log_mag_R` 经过同一个 encoder 得到 `F_R`
- 再与 `ILD / sin(IPD) / cos(IPD) / coherence` 形成的 cue stream 融合

### 结构概览

`EncoderV2Balanced` 默认通道配置为：

- `encoder_channels = [24, 40, 64]`
- `encoder_out_dim = 96`

整体结构是：

```text
[Balanced Stage 1] 1   -> 24
[Balanced Stage 2] 24  -> 40
[Balanced Stage 3] 40  -> 64
-> AdaptiveAvgPool2d((None, 1))
-> Linear(64 -> 96)
```

每个 `Balanced Stage` 都由两步组成：

1. `pre_conv`
   - `Conv2d(3x3, stride=(1,1), padding=1)`
   - `BatchNorm2d`
   - `ReLU`

2. `downsample`
   - `Depthwise Conv2d(3x3, stride=(1,2), padding=1, groups=C)`
   - `Pointwise Conv2d(1x1)`
   - `BatchNorm2d`
   - `ReLU`
   - `Dropout2d`

也就是说，每个 stage 都遵循：

```text
先提局部特征 -> 再沿频率轴下采样
```

而不是旧版 encoder 那种：

```text
单层 stride=(1,2) 卷积直接边提特征边下采样
```

### 三个 stage 的具体写法

默认 `v7 encoder v2 balanced` 的 3 个 stage 可以写成：

```text
Stage 1
  Conv3x3, 1  -> 24, stride=(1,1)
  DWConv3x3, 24 -> 24, stride=(1,2)
  PWConv1x1, 24 -> 24

Stage 2
  Conv3x3, 24 -> 40, stride=(1,1)
  DWConv3x3, 40 -> 40, stride=(1,2)
  PWConv1x1, 40 -> 40

Stage 3
  Conv3x3, 40 -> 64, stride=(1,1)
  DWConv3x3, 64 -> 64, stride=(1,2)
  PWConv1x1, 64 -> 64

Tail
  AdaptiveAvgPool2d((None, 1))
  Linear(64 -> 96)
```

性质上：

- 时间维基本不压缩
- 频率维每个 stage 压一半
- depthwise separable 部分控制了参数量

### 和旧版 encoder 的区别

旧版 `BinauralEncoder` 是：

```text
[Conv3x3 stride=(1,2) + BN + ReLU + Dropout] x 3
-> freq pool
-> linear
```

新版 `BinauralEncoderV2Balanced` 是：

```text
[(Conv3x3 stride=1) + (DWConv3x3 stride=(1,2) + PWConv1x1)] x 3
-> freq pool
-> linear
```

区别主要有三点：

1. 旧版是一层卷积直接下采样；新版先提特征、再下采样。
2. 新版的下采样层用 depthwise separable conv，参数更省。
3. 新版通道数更克制：`[24, 40, 64]`，不是 `v7` 旧版的 `[24, 48, 96]`。

## LiteCueEncoder-A 结构

当前轻量 cue 独立流主线使用的是：

- `models/native_lite_v7.py` 中的 `NativeLiteLiteCueConcatDOANet`
- 对应配置：
  - `configs/train_librispeech_multisubject_robust50h_v7_litecueenc_concat_all_nocsl_fbaux_cohfix.yaml`

它的目标不是复制一条完整的 2D cue-CNN，而是让 cue 流保留独立身份，同时把计算量控制在比较克制的范围内。

### 核心思路

```text
内容流:
  log_mag_L / log_mag_R
  -> shared EncoderV2Balanced
  -> mean / diff / absdiff
  -> content_fusion(288 -> 96)

cue 流:
  [ILD, sin(IPD), cos(IPD), coherence]
  -> band pooling (F -> 16)
  -> temporal Conv1d
  -> cue_feat(32)

融合:
  concat([content_feat(96), cue_feat(32)])
  -> fused_feat(128)
  -> BiGRU + attention pooling
  -> classifier + front/back auxiliary head
```

### 为什么要单独做 LiteCueEncoder

上一版完整 `cue encoder + concat` 在 `T x F` 大图上跑了额外 2D CNN，结果是：

- 训练更慢
- 参数更少但计算并不划算
- 最终效果明显不如 `encoder v2 balanced`

`LiteCueEncoder-A` 改成：

- 先沿频率维做压缩
- 再只在时间维做轻量 1D 卷积

这样做的直觉是：

- cue 本来就是高度结构化的空间线索
- 不必复制一条重型内容 encoder
- 先压缩频率维，再做局部时序提纯，更符合 cue 的信息性质

### 参数量与计算量

`encoder v2 balanced`：

- 参数量：`297,667`
- 参数存储（FP32）：`1.136 MB`
- checkpoint 大小：约 `3.5 MB`
- MACs：`1.315 G`
- 估算 FLOPs：`2.631 G`

`lite cue encoder all`：

- 参数量：`251,795`
- 参数存储（FP32）：`0.961 MB`
- checkpoint 大小：约 `3.0 MB`
- MACs：`1.306 G`
- 估算 FLOPs：`2.612 G`

`lite cue encoder all + cf80 + cue24 + gru80`：

- 参数量：`196,987`
- 参数存储（FP32）：`0.751 MB`
- GRU 输入维度：`104`
- 说明：
  - `content_fusion_dim: 80`
  - `cue_encoder_out_dim: 24`
  - `gru_hidden_size: 80`

### 和 `encoder v2 balanced` 的关系

`encoder v2 balanced` 更像：

- `MAE` 更优
- `front/back` 和 `large error` 控制更稳

`lite cue encoder all` 更像：

- `Accuracy / F1 / Acc@5° / Acc@10°` 更强
- 参数量更小
- 但尾部大错更重，导致 `MAE` 没赢

当前仓库里，这三条线的定位可以理解为：

- `encoder v2 balanced`：轻量 `MAE` 主线 / 稳定参照
- `lite cue encoder all`：轻量分类 / 近邻命中主线
- `lite cue encoder all + cf80 + cue24 + gru80`：当前更均衡、最值得继续推进的轻量候选主线 / 当前推荐轻量主线

### 参数量与收益

在当前 `native_lite_v7` 主线里：

- 原始 `v7` 总参数量：`313,083`
- `encoder v2 balanced` 总参数量：`297,667`

也就是说，这次改动并不是“更重换更好”，而是：

- 参数量略降
- `F1 / MAE / Acc@5° / Acc@10°` 同时提升

在 unseen-subject test 上，相比原始 `v7 native_lite`：

- `F1`: `0.3570 -> 0.3856`
- `MAE`: `14.45° -> 12.91°`
- `Acc@5°`: `0.7298 -> 0.7526`
- `Acc@10°`: `0.8441 -> 0.8550`

### 为什么它更适合轻量鲁棒主线

这版 encoder 更适合当前目标“轻量化 + 抗噪 + 抗混响”的原因是：

- 它保留了 `native_lite_v7` 的轻量时序头和 cue stream 设计
- 只在内容编码前端做增强，不破坏整体主线结构
- 先做局部建模再压频率轴，更符合 noisy / reverberant 条件下的表征需求
- 用 depthwise separable conv 控制新增复杂度

如果后续继续做轻量化实验，当前最自然的出发点已经从 `encoder v2 balanced` 转到：

- `v7_litecueenc_concat_all_cf80_cue24_gru80`

因为这条线已经证明：

- 独立 cue 流是有效的
- `coherence` 和 `temporal conv` 值得保留
- 从特征提取和融合层减小 GRU 负担是有效方向

## 当前保留的配置

主线与强 baseline：

- `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_nocsl_enhanced_fbaux_only_cohfix.yaml`
- `configs/train_librispeech_multisubject_robust50h_sdel_doa_cls_fbaux.yaml`
- `configs/train_librispeech_multisubject_robust50h_sdel_doa_reg_baseline.yaml`
- `configs/train_librispeech_multisubject_robust50h_sdel_doa_cls_baseline.yaml`
- `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_reg_enhanced_fbaux_only_cohfix.yaml`
- `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_pure_reg_enhanced_fbaux_only_cohfix.yaml`

历史对照与消融：

- `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl.yaml`
- `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced.yaml`
- `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux.yaml`
- `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux_gru96.yaml`
- `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux_lite.yaml`
- `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux_only.yaml`
- `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux_only_cohfix.yaml`
- `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbfocus_only.yaml`
- `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_nols_enhanced_fbaux_only_cohfix.yaml`
- `configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_nocsl_nols_enhanced_fbaux_only_cohfix.yaml`

轻量 native 支线：

- `configs/train_librispeech_multisubject_robust50h_v6_native_simple_nocsl_fbaux_cohfix.yaml`
- `configs/train_librispeech_multisubject_robust50h_v6_native_simple_xattn_nocsl_fbaux_cohfix.yaml`
- `configs/train_librispeech_multisubject_robust50h_v6_native_simple_gru96_attnpool_nocsl_fbaux_cohfix.yaml`
- `configs/train_librispeech_multisubject_robust50h_v6_native_simple_gru96_nocsl_fbaux_cohfix.yaml`
- `configs/train_librispeech_multisubject_robust50h_v6_native_simple_gru96_xattn_nocsl_fbaux_cohfix.yaml`
- `configs/train_librispeech_multisubject_robust50h_v7_native_lite_nocsl_fbaux_cohfix.yaml`
- `configs/train_librispeech_multisubject_robust50h_v7_native_lite_encoderv2_balanced_nocsl_fbaux_cohfix.yaml`
- `configs/train_librispeech_multisubject_robust50h_v7_native_lite_xear_nocsl_fbaux_cohfix.yaml`
- `configs/train_librispeech_multisubject_robust50h_v7_native_lite_complexri_contentonly_nocsl_fbaux_cohfix.yaml`
- `configs/train_librispeech_multisubject_robust50h_v7_native_lite_complexri_phasecue_nocsl_fbaux_cohfix.yaml`

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

训练当前原生主线：

```bash
python train.py \
  --config configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_nocsl_enhanced_fbaux_only_cohfix.yaml
```

训练当前轻量 `v7` 主线：

```bash
python train.py \
  --config configs/train_librispeech_multisubject_robust50h_v7_native_lite_encoderv2_balanced_nocsl_fbaux_cohfix.yaml
```

训练当前推荐的轻量候选主线：

```bash
python train.py \
  --config configs/train_librispeech_multisubject_robust50h_v7_litecueenc_concat_all_cf80_cue24_gru80_nocsl_fbaux_cohfix.yaml
```

评估当前原生主线：

```bash
python evaluate.py \
  --checkpoint outputs/checkpoints_multisubject_robust50h_v5_bias_gating_attnpool_nocsl_enhanced_fbaux_only_cohfix/best.pth \
  --config configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_nocsl_enhanced_fbaux_only_cohfix.yaml \
  --output.log_dir outputs/logs_multisubject_robust50h_v5_bias_gating_attnpool_nocsl_enhanced_fbaux_only_cohfix_test_best_workers4
```

训练 SDEL baseline：

```bash
python train.py \
  --config configs/train_librispeech_multisubject_robust50h_sdel_doa_cls_fbaux.yaml
```

训练原生联合回归：

```bash
python train.py \
  --config configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_reg_enhanced_fbaux_only_cohfix.yaml
```

训练原生纯回归：

```bash
python train.py \
  --config configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_pure_reg_enhanced_fbaux_only_cohfix.yaml
```

## 可用工具

当前仍保留的训练/实验工具：

- `tools/run_training_tmux.sh`
- `tools/run_training_background.sh`
- `tools/run_fbaux_weight_sweep.sh`
- `tools/run_fbaux_weight_sweep_sequential.sh`
- `tools/run_cohfix_softlabel_ablation_sequential.sh`
- `tools/run_binmov2023_static_compare.sh`
- `tools/monitor_training.sh`
- `tools/diagnostics/analyze_multisubject_checkpoint.py`

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
├── dataset/
├── engine/
├── models/
├── tools/
├── utils/
└── outputs/
```

## 备注

- 当前工作树已经清理掉大部分 `subject003` 历史文件和对应输出。
- README 只描述当前仍保留在仓库中的配置、脚本和结果目录。
- 若后续继续精简 `configs/` 或 `outputs/`，建议同步更新这里的“当前保留的配置”部分。
