# 双耳 DOA-Net

面向**双耳水平声源定位（DOA）**的研究代码库。当前项目已经收束到一条明确主线：

- **任务**：72 类水平角双耳 DOA 分类（`0:5:355`）
- **数据协议**：`KEMAR + SofaMyRoom` 静态双耳渲染
- **鲁棒性评测**：在原测试集基础上，新增 `diffusefg` 扩散噪声测试协议
- **目标**：在保证模型复杂度可控的前提下，提高噪声与混响条件下的定位稳健性

当前 README 聚焦**现在仍在使用、仍有论文价值**的内容：项目背景、数据集协议、主模型、实验设置、结果与结论。

---

## 1. 项目背景

双耳 DOA 估计的核心信息来自双耳空间线索，例如：

- `ILD`：左右耳能量差
- `IPD`：左右耳相位差
- `coherence`：双耳一致性 / 线索可靠性

但在真实条件下，这些线索并不总是稳定：

- 混响会扰乱相位关系
- 噪声会污染能量差
- 低能量时频单元上的 cue 往往不可靠
- 前后方向存在天然混淆

因此，当前项目的核心问题不是“是否存在空间线索”，而是：

> **如何在噪声、混响和扩散背景噪声下，从不总是稳定的双耳线索中提取可靠的方向信息。**

围绕这个问题，本项目当前重点研究两件事：

1. **更真实、更困难的双耳静态鲁棒测试协议**
2. **结构化、轻量化的双耳 DOA 模型设计**

---

## 2. 当前数据集主线

## 2.1 静态 KEMAR 主数据集

当前正式主线使用：

- **HRTF / dummy head**：`KEMAR`
- **房间渲染**：`SofaMyRoom`
- **目标语音**：干净语音经 BRIR 渲染为双耳信号
- **采样率**：`16 kHz`
- **片段长度**：`2.0 s`
- **类别数**：`72` 类水平角
- **方位角**：`0:5:355`

当前实际使用的目录：

- 训练集：`data/kemar_sofamyroom/train_20h_minimal_diffusefg/train`
- 验证集：`data/kemar_sofamyroom/val_4h_diffusefg_officialsplit/val`
- 测试集：`data/kemar_sofamyroom_diffusefg_static_v1_test_officialsplit/test`

---

## 2.2 新 diffusefg 测试协议

这是当前项目里最重要的数据协议升级。

与旧测试集相比，它**不改变目标语音渲染、角度设置、房间测试网格和评测指标**，只替换背景噪声生成方式：

- 旧方式：`post-mix scene noise`
- 新方式：`mono scene noise -> ANF -> 2-channel diffuse noise`

当前 diffusefg 协议的关键点：

- 背景噪声来自场景噪声底材
- 经 `ANF` 生成双通道扩散噪声场
- 再按目标 `SNR` 混入双耳目标语音

这个协议的意义是：

1. 背景噪声比简单后混更像空间化扩散噪声
2. 测试难度更高，更能拉开模型差异
3. 更适合评价“鲁棒双耳定位”而不是理想条件分类

---

## 2.3 当前使用的 SNR 协议

训练 / 验证：

- `SNR ~ Uniform[-10, 10] dB`

测试集包含：

- `clean`
- `10 dB`
- `5 dB`
- `0 dB`
- `-5 dB`
- `-10 dB`

论文主表统一统计 `10 / 5 / 0 / -5 / -10 dB`，不包含 `clean`，也不使用额外的 `-15 dB` 压力测试。完整评测文件仍保留 clean 结果用于审计。

---

## 3. 模型设计主线

当前项目中最重要的方法线是 `v7` 系列，尤其是：

- `v7_dualcue_fbfocus`
- `v7_dualcue_liteenc_v1`

这条模型的设计出发点不是“堆模块”，而是围绕双耳 DOA 的几个真实矛盾来设计：

1. **只看空间 cue 不够稳**
2. **直接把原始 STFT 全交给大模型学习代价太高**
3. **方向估计不仅要看 cue 的值，还要看 cue 是否可靠**

因此当前 `v7` 的核心思路是：

> **先提取共享内容表示，再显式建模双耳差异与空间 cue，最后在低维表示上做轻量时序聚合。**

---

## 3.0 代表模型的完整前向流程

为了让模型描述更落地，下面以当前最重要的轻量主线
`v7_dualcue_liteenc_v1` 为例，把从输入到输出的张量流完整写出来。

对应配置的代表参数是：

- `n_fft = 512`，因此频点数 `F = 257`
- 片段长度 `2.0 s`，`hop_length = 160`，因此时间帧数通常约为 `T ≈ 201`
- `content_encoder_type = lite_v1`
- `encoder_channels = [16, 24, 32]`
- `encoder_out_dim = 64`
- `content_relation_mode = mean_diff_absdiff`
- `content_fusion_dim = 80`
- `lite_cue_bands = 16`
- `cue_value_out_dim = 24`
- `cue_reliability_out_dim = 8`
- `gru_hidden_size = 80`
- `num_classes = 72`

下面的张量维度都按这个主配置来说明。

### Step 1. 输入特征组织

每条样本先提取以下双耳特征：

- `log_mag_L`：`[B, T, F]`
- `log_mag_R`：`[B, T, F]`
- `ILD`：`[B, T, F]`
- `IPD` 再拆成：
  - `sin(IPD)`：`[B, T, F]`
  - `cos(IPD)`：`[B, T, F]`
- `coherence`：`[B, T, F]`

其中：

- content 流使用 `log_mag_L / log_mag_R`
- cue value 流使用 `ILD / sin(IPD) / cos(IPD)`
- cue reliability 流使用 `coherence`

### Step 2. 左右耳内容输入成形

对 content 流，左右耳各自扩一维通道：

- 左耳：`log_mag_L -> [B, 1, T, F]`
- 右耳：`log_mag_R -> [B, 1, T, F]`

左右耳共用**同一个**内容编码器，权重共享，不分别训练两套卷积核。

### Step 3. Content Encoder：`LightContentEncoderV1`

这是 `liteenc_v1` 的关键轻量化部分，定义在 [`models/encoder.py`](models/encoder.py)。

它由 3 个轻量 stage 组成，每个 stage 的结构是：

1. `1x1 pointwise conv`
2. `3x3 depthwise conv`
3. `1x3 depthwise conv, stride=(1,2)`，只沿频率轴下采样
4. `1x1 pointwise conv`
5. `BatchNorm + ReLU + Dropout`

对应通道数变化：

- Stage 1: `1 -> 16`
- Stage 2: `16 -> 24`
- Stage 3: `24 -> 32`

因为每个 stage 只压频率轴，不压时间轴，所以：

- 时间维基本保持：`T -> T`
- 频率维大约变成：`257 -> 129 -> 65 -> 33`

于是左右耳分别得到中间张量：

- `H_L, H_R : [B, 32, T, 33]`

然后做：

1. `AdaptiveAvgPool2d((None, 1))`，把频率维压到 `1`
2. `squeeze + permute`
3. `Linear(32 -> 64)` 投影

最终得到：

- `F_L : [B, T, 64]`
- `F_R : [B, T, 64]`

也就是说，content encoder 的输出不是完整 `T x F` 网格，而是每个时间步一个 `64` 维内容向量。

### Step 4. 双耳内容关系分解

当前主线不是直接 `concat(F_L, F_R)`，而是先做显式关系拆分：

- `mean_feat = 0.5 * (F_L + F_R)` -> `[B, T, 64]`
- `diff_feat = F_L - F_R` -> `[B, T, 64]`
- `abs_diff_feat = |F_L - F_R|` -> `[B, T, 64]`

在 `mean_diff_absdiff` 模式下，把三者拼接：

- `content_relation = concat(mean, diff, absdiff)` -> `[B, T, 192]`

然后经过 `content_fusion`：

- `Linear(192 -> 80)`
- `LayerNorm`
- `ReLU`
- `Dropout`

得到压缩后的内容表示：

- `content_feat : [B, T, 80]`

这一步的含义是：

- 用 `mean` 保留共享内容
- 用 `diff` 保留带符号双耳差异
- 用 `absdiff` 保留差异强度
- 再把三种关系压到一个低维内容子空间中

### Step 5. Dual-Cue Encoder：value / reliability 双分支

这部分定义在 [`models/native_lite_v7.py`](models/native_lite_v7.py)。

#### 5.1 Value branch

先把三种方向 cue 组起来：

- `value_tensor = stack([ILD, sin(IPD), cos(IPD)], dim=1)`
- 形状：`[B, 3, T, F]`

然后进入 `LiteCueEncoder(in_channels=3)`：

1. 先把每个时间步的频率轴自适应压到 `16` 个 band
2. 张量从 `[B, 3, T, 257]` 变成 `[B, T, 3, 16]`
3. 再展平为 `[B, T, 48]`
4. 转成 `Conv1d` 需要的格式 `[B, 48, T]`
5. 两层时间卷积：
   - `Conv1d(48 -> 48, k=3)`
   - `Conv1d(48 -> 24, k=3)`

最后转回：

- `cue_value_feat : [B, T, 24]`

#### 5.2 Reliability branch

可靠性分支只吃 `coherence`：

- `reliability_tensor = coherence.unsqueeze(1)` -> `[B, 1, T, F]`

同样先压到 `16` 个 band，再展平：

- `[B, 1, T, 257] -> [B, T, 1, 16] -> [B, T, 16]`

再用更轻的时间卷积编码，输出：

- `cue_reliability_feat : [B, T, 8]`

#### 5.3 Dual-Cue 融合

当前默认是 `concat` 融合：

- `cue_feat = concat(cue_value_feat, cue_reliability_feat)`
- `cue_feat : [B, T, 32]`

因此，dual-cue 分支最后提供的是：

- `24` 维“cue value”
- `8` 维“cue reliability”
- 合起来 `32` 维 cue 表示

### Step 6. Content / Cue 融合

把两条流在特征维拼接：

- `fused = concat(content_feat, cue_feat)`
- `[B, T, 80] + [B, T, 32] -> [B, T, 112]`

然后经过：

- `LayerNorm(112)`
- `Dropout`

所以送入时序头的最终输入是：

- `fused_feat : [B, T, 112]`

### Step 7. Temporal Head

这部分定义在 [`models/temporal_head.py`](models/temporal_head.py)。

默认使用：

- `BiGRU`
- `hidden_size = 80`
- `num_layers = 1`

因此：

- 输入：`[B, T, 112]`
- 输出：`[B, T, 160]`

因为双向 GRU 的两个方向各输出 `80` 维，拼起来是 `160` 维。

### Step 8. Attention Pooling

时序头默认不是简单平均，而是注意力池化：

1. 对每个时间步的 `160` 维特征打一个分数
2. 经 `softmax` 变成时间权重
3. 做加权求和

于是：

- `temporal_out : [B, T, 160]`
- `attn_weights : [B, T, 1]`
- `pooled : [B, 160]`

这个 `pooled` 就是整段 2 秒语音的片段级表示。

### Step 9. 输出头

主任务输出头是：

- `Linear(160 -> 72)`

得到：

- `logits : [B, 72]`

如果启用前后辅助头，还会额外输出：

- `front_back_logits : [B, 2]`

因此当前主模型的最终任务形式是：

- 主任务：72 类水平角分类
- 辅助任务：前 / 后二分类（可选）

### Step 10. 整体张量流总结

把整个前向过程压缩成一行，可以写成：

```text
log_mag_L / log_mag_R : [B,T,F]
    -> shared LightContentEncoderV1
    -> F_L, F_R : [B,T,64]
    -> mean/diff/absdiff
    -> concat : [B,T,192]
    -> Linear + LN + ReLU
    -> content_feat : [B,T,80]

ILD / sin(IPD) / cos(IPD) : [B,T,F]
    -> value branch
    -> cue_value_feat : [B,T,24]

coherence : [B,T,F]
    -> reliability branch
    -> cue_reliability_feat : [B,T,8]

concat(content_feat, cue_feat)
    -> fused_feat : [B,T,112]
    -> BiGRU : [B,T,160]
    -> attention pooling : [B,160]
    -> classifier : [B,72]
```

这个版本也解释了为什么 `v7_dualcue_liteenc_v1` 的复杂度能压得很低：

- content 流没有在高分辨率上使用重型常规 2D CNN
- cue 流先做频带压缩，再做时间建模
- 时序头只处理 `112` 维低维融合序列，而不是完整 `T x F` 网格

---

## 3.1 Shared Content Stream

输入：

- `log_mag_L`
- `log_mag_R`

左右耳分别通过**同一个内容编码器**，得到：

- `F_L`
- `F_R`

这条 content 流的作用不是直接输出方向，而是：

1. 提取左右耳共享的主声学内容
2. 为不稳定的空间 cue 提供上下文参考
3. 帮助模型区分有效语音结构与噪声/混响扰动

也就是说，content 流回答的是：

> 当前这段双耳观测里，左右耳共同听到了什么。

---

## 3.2 Binaural Relation Decomposition

当前 `v7` 没有直接把左右耳内容特征生拼，而是构造：

- `mean(F_L, F_R)`
- `diff(F_L, F_R)`
- `absdiff(F_L, F_R)`

这一步的意义：

- `mean`：共享内容
- `diff`：带符号方向差异
- `absdiff`：差异强度

相比直接 `concat(F_L, F_R)`，这种做法更结构化，能显式区分：

1. 左右耳共同成分
2. 与方向有关的差异
3. 差异幅度信息

而且这一步计算代价很低，是当前模型最有代表性的结构点之一。

---

## 3.3 Spatial Cue Stream

当前主线显式使用的双耳 cue 包括：

- `ILD`
- `sin(IPD)`
- `cos(IPD)`
- `coherence`

这条 cue 流不是可有可无的附属分支，而是为了保留 DOA 任务最直接的空间信息来源。

但项目不再采用重型 2D cue-CNN，而是采用更轻的处理顺序：

1. **先频带压缩**
2. **再时间建模**

原因是：

- cue 在相邻频点上存在冗余
- 逐频点 cue 在噪声/混响下波动较大
- 先压成 band-level 表示后，时间模块看到的是更稳定的 cue 轨迹

这也是当前模型保持低复杂度的重要来源之一。

---

## 3.4 Dual-Cue 设计：Value / Reliability 分离

这是当前 `dualcue` 主线最重要的模型思想。

它把 cue 分成两类：

### Value branch

输入：

- `ILD`
- `sin(IPD)`
- `cos(IPD)`

作用：

- 编码方向线索本身

### Reliability branch

输入：

- `coherence`

作用：

- 编码这些方向线索当前是否可信

这一步的动机很直接：

> 在复杂声学条件下，模型不应该只问“cue 是多少”，还应该问“这个 cue 值不值得信”。

这使得模型在 diffuse noise 和混响条件下更有鲁棒性解释。

---

## 3.5 Temporal Aggregation

content 与 cue 融合后，当前主线使用轻量时序聚合头（以 `GRU` 为主）整合多帧证据。

这里的关键不是“时序头很强”，而是：

- 前端已经把共享内容、双耳差异和 cue 可靠性整理成较干净的低维表示
- 因此后端不需要像重型时频序列模型那样在完整 `T x F` 网格上反复做昂贵建模

这也是当前 `v7` 和 `FN-SSL` 的重要区别之一。

---

## 3.6 Front/Back Auxiliary

当前多条主线都保留了轻量 `front/back auxiliary head`。

它的作用不是引入新任务，而是帮助缓解双耳定位中的典型歧义：

- front/back confusion

但从当前实验看，这个辅助头**并不是所有模型都同样受益**：

- 在 `v7` 主线中，它仍然可以作为有意义的训练约束
- 但在 `SDEL`、`FN-SSL` 这类外部 baseline 中，开关辅助头带来的收益并不稳定

因此当前更严谨的说法是：

> `front/back auxiliary` 是一个**可选正则项**，可以作为消融因素单独检验，
> 但不能把外部 baseline 的强结果简单归因于它。

---

## 4. 当前重点模型

### 4.1 `v7_dualcue_liteenc_v1`

这是当前论文主模型：共享轻量内容编码器、`mean/diff/absdiff` 双耳关系表示、cue value/reliability 双分支，以及紧凑 BiGRU 时序头。模型参数量为 `152,803`。

### 4.2 `rawconcat_controlled`

该受控变体只把内容关系从 `mean/diff/absdiff` 改为 `concat(F_L, F_R)`，其余宽度和训练协议保持一致。它用于检验显式关系分解，而不是作为独立主模型。参数量为 `147,683`。

---

## 5. 当前外部 baseline

主表保留 `SDEL`、`FN-SSL`、`DP-RTF` 和 `BiL`。所有正式比较都使用相同的官方验证集、官方测试集、72 类角度定义和 grouped 评测脚本。FN-SSL 与 BiL 当前各有一个完整 seed；SDEL、DP-RTF 各有三个 seed。

---

## 6. 官方测试结果

以下结果最后核验于 `2026-07-24`。论文口径排除 clean，统计 `10/5/0/-5/-10 dB` 共 `6480` 个样本。模型均使用验证集 MAE 最优的 `best.pth`，没有按测试集挑 checkpoint。

### 6.1 固定 seed43

| 模型 | Params | Acc (%) | MAE (deg) | Acc@5 (%) | Acc@10 (%) |
|---|---:|---:|---:|---:|---:|
| `FN-SSL` | 658,890 | **96.590** | **2.231** | **98.164** | **98.287** |
| `v7_dualcue_liteenc_v1` | **152,803** | 96.559 | 2.590 | 97.855 | 97.886 |
| `SDEL` | 925,834 | 96.543 | 2.796 | 97.762 | 97.901 |
| `rawconcat_controlled` | 147,683 | 95.139 | 3.566 | 96.867 | 96.960 |
| `DP-RTF` | 876,608 | 94.938 | 4.678 | 96.543 | 96.682 |
| `BiL` | 865,228 | 94.352 | 5.136 | 95.849 | 96.049 |

固定 seed43 下，主模型与 FN-SSL 的 Accuracy 相差 `0.031 pp`，与 SDEL 相差 `0.015 pp`。这些差异只有 1-2 个样本，尚未做配对显著性检验，不能解释为统计显著领先。

### 6.2 多 seed 稳定性

| 模型 | Seeds | Acc mean +/- std (%) | MAE mean +/- std (deg) |
|---|---:|---:|---:|
| `SDEL` | 3 | **96.713 +/- 0.157** | **2.655 +/- 0.126** |
| `v7_dualcue_liteenc_v1` | 3 | 96.085 +/- 0.420 | 2.875 +/- 0.257 |
| `rawconcat_controlled` | 3 | 95.983 +/- 0.750 | 2.852 +/- 0.619 |
| `DP-RTF` | 3 | 95.453 +/- 0.489 | 4.389 +/- 0.251 |

FN-SSL 和 BiL 只有一个完整 seed，因此不在该表中报告伪造的标准差。

---

## 7. 分 SNR 结果

下表固定 seed43，每格为 `Acc (%) / MAE (deg)`：

| 模型 | 10 dB | 5 dB | 0 dB | -5 dB | -10 dB |
|---|---:|---:|---:|---:|---:|
| `v7_dualcue_liteenc_v1` | 99.07/0.76 | **98.77/0.95** | **98.30/1.22** | 94.68/4.09 | 91.98/5.93 |
| `FN-SSL` | 98.77/1.08 | 98.30/1.17 | 97.76/1.64 | **95.68/1.96** | **92.44/5.31** |
| `SDEL` | **99.15/1.25** | 98.53/1.37 | 97.76/2.26 | 95.52/3.01 | 91.74/6.09 |
| `rawconcat_controlled` | 98.61/1.15 | 98.15/1.27 | 96.99/2.09 | 93.06/5.16 | 88.89/8.16 |
| `DP-RTF` | 98.38/2.21 | 97.61/3.37 | 95.83/4.51 | 93.29/6.02 | 89.58/7.28 |
| `BiL` | 98.15/2.57 | 97.53/2.54 | 95.91/3.67 | 93.13/6.08 | 87.04/10.82 |

主模型的优势集中在 `0 dB` 及以上；FN-SSL 在 `-5/-10 dB` 更强。论文因此将结论限定为“轻量模型在中高 SNR 接近或达到更大基线”，不声称所有噪声条件全面领先。

---

## 8. 受控消融

以下为 seed42 noisy-only 结果：

| 变体 | Acc (%) | MAE (deg) | Delta Acc | Delta MAE |
|---|---:|---:|---:|---:|
| 主模型 | 95.756 | 2.945 | - | - |
| `w/o reliability` | 95.694 | 3.120 | -0.062 pp | +0.175 |
| `w/o content` | 94.907 | 3.847 | -0.849 pp | +0.902 |
| `merged cue` | 96.003 | 3.069 | +0.247 pp | +0.124 |
| `rawconcat_controlled` | 96.235 | 2.463 | +0.478 pp | -0.482 |

当前证据支持 content 分支的重要性；reliability 分支的主要收益体现在 MAE。`merged cue` 呈现 Accuracy/MAE 权衡。Rawconcat 在 seed42、44 更好，但 seed43 明显退化；三 seed 平均 Accuracy 比主模型低 `0.103 pp`，且方差更大。因此 `mean/diff/absdiff` 只作为低成本关系归纳偏置，不作为已经被证明独立提升精度的核心创新点。

---

## 9. 参数量与复杂度

| 模型 | Params (M) | FLOPs (G) |
|---|---:|---:|
| `v7_dualcue_liteenc_v1` | **0.153** | **0.391** |
| `FN-SSL` | 0.659 | 65.687 |
| `SDEL` | 0.926 | 1.402 |
| `DP-RTF` | 0.877 | 8.215 |
| `BiL` | 0.865 | 3.117 |

主模型参数量约为 FN-SSL 的 `1/4.3`、SDEL 的 `1/6.1`。轻量化来自前端的频率压缩和低维关系/cue 表示，而不是在完整时频平面上堆叠序列模块。

---

## 10. 结论与限制

当前可复现结论是：

1. content 上下文是性能最关键的组成部分。
2. value/reliability 分支提供了可解释的 cue 组织方式，移除 reliability 会增加 MAE。
3. 主模型在 152.8K 参数下，固定 seed43 的 noisy-only Accuracy 与 FN-SSL、SDEL 接近。
4. 低 SNR 仍是主要缺口，FN-SSL 在 `-5/-10 dB` 表现更强。
5. Rawconcat 的结果否定了“显式 `mean/diff/absdiff` 必然更准”的强结论；当前只能观察到主模型的跨 seed 方差较小。

由于已经进行过多轮结构探索，测试集结果不得继续用于选择主结构。后续论文结论需要补充预先固定规则下的额外 seed 或配对 bootstrap，当前 README 不声称统计显著性。

---

## 11. 当前推荐训练与评测入口

训练官方划分的 `v7_dualcue_liteenc_v1` seed43：

```bash
python train.py \
  --config configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_officialsplit_seed43_g5.yaml
```

训练受控 Rawconcat seed43：

```bash
python train.py \
  --config configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_rawconcat_controlled_officialsplit_seed43_g5.yaml
```

评测 grouped 结果：

```bash
python tools/evaluate_kemar_grouped.py \
  --config configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_officialsplit_seed43_g5.yaml \
  --checkpoint outputs/checkpoints_kemar_v7_dualcue_liteenc_v1_diffusefg_officialsplit_seed43_g5/best.pth \
  --test_root data/kemar_sofamyroom_diffusefg_static_v1_test_officialsplit/test \
  --output_dir outputs/grouped_eval_runs_officialsplit_retrained/main_seed43 \
  --device cuda:0 \
  --num_workers 4
```

生成论文汇总表：

```bash
python tools/summarize_officialsplit_retrained.py
```

---

## 12. 当前代码结构

```text
DOA-net/
├── README.md
├── train.py
├── evaluate.py
├── losses.py
├── metrics.py
├── configs/
├── dataset/
├── engine/
├── models/
├── tools/
├── utils/
└── outputs/
```

关键文件：

- 数据与特征：
  - `dataset/static_dataset.py`
  - `dataset/feature_extractor.py`
- 主模型：
  - `models/native_lite_v7.py`
  - `models/encoder.py`
  - `models/temporal_head.py`
- baseline：
  - `models/sdel_crnn_baseline.py`
  - `models/fn_ssl_baseline.py`
- 训练与评测：
  - `train.py`
  - `tools/evaluate_kemar_grouped.py`

---

## 13. 当前一句话总结

当前项目最清晰的结论是：

> **在 KEMAR + SofaMyRoom 静态双耳渲染与 diffusefg 扩散噪声测试协议下，基于共享内容上下文、显式双耳关系分解以及 cue value/reliability 分离建模的轻量 `v7` 主线，已经在性能、误差控制和复杂度之间形成了有竞争力的平衡。**
