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

## 推荐主线地图

这一版 README 只保留当前真正有研究价值的静态主线，并把已经确认价值不高的变体单独归档。现在最值得记住的是下面四条：

### 1. 论文方法主线：`dual cue value/reliability`

- 配置：
  - `configs/train_librispeech_multisubject_robust50h_v7_dualcue_vr_cf80_gru80_nocsl_fbaux_cohfix.yaml`
- 任务角色：
  - **当前论文方法主线**
  - 用来回答：`ILD / IPD` 这类空间差异值和 `coherence` 这类线索可靠性，是否应该显式分开建模
- 结构关键词：
  - `shared content encoder`
  - `mean / diff / absdiff`
  - `dual cue branches`
  - `cue value = ILD + sin(IPD) + cos(IPD)`
  - `cue reliability = coherence`
  - `compact fusion + BiGRU`
- 结构概念：

```text
log_mag_L / log_mag_R
  -> shared EncoderV2Balanced
  -> mean / diff / absdiff
  -> content_fusion(288 -> 80)

[ILD, sin(IPD), cos(IPD)]
  -> cue value encoder
  -> cue_value_feat(24)

[coherence]
  -> cue reliability encoder
  -> cue_reliability_feat(8)

concat([cue_value_feat, cue_reliability_feat])
  -> cue_feat(32)

concat([content_feat(80), cue_feat(32)])
  -> fused_feat(112)
  -> BiGRU(80)
  -> classifier + fbaux
```

- 三次 seed 平均：
  - `accuracy = 0.6838 ± 0.0076`
  - `f1_score = 0.3982 ± 0.0094`
  - `acc_at_5deg = 0.7616 ± 0.0058`
  - `acc_at_10deg = 0.8681 ± 0.0153`
  - `mean_angular_error = 11.76° ± 1.10`

### 2. 轻量 compact 主线：`cf80_cue24_gru80`

- 配置：
  - `configs/train_librispeech_multisubject_robust50h_v7_litecueenc_concat_all_cf80_cue24_gru80_nocsl_fbaux_cohfix.yaml`
- 任务角色：
  - **当前 compact / 部署主线**
  - 用来回答：在不破坏整体表现的前提下，内容流、cue 流和 GRU 输入能压缩到什么程度
- 结构关键词：
  - `shared content encoder`
  - `mean / diff / absdiff`
  - `single lite cue encoder`
  - `content_fusion_dim = 80`
  - `cue_dim = 24`
  - `gru_hidden = 80`
- 结构概念：

```text
log_mag_L / log_mag_R
  -> shared EncoderV2Balanced
  -> mean / diff / absdiff
  -> content_fusion(288 -> 80)

[ILD, sin(IPD), cos(IPD), coherence]
  -> LiteCueEncoder
  -> cue_feat(24)

concat([content_feat(80), cue_feat(24)])
  -> fused_feat(104)
  -> BiGRU(80)
  -> classifier + fbaux
```

- 三次 seed 平均：
  - `accuracy = 0.6772 ± 0.0217`
  - `f1_score = 0.3901 ± 0.0140`
  - `acc_at_5deg = 0.7498 ± 0.0190`
  - `acc_at_10deg = 0.8625 ± 0.0092`
  - `mean_angular_error = 12.58° ± 0.43`

### 3. 稳定参照线：`encoder v2 balanced`

- 配置：
  - `configs/train_librispeech_multisubject_robust50h_v7_native_lite_encoderv2_balanced_nocsl_fbaux_cohfix.yaml`
- 任务角色：
  - **稳定参照线 / MAE 参照**
  - 用来说明：在不引入独立强 cue encoder 的情况下，把内容 encoder 做扎实就已经能明显改善鲁棒定位
- 结构关键词：
  - `shared content encoder`
  - `cue stream as simple auxiliary`
  - `additive fusion`
- 结构概念：

```text
log_mag_L / log_mag_R
  -> shared EncoderV2Balanced
  -> mean / diff / absdiff

[ILD, sin(IPD), cos(IPD), coherence]
  -> band-pool + MLP
  -> cue auxiliary feature

mean_proj + diff_proj + absdiff_proj + cue_proj
  -> fused_feat(160)
  -> BiGRU(96)
  -> classifier + fbaux
```

- test：
  - `accuracy = 0.6644`
  - `f1_score = 0.3856`
  - `acc_at_5deg = 0.7526`
  - `acc_at_10deg = 0.8550`
  - `mean_angular_error = 12.91°`

### 4. 分类强参照线：`lite cue all`

- 配置：
  - `configs/train_librispeech_multisubject_robust50h_v7_litecueenc_concat_all_nocsl_fbaux_cohfix.yaml`
- 任务角色：
  - **分类强参照线**
  - 用来说明：独立 cue 流一旦成立，`Acc / F1 / Acc@5 / Acc@10` 会迅速变强
- 结构关键词：
  - `shared content encoder`
  - `single lite cue encoder`
  - `content_feat(96) + cue_feat(32)`
- 结构概念：

```text
log_mag_L / log_mag_R
  -> shared EncoderV2Balanced
  -> mean / diff / absdiff
  -> content_fusion(288 -> 96)

[ILD, sin(IPD), cos(IPD), coherence]
  -> LiteCueEncoder
  -> cue_feat(32)

concat([content_feat(96), cue_feat(32)])
  -> fused_feat(128)
  -> BiGRU(96)
  -> classifier + fbaux
```

- test：
  - `accuracy = 0.7133`
  - `f1_score = 0.4264`
  - `acc_at_5deg = 0.7800`
  - `acc_at_10deg = 0.8656`
  - `mean_angular_error = 13.89°`

### 5. 关键 baseline：`content-only` 与 `early-fusion`

这两条不是最终主线，但在论文里很关键，因为它们把“显式 cue 到底值不值”讲清楚了。

#### `content-only baseline`

- 配置：
  - `configs/train_librispeech_multisubject_robust50h_v7_contentonly_cf80_gru80_nocsl_fbaux_cohfix.yaml`
- 任务角色：
  - **最关键的消融 baseline**
  - 用来回答：只靠左右耳内容关系，不使用显式 `ILD/IPD/coherence`，到底会掉多少
- 结构概念：

```text
log_mag_L / log_mag_R
  -> shared EncoderV2Balanced
  -> mean / diff / absdiff
  -> content_fusion(288 -> 80)
  -> BiGRU(80)
  -> classifier + fbaux
```

- test：
  - `accuracy = 0.3062`
  - `f1_score = 0.1425`
  - `acc_at_5deg = 0.3642`
  - `acc_at_10deg = 0.5163`
  - `mean_angular_error = 25.02°`

#### `early-fusion single-encoder baseline`

- 配置：
  - `configs/train_librispeech_multisubject_robust50h_v7_earlyfusion_all_cf80_gru80_nocsl_fbaux_cohfix.yaml`
- 任务角色：
  - **强单流 baseline**
  - 用来回答：把内容和 cue 从输入端直接混合，能否替代显式分流建模
- 结构概念：

```text
[mean log-magnitude, ILD, sin(IPD), cos(IPD), coherence]
  -> shared EncoderV2Balanced
  -> Linear(96 -> 80)
  -> BiGRU(80)
  -> classifier + fbaux
```

- test：
  - `accuracy = 0.5944`
  - `f1_score = 0.3278`
  - `acc_at_5deg = 0.6664`
  - `acc_at_10deg = 0.8139`
  - `mean_angular_error = 13.86°`

## 各 encoder 结构

这一节只讲当前主线里真正还在使用的 encoder，不再展开已经明确放弃的重型历史结构。

### 1. `EncoderV2Balanced`（内容流 encoder）

对应实现：
- [models/encoder.py](/disk2/bywang/DOA-net/models/encoder.py:1)

当前使用它的模型：
- `encoder v2 balanced`
- `lite cue all`
- `cf80_cue24_gru80`
- `dual cue value/reliability`
- `early-fusion baseline`（作为共享单流 encoder）

#### 输入 / 输出

- 输入：
  - 单耳内容谱图 `x ∈ [B, 1, T, F]`
  - 早融合 baseline 中输入改成 `x ∈ [B, 5, T, F]`
- 输出：
  - `h ∈ [B, T, D]`
  - 默认 `D = 96`

#### 默认通道配置

```text
encoder_channels = [24, 40, 64]
encoder_out_dim = 96
```

#### 结构概念

```text
Stage 1: 1   -> 24
Stage 2: 24  -> 40
Stage 3: 40  -> 64
Tail: AdaptiveAvgPool2d((None, 1)) + Linear(64 -> 96)
```

每个 stage 都是：

```text
pre_conv:
  Conv2d(3x3, stride=1, padding=1)
  + BN + ReLU

downsample:
  Depthwise Conv2d(3x3, stride=(1,2), padding=1)
  + Pointwise Conv2d(1x1)
  + BN + ReLU + Dropout2d
```

#### 关键特点

- 时间维基本保留
- 频率维逐 stage 压缩
- 先局部建模，再沿频率轴下采样
- 相比旧版单层 stride 卷积，更适合 noisy / reverberant 条件

#### 作用

- 这是当前所有主线共享的**内容流 backbone**
- 负责把左右耳单耳谱图先编码成稳定的内容表示
- 后面的 `mean / diff / absdiff` 都建立在它的输出之上

---

### 2. 简单 cue MLP encoder（`encoder v2 balanced` 里的 cue 辅助流）

对应实现：
- [models/native_lite_v7.py](/disk2/bywang/DOA-net/models/native_lite_v7.py:1)

只用于：
- `encoder v2 balanced`

#### 输入

```text
[ILD, sin(IPD), cos(IPD), coherence]
```

#### 结构概念

```text
cue tensor
  -> band pooling
  -> flatten
  -> MLP
  -> cue_feat
  -> Linear projection to fusion dim
```

#### 角色

- 它不是独立强 cue encoder
- 更像给内容流提供一个轻量空间辅助项

#### 作用

- 用最小代价把 `ILD / IPD / coherence` 作为辅助 cue 注入内容主线
- 主要服务于 `encoder v2 balanced` 这条稳健参照线

---

### 3. `LiteCueEncoder`（单分支轻量 cue encoder）

对应实现：
- [models/native_lite_v7.py](/disk2/bywang/DOA-net/models/native_lite_v7.py:1)

当前使用它的模型：
- `lite cue all`
- `cf80_cue24_gru80`

#### 输入

默认 `all` 模式下：

```text
[ILD, sin(IPD), cos(IPD), coherence]
```

#### 默认配置

```text
lite_cue_bands = 16
lite_cue_hidden_dim = 48
cue_encoder_out_dim = 32   # lite cue all
cue_encoder_out_dim = 24   # cf80_cue24_gru80
lite_cue_kernel_size = 3
lite_cue_encoder_type = "temporal_conv"
```

#### 结构概念

```text
cue tensor [B, C, T, F]
  -> adaptive band-pool (F -> 16)
  -> reshape to [B, T, C*bands]
  -> Conv1d(k=3) over time
  -> BN + ReLU + Dropout
  -> Conv1d(k=3) over time
  -> BN + ReLU
  -> cue_feat
```

#### 关键特点

- 不复制一整条重型 2D cue-CNN
- 先把频率维压成粗频带
- 再只在时间维做轻量 cue 提纯
- 是当前独立 cue 流成功的基础版本

#### 作用

- 负责把显式 binaural cues 编码成紧凑 cue 表征
- 是 `lite cue all` 与 `cf80_cue24_gru80` 两条主线的核心 cue encoder

---

### 4. `DualBranchCueEncoder`（双分支 cue encoder）

对应实现：
- [models/native_lite_v7.py](/disk2/bywang/DOA-net/models/native_lite_v7.py:144)

当前使用它的模型：
- `dual cue value/reliability`
- `dual cue + reliability gating`（实验中）

#### 分支定义

##### value branch

输入：

```text
[ILD, sin(IPD), cos(IPD)]
```

输出：

```text
cue_value_feat ∈ [B, T, 24]
```

##### reliability branch

输入：

```text
[coherence]
```

输出：

```text
cue_reliability_feat ∈ [B, T, 8]
```

两支内部都复用 `LiteCueEncoder` 的轻量思路：
- band-pool
- temporal Conv1d

#### `concat` 版本（当前 README 主方法结果）

```text
cue_value_feat(24)
cue_reliability_feat(8)
-> concat
-> cue_feat(32)
```

再和内容流融合：

```text
content_feat(80) + cue_feat(32)
-> fused_feat(112)
-> BiGRU(80)
```

#### `gate` 版本（当前新增实验）

```text
cue_reliability_feat(8)
  -> Linear(8 -> 24)
  -> Sigmoid
  -> gate(24)

cue_feat = cue_value_feat * gate
```

这时：

```text
content_feat(80) + cue_feat(24)
-> fused_feat(104)
-> BiGRU(80)
```

#### 关键特点

- `coherence` 不再只是并列 cue
- 而是被解释成 `reliability`
- 更贴近“空间差异值 + 线索可靠性”这个方法叙事

#### 作用

- 把“空间差异值”和“线索可靠性”拆开建模
- 是当前 `dual cue value/reliability` 方法主线最核心的结构点

---

### 5. `Early-Fusion Shared Encoder`（单流早融合 baseline）

对应实现：
- [models/native_lite_v7.py](/disk2/bywang/DOA-net/models/native_lite_v7.py:1)

当前使用它的模型：
- `early-fusion single-encoder baseline`

#### 输入

```text
[mean log-magnitude, ILD, sin(IPD), cos(IPD), coherence]
```

也就是一个 `5` 通道输入：

```text
x ∈ [B, 5, T, F]
```

#### 结构概念

```text
5-channel early fusion input
  -> shared EncoderV2Balanced
  -> Linear(96 -> 80)
  -> BiGRU(80)
  -> classifier + fbaux
```

#### 关键特点

- 内容和 cue 从输入端就混合
- 是一个强 baseline
- 但在 unseen-subject test 上最终不如分流主线稳

#### 作用

- 作为“单流早融合是否已经足够强”的关键对照
- 用来证明后续的 content/cue 解耦不是建立在弱 baseline 上

## 该继续挖掘 / 该保留 / 该停止

### 值得继续挖掘

- `dual cue value/reliability`
  - 当前论文方法主线
  - 已经有 3-seed 结果和完整 robustness 分析基础
- `cf80_cue24_gru80`
  - 当前 compact 主线
  - 参数最省、稳定性最好，适合作为 compact variant

### 保留但不建议深挖

- `encoder v2 balanced`
  - 稳定参照线
- `lite cue all`
  - 强分类参照线

### 作为 baseline / 对照保留

- `v7 early-fusion single-encoder baseline`
  - 验证早融合是否足够强
  - 结果证明：它是强 baseline，但在 unseen-subject test 上不如分流主线稳
- `BiL-style GCC-PHAT CRN`
  - 当前最有说服力的外部轻量深度 baseline
- `FAViT-style ILD/IPD`
  - 作为外部 Transformer baseline 保留，用于说明“显式 cue + Transformer”并不天然优于更贴近双耳定位结构的设计
- `v5 enhanced + fbaux_only + cohfix`
  - 原生重模型参照
- `SDEL-DOA-Cls + fbaux`
  - 当前外部最强分类 baseline

### 可以停止深挖（单独归档）

下面这些实验已经有结论，但不再作为主线推进对象：

- 完整 cue-CNN concat
  - 更重、更慢，结果没有超过主线
- multi-scale temporal cue encoder
  - 改动不大，但无一致收益
- band-weighting / band-attention cue encoder
  - `MAE / Acc@10 / large error` 明显变差
- `dual cue + TF-mask`
  - 时频 mask 没有带来收益，反而损伤泛化
- `dual cue + reliability gate`
  - 作为“更稳健的变体”有分析价值，但不替代 concat 主线
- `dual cue + LSTM`
  - 比 `GRU` 更重、更差
- `dual cue + Mamba`
  - 比 `GRU` 更重、更差，也没有比 `LSTM` 更好
- 更重的 `v5` gate / cross-attention 变体
  - 复杂度高，收益和复杂度不匹配

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

主结果表优先按 `Accuracy / F1 / Acc@5° / Acc@10° / MAE` 排列，方便和你当前论文主叙事保持一致。

| 模型 | 配置 | Accuracy | F1-score | Acc@5° | Acc@10° | MAE | FB err | Opp err | Large err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 论文方法主线 `dual cue value/reliability (3-seed avg)` | `seed42 / seed43 / seed44` | 0.6838 | 0.3982 | 0.7616 | 0.8681 | 11.76° | 0.0886 | 0.0083 | 0.0646 |
| 轻量 compact 主线 `cf80_cue24_gru80 (3-seed avg)` | `seed42 / seed43 / seed44` | 0.6772 | 0.3901 | 0.7498 | 0.8625 | 12.58° | 0.0896 | 0.0090 | 0.0698 |
| 稳定参照线 `encoder v2 balanced` | `train_librispeech_multisubject_robust50h_v7_native_lite_encoderv2_balanced_nocsl_fbaux_cohfix.yaml` | 0.6644 | 0.3856 | 0.7526 | 0.8550 | 12.91° | 0.0916 | 0.0064 | 0.0731 |
| 分类强参照线 `lite cue all` | `train_librispeech_multisubject_robust50h_v7_litecueenc_concat_all_nocsl_fbaux_cohfix.yaml` | 0.7133 | 0.4264 | 0.7800 | 0.8656 | 13.89° | 0.1078 | 0.0088 | 0.0821 |
| 外部轻量 baseline `BiL-style GCC-PHAT CRN` | `train_librispeech_multisubject_robust50h_bilstyle_gccphat_crn72_nocsl.yaml` | 0.6962 | 0.4032 | 0.7621 | 0.8629 | 14.98° | 0.1029 | 0.0082 | 0.0892 |
| 外部 Transformer baseline `FAViT-style ILD/IPD` | `train_librispeech_multisubject_robust50h_favitstyle_ildipd_nocsl_fbaux_cohfix.yaml` | 0.6162 | 0.3303 | 0.7002 | 0.8300 | 16.58° | 0.1164 | 0.0109 | - |
| 内容-only baseline | `train_librispeech_multisubject_robust50h_v7_contentonly_cf80_gru80_nocsl_fbaux_cohfix.yaml` | 0.3062 | 0.1425 | 0.3642 | 0.5163 | 25.02° | 0.2241 | 0.0199 | 0.1383 |
| Early-fusion baseline | `train_librispeech_multisubject_robust50h_v7_earlyfusion_all_cf80_gru80_nocsl_fbaux_cohfix.yaml` | 0.5944 | 0.3278 | 0.6664 | 0.8139 | 13.86° | 0.0877 | 0.0073 | - |
| 原生重模型参照 `v5 + enhanced + fbaux_only + cohfix + no-csl` | `train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_nocsl_enhanced_fbaux_only_cohfix.yaml` | 0.6661 | 0.3828 | 0.7429 | 0.8720 | 11.92° | 0.0967 | 0.0089 | 0.0667 |
| 纯回归参考 `DOA-Net pure-reg + fbaux` | `train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_pure_reg_enhanced_fbaux_only_cohfix.yaml` | 0.4457 | 0.3196 | 0.7438 | 0.8834 | 11.01° | 0.0816 | 0.0063 | 0.0578 |

### `unseen-noise` 附加鲁棒性结果

为了把“噪声鲁棒性”说得更硬，额外生成了一个只改噪声场景、不改 subject / SNR / RT60 / 房间参数协议的 test-only set：

- 数据集：
  - `data/librispeech_cipic_multisubject_robust50h_v1/test_subjects_unseen_noiseheldout`
- 特点：
  - `subject unseen`
  - `noise scenes unseen`
  - 其余协议与原始 `test_subjects_unseen` 保持一致

当前最关键的主线和外部 baseline 在 `unseen-noise` 上的结果如下：

| 模型 | 评估配置 | Accuracy | F1-score | Acc@5° | Acc@10° | MAE | FB err | Opp err | Large err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 论文方法主线 `dual cue value/reliability` | `configs/eval_librispeech_multisubject_robust50h_v7_dualcue_vr_cf80_gru80_noiseheldout.yaml` | 0.6891 | 0.4101 | 0.7648 | 0.8611 | 12.10° | 0.0891 | 0.0124 | 0.0623 |
| 轻量 compact 主线 `cf80_cue24_gru80` | `configs/eval_librispeech_multisubject_robust50h_v7_cf80_cue24_gru80_noiseheldout.yaml` | 0.6463 | 0.3714 | 0.7231 | 0.8419 | 13.32° | 0.0974 | 0.0091 | 0.0791 |
| 外部轻量 baseline `BiL-style GCC-PHAT CRN` | `configs/eval_librispeech_multisubject_robust50h_bilstyle_gccphat_crn72_noiseheldout.yaml` | 0.6777 | 0.3929 | 0.7390 | 0.8487 | 16.17° | 0.1112 | 0.0112 | 0.0968 |
| 外部 Transformer baseline `FAViT-style ILD/IPD` | `configs/eval_librispeech_multisubject_robust50h_favitstyle_ildipd_noiseheldout.yaml` | 0.6123 | 0.3274 | 0.6978 | 0.8351 | 16.32° | 0.1118 | 0.0090 | - |
| Early-fusion baseline | `configs/eval_librispeech_multisubject_robust50h_v7_earlyfusion_all_cf80_gru80_noiseheldout.yaml` | 0.5968 | 0.3321 | 0.6697 | 0.8117 | 13.72° | 0.0882 | 0.0061 | 0.0756 |

#### `unseen-noise` 结论

- `dual cue value/reliability`
  - 在 `noiseheldout` 条件下几乎没有掉点
  - 说明显式 `value / reliability` 分解对未见噪声场景泛化非常稳
- `cf80_cue24_gru80`
  - 相比原始 seen-noise test 有明显下降
  - 说明 compact 化会牺牲一部分未见噪声鲁棒性
- `BiL-style GCC-PHAT CRN`
  - 在 `Accuracy / F1 / Acc@5°` 上仍然是一个很强的外部 baseline
  - 但 `MAE / front-back / large error` 明显落后于 `dual cue`，说明统一 GCC-PHAT CRN 更容易被少量前后混淆样本拉高尾部误差
- `FAViT-style ILD/IPD`
  - 说明频率优先 patch Transformer 并没有自然带来更强的双耳定位泛化
  - 在当前协议下整体弱于 `BiL-style` 和当前分流主线
- `early-fusion`
  - 仍然是一个有竞争力的 baseline
  - 但在 unseen-noise 条件下仍明显落后于 `dual cue`

### 已验证但不继续推进的变体

这些模型已经给出明确结论，但不再作为主线推进对象。保留在 README 里，是为了后面写论文时能快速引用负结果。

| 变体 | 结论 | 代表结果 |
|---|---|---|
| `dual cue + reliability gate` | 更保守、更稳健，但分类锐度下降；适合作为分析性变体，不替代主线 | `Acc 0.6642 / F1 0.3785 / Acc@10 0.8828 / MAE 11.44°` |
| `dual cue + TF-mask` | cue-side 时频 mask 没带来收益，整体落后于 concat 主线 | `Acc 0.6477 / F1 0.3755 / Acc@10 0.8374 / MAE 13.16°` |
| `dual cue + LSTM` | 更重、更差，不如当前 GRU | `Acc 0.6474 / F1 0.3700 / Acc@10 0.8438 / MAE 13.30°` |
| `dual cue + Mamba` | 更重、更差，不如当前 GRU，也没有优于 LSTM | `Acc 0.6468 / F1 0.3664 / Acc@10 0.8374 / MAE 13.31°` |
| `multi-scale temporal cue` | 改动不大，但无一致收益 | `Acc 0.6699 / F1 0.3897 / Acc@10 0.8552 / MAE 12.51°` |
| `band-weighting / band-attention cue` | 结构更花，但 `MAE / front-back / large error` 明显变差 | `Acc 0.6602 / F1 0.4003 / Acc@10 0.8118 / MAE 17.07°` |
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
  - `v7 + lite cue + cf80 + cue24 + gru80` 是当前更稳的轻量候选主线；它比 `encoder v2 balanced` 参数更少，并在三次 seed 下都保持了稳定、且有竞争力的 `Acc / F1 / MAE / Acc@10°`。
  - `v7 + dual cue value/reliability + cf80 + gru80` 是当前更适合作为论文方法主线的版本；它把 `ILD/sin(IPD)/cos(IPD)` 和 `coherence` 显式拆成 value / reliability 两支，在三次 seed 平均下取得了比 `cf80_cue24_gru80` 略优的综合结果。
  - 单流早融合 baseline 在验证阶段表现很好，但在 unseen-subject 测试集上最终明显落后于当前分流主线；这说明早融合可以形成强 baseline，但 content/cue 解耦对跨 subject 泛化更有优势。
  - 在轻量 cue 独立流上，`coherence` 不是冗余项；去掉 `coherence` 的 `ild_phase` 版本没有超过 `encoder v2 balanced`。
  - 在 lite cue 独立流上，`temporal conv` 不是冗余项；改成 `MLP-only` 会明显削弱分类表现。
  - `absdiff` 更偏向帮助分类锐度，而不是帮助 `MAE`；去掉后 `Acc / F1 / Acc@5° / Acc@10°` 会掉，但 `MAE` 和尾部错误会更保守。
  - 多尺度 temporal cue encoder 与 band-weighting cue encoder 都没有稳定超过当前主线；前者收益有限，后者在 `MAE / Acc@5° / Acc@10° / front-back error` 上明显变差。
  - 纯回归进一步降低 `MAE` 和结构性大错，但分类指标明显下降。
  - 分类 + 回归联合没有超过当前原生分类主线。
- 在外部 backbone 上：
  - `SDEL-DOA-Cls + fbaux` 是当前最强分类结果。
  - `SDEL-DOA-Reg` 的 `MAE` 最低，仍值得继续做 `+ fbaux` 或联合任务验证。
  - `BiL-style GCC-PHAT CRN` 是当前最强、最贴题的外部轻量 baseline；它在 `Accuracy / F1 / Acc@5°` 上已经接近甚至局部超过 `dual cue`，但 `MAE / front-back / large error` 明显更差，说明显式 `value / reliability` 分解更有利于控制结构性大错。
  - `FAViT-style ILD/IPD` 作为外部 Transformer baseline 没有超过 `BiL-style`，也没有接近当前主线；这说明在当前 `360° / 72类 / unseen-subject / unseen-noise` 协议下，频率优先 patch Transformer 并不会自然优于更贴近双耳定位物理结构的 GCC-PHAT 或显式 cue 分流建模。

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
