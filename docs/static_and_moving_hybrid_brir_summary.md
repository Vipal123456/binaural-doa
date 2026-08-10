# 静态与移动说话人双耳 DOA 实验总结

本文档只总结当前两套正式数据集与正式实验结果，不包含早期 `order2`、debug、smoke、clean-only 等试错数据。

## 1. 项目背景

本项目研究双耳语音声源定位（binaural DOA estimation）。输入为左右耳语音信号提取的双耳特征，输出为水平面 360 度方位角类别。

当前任务分为两条主线：

- 静态单说话人 DOA 分类：每个 2s 片段输出一个方位类别 `[B, 72]`。
- 移动单说话人 DOA 序列估计：每个 4s 样本每 100ms 输出一次方位类别，输出 `[B, 40, 72]`。

类别设置保持一致：水平面 `[-180, 180)`，72 类，每 5 度一个 bin。数据划分采用 CIPIC subject-disjoint split，即训练、验证、测试使用不同 HRTF subjects，以模拟更真实的 HRTF mismatch 泛化场景。

## 2. 正式数据集

### 2.1 静态数据集

数据集路径：

```text
data/librispeech_cipic_multisubject_static_hybridbrir_gate2_50h_v1
```

核心设置：

| 项目 | 设置 |
|---|---|
| 语音源 | LibriSpeech train-clean-100 |
| HRTF | CIPIC / SOFA，多 subject |
| 采样率 | 16 kHz |
| 原始样本时长 | 10s wav |
| 训练切片 | 2s segment |
| DOA 类别 | 72 类，每 5 度 |
| 距离 | 1.0-1.5m |
| 噪声 | DEMAND |
| SNR | Uniform[-10, 10] dB |
| split | 24 train / 3 val / 3 unseen test subjects |
| 数量 | train 14400, val 1800, test 1800 recordings |

Subject split：

```text
train: 003,008,011,012,020,021,027,028,033,048,051,058,059,060,061,065,119,124,126,131,133,134,135,147
val:   152,154,155
test:  156,163,165
```

### 2.2 移动说话人数据集

数据集路径：

```text
data/librispeech_cipic_moving_hybridbrir_gate2_50h_v1
```

核心设置：

| 项目 | 设置 |
|---|---|
| 样本时长 | 4s |
| 标签步长 | 100ms |
| 标签长度 | 40 |
| 输出形式 | `[B, 40, 72]` |
| 总时长 | 50h |
| 数量 | train 36000, val 4500, test 4500 |
| split | 与静态数据集相同的 subject-disjoint split |
| 轨迹类型 | static 20%, linear 60%, piecewise 20% |
| 距离 | 1.0-1.5m，样本内固定 |
| SNR | Uniform[-10, 10] dB |
| 默认训练标签 | `rendered_label_seq` |

移动轨迹中保存两套标签：

- `target_angle_seq`：轨迹生成器产生的理论连续角度。
- `direct_rendered_angle_seq`：direct path 实际选择的最近 CIPIC HRIR 方位角。
- `target_label_seq`：理论角量化后的 72 类标签。
- `rendered_label_seq`：实际渲染角量化后的 72 类标签。

当前正式移动实验默认使用 `label_source=rendered`。原因是模型听到的是实际 HRIR/BRIR 渲染后的声学角度，而不是连续理论角度。评估时同时报告 pred-vs-rendered 和 pred-vs-target，用来量化 CIPIC 最近邻离散化误差。

## 3. 混响与噪声合成方式

两套正式数据集使用同一类混响思想：`hybrid_pathwise_hrtf_brir`。

### 3.1 Early reflections

早期反射使用 image-source method 生成 direct path 和早期 reflection paths。每条路径计算：

- image source position
- path distance
- path delay
- attenuation/gain
- arrival azimuth/elevation
- selected CIPIC HRIR index

每条路径都根据其到达方向选择当前 subject 的最近 CIPIC HRIR，并把左右耳 HRIR 按 delay 和 gain 叠加到 BRIR_L / BRIR_R。左右耳 ITD/ILD 由 CIPIC HRIR 引入，path delay 只按 head center 计算，避免重复加入双耳时延。

正式设置：

```text
early_max_order = 3
early_cut_ms = 80
keep path if delay <= direct_delay + 80ms
```

### 3.2 Late tail

晚期混响使用 binaural diffuse statistical late tail，而不是简单两个独立白噪声。设计目标是让混响尾巴更接近真实房间的双耳扩散场：

- late tail 从 80ms 后平滑接入。
- 衰减按目标 RT60 标定。
- 左右耳 late tail 做频率相关 coherence 控制。
- 低频左右更相关，高频更去相关。
- late tail 能量锚定 early decay energy，不做任意比例拼接。
- 最终 BRIR 上估计 RT60，并通过 quality gate 筛选。

正式设置：

```text
late_start_ms = 80
brir_seconds = 1.8
quality_gate_required = true
```

### 3.3 噪声

DEMAND 噪声在完整 reverb binaural waveform 渲染并全局归一化后加入。左右耳优先使用同一 DEMAND scene 的不同 channel；若只有单通道，则使用同一 segment 做轻微 decorrelation，不使用左右完全不同 scene。

静态正式数据集统计显示：

```text
SNR range: about -10 to 10 dB
target RT60 range: about 0.20 to 0.80 s
estimated RT60 range: about 0.10 to 0.86 s
num_paths range: 38 to 63
```

## 4. 标签与方位一致性检查

当前正式数据集没有出现“后方目标角被错误映射到 ±80° HRIR”的问题。SOFA 文件在水平附近包含后方方向，例如 `100, 115, 125, 135, 150, 170, 175, -180` 等。

实际 target-rendered mismatch 统计：

| 数据集 | mean | median | p95 | max |
|---|---:|---:|---:|---:|
| static gate2 | 3.07° | 2.50° | 7.50° | 7.50° |
| moving gate2 | 2.35° | 1.79° | 6.83° | 10.00° |

后方区域 `|az| > 100°` 没有大面积错配：

| 数据集 | 后方 mean mismatch | 后方 max mismatch |
|---|---:|---:|
| static gate2 | 2.81° | 7.50° |
| moving gate2 | 2.03° | 7.50° |

因此当前正式数据集可以作为 360° DOA 实验使用。主要离散误差集中在 `±90°` 附近，最大约 10°。

## 5. 模型框架

### 5.1 输入特征

所有模型都基于同一套双耳特征接口：

- `log_mag_L / log_mag_R`：左右耳 log magnitude，用于提供语音内容、频谱包络和 HRTF 频谱着色。
- `ILD`：interaural level difference，主要提供中高频左右能量差线索。
- `IPD`：interaural phase difference。实际输入通常使用 `sin(IPD)` 和 `cos(IPD)`，避免相位在 `-pi/pi` 处跳变。
- `coherence`：左右耳相干性，反映当前 T-F bin 的双耳线索是否可靠，在混响和噪声下尤其重要。

这些特征的意义是把“听到什么内容”和“左右耳之间有什么空间差异”分开建模。前者帮助模型避开语音内容变化带来的干扰，后者直接服务 DOA 判断。

### 5.2 v7 dual-cue value/reliability

v7 dual-cue 是当前项目的主线结构，核心思想是显式分解双耳线索的数值和可靠性。

content encoder：

- 左右耳 `log_mag` 分别进入共享的 `BinauralEncoderV2Balanced`。
- encoder 通道设置通常为 `[24, 40, 64]`，输出维度 `encoder_out_dim=96`。
- 左右耳共享权重，保证两耳内容表征处在同一特征空间。
- 得到 `f_l` 和 `f_r` 后构造 `mean / diff / absdiff` 关系特征。
- `mean` 表示双耳共同语音内容，`diff` 表示左右差异方向线索，`absdiff` 表示差异强度。
- 关系特征经过 linear + layer norm + ReLU 压到 `content_fusion_dim=80`。

cue value encoder：

- 输入 `ILD, sin(IPD), cos(IPD)`。
- 先做 frequency band pooling，再用轻量 temporal convolution 编码。
- 输出维度通常为 `cue_value_out_dim=24`。
- 作用是提取“当前双耳差异指向哪个方向”的方向性线索。

cue reliability encoder：

- 输入 `coherence`。
- 使用比 value branch 更轻的 temporal convolution。
- 输出维度通常为 `cue_reliability_out_dim=8`。
- 作用是判断当前双耳线索是否可信。例如强混响、噪声或低相干区域不应被过度依赖。

fusion：

- 默认 `concat`：把 content feature、cue value feature、cue reliability feature 拼接后送入 temporal head。
- 也支持 gate 形式：用 reliability 去调制 value cue，但正式 moving rendered-label 主线采用 concat 更稳定。

### 5.3 v7 litecueenc concat all

litecueenc 是 dual-cue 的同族轻量强 baseline。它保留 content encoder 和 `mean/diff/absdiff` 内容关系，但 cue 侧不再拆分 value/reliability。

cue encoder：

- `cue_feature_mode=all` 时输入 `ILD, sin(IPD), cos(IPD), coherence` 四类 cue。
- 统一经过 band pooling + temporal convolution。
- 输出维度通常为 `cue_encoder_out_dim=32`。

作用意义：

- 优点是结构更简单，参数和分支更少，所有双耳线索直接联合编码。
- 缺点是可解释性弱于 dual-cue，不能显式区分“方向线索值”和“线索可靠性”。
- 当前结果显示，它在静态 MAE 和部分移动轨迹上表现很强，说明统一 cue 编码对尾部大错有一定抑制作用。

### 5.4 GRU / sequence temporal head

静态和移动任务的 temporal head 不完全相同。

静态版本：

- 输入为 encoder 输出的帧级特征 `[B, T, D]`。
- 使用 BiGRU 或同类 temporal encoder 建模时间上下文。
- 再通过 mean pooling 或 attention pooling 得到 clip-level 表征。
- 最终输出 `[B, 72]`。

移动版本：

- 输入仍是帧级特征 `[B, T, D]`，4s 音频约 400 个 STFT frame。
- 先用 `AdaptiveAvgPool1d(label_steps=40)` 显式聚合到 40 个 100ms label step。
- 再用 BiGRU 建模相邻 DOA step 的轨迹上下文。
- v7 moving 中通常使用 `gru_hidden_size=80, gru_num_layers=1`，输出维度为 `2H=160`。
- 每个 step 经过同一个 linear classifier 输出 72 类，得到 `[B, 40, 72]`。

GRU 的作用不是简单平滑预测，而是在每个 100ms step 上利用前后语音和空间线索，减少孤立跳变，并帮助 linear/piecewise trajectory 的动态跟踪。`jitter` 指标可以侧面反映 temporal head 是否过度跳变。

### 5.5 SDEL-DOA CRNN baseline

SDEL-DOA 是新增的论文风格 CRNN 对比模型，目前在 moving 测试中表现最好。

输入构造：

- `MBMS proxy = 0.5 * (log_mag_L + log_mag_R)`，近似表示双耳平均谱。
- `ILD`
- `cos(IPD)`
- `sin(IPD)`
- 四个特征 stack 成 `[B, 4, T, F]`。

encoder 结构：

- 3 个 2D CNN block，通道为 `[32, 64, 128]`。
- 每个 block 为 `Conv2d + BatchNorm2d + Tanh + MaxPool2d + Dropout2d`。
- 频率池化为 `[4, 4, 4]`，时间池化为 `[1, 1, 1]`，因此主要压缩频率维，不破坏时间轨迹。

GRU 与融合：

- 使用 2 层 BiGRU，`gru_hidden_size=128`。
- BiGRU 输出 `[B, T', 2H]` 后，把前向和后向 hidden 分成两半。
- 使用 `tanh(forward) * tanh(backward)` 做 bidirectional multiplicative fusion，得到 `[B, T', H]`。
- 静态版本对时间平均池化后输出 `[B, 72]`。
- 移动版本先 adaptive pooling 到 40 step，再对每个 step 共享 MLP + classifier，输出 `[B, 40, 72]`。

作用意义：

- CNN block 负责提取局部 T-F 空间线索。
- BiGRU 负责长时间上下文。
- 乘性融合比直接 concat 更强调前后文一致的方向证据。
- 对 moving DOA 来说，它比 v7 主线更重视连续时序建模，因此 linear trajectory 上 MAE 很低。

### 5.6 外部/论文风格 baseline

当前已复现或适配的其他对比模型：

- BiL-style GCC-PHAT CRN：以 GCC-PHAT cross-correlation 特征为核心的双耳定位 baseline。采用 3 层渐进式 CNN [32,64,96] + 1 层双向 GRU (hidden=96) + PReLU MLP，约 600k 参数。双向 GRU 使其在移动 DOA 序列任务上能有效利用轨迹时序上下文（同时看到前后帧），因此保留为主 baseline。另有一个 official-backbone 变体（3 层 128-channel CNN + 2 层单向 GRU，865k 参数），但因其单向 GRU 无法建模未来帧、且无 dropout 正则化导致移动任务严重过拟合（MAE 从 14.58° 退化到 21.94°），故不作为主 baseline。详见第 11 节分析。
- FAViT-style ILD/IPD：基于 ILD/IPD patch/token 的 Transformer-style baseline，用来测试 patch attention 对双耳 cue 的建模能力。

## 6. 训练与指标

静态训练：

- 输入 2s segment。
- 输出 `[B, 72]`。
- loss 使用分类交叉熵，并按模型配置使用 label smoothing / front-back auxiliary 等。

移动训练：

- 输入 4s waveform。
- 输出 `[B, 40, 72]`。
- 默认 loss 为 frame-wise cross entropy：

```python
loss = F.cross_entropy(
    doa_logits.reshape(-1, 72),
    doa_labels.reshape(-1)
)
```

移动指标重点：

- `frame_mae`：逐 100ms DOA 平均角误差。
- `frame_accuracy`：逐帧 exact class accuracy。
- `Acc@5° / Acc@10°`：容忍 5/10 度内的准确率，比 exact accuracy 更适合动态 DOA。
- `front_back_error_rate`：前后半平面错误率。
- `large_error_rate`：误差 > 90°。
- `opposite_error_rate`：误差 > 150°。
- `jitter`：相邻预测角度变化均值，反映轨迹平滑性。
- `trajectory_mae/static, linear, piecewise`：按轨迹类型分组 MAE。

## 7. 静态正式结果

测试集为 unseen subjects，共 9000 个 2s segments。

| 模型 | Acc | F1 | MAE | Acc@5 | Acc@10 | FB err | Large err | Opp err | 参数量 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v7 dual-cue value/reliability | 0.6859 | 0.3956 | 15.77° | 0.7831 | 0.8600 | 0.1100 | 0.0949 | 0.0126 | ~187k |
| v7 litecueenc concat all cf80 gru80 | 0.6708 | 0.3879 | **14.57°** | 0.7720 | 0.8478 | **0.1008** | **0.0828** | **0.0054** | ~150k |
| BiL-style GCC-PHAT CRN | 0.6426 | 0.3614 | 18.43° | 0.7474 | 0.8308 | 0.1261 | 0.1168 | 0.0116 | ~600k |
| FAViT-style ILD/IPD | 0.5740 | 0.3136 | 24.08° | 0.6622 | 0.7566 | 0.1857 | 0.1678 | 0.0167 | — |

注：BiL 保留了旧版结构 ([32,64,96] CNN + 双向 GRU) 作为主 baseline。官方 backbone 变体 (865k, 3×128 CNN + 单向 GRU + dropout=0) 的 MAE=17.58°，opposite error 最低 (0.66%) 但 front MAE 偏高 (28.09°)，且参数量过大，仅在消融中作为参考。

按空间区域分解 MAE：

| 模型 | front MAE | back MAE | side MAE |
|---|---:|---:|---:|
| v7 dual-cue | 15.42° | 31.27° | 6.11° |
| v7 litecueenc | 17.66° | 26.15° | 7.10° |
| BiL-style GCC-PHAT CRN | — | — | — |

静态结果解读：

- litecueenc 整体 MAE 最低（14.57°），large error 和 opposite error 也最低，是用最少参数获得最好鲁棒性的主线。
- dual-cue 在 Acc / F1 / Acc@5 / Acc@10 分类指标上最好，说明显式 value/reliability 分解有利于整体细粒度分类。
- BiL-style GCC-PHAT CRN 是有价值的传统/论文风格 baseline，GCC-PHAT 特征在强混响下仍有竞争力。作为外部 baseline 证明了时延特征与 CRN 组合的有效性。
- FAViT-style 在本数据协议下明显落后，说明直接把 ILD/IPD patch 化后做 Transformer 对 HRTF mismatch 和混响噪声不够稳。

## 8. 移动正式结果

以下结果使用 `label_source=rendered`，主评估为 pred-vs-rendered HRTF angles。测试集为 4500 条 4s 样本，共 180000 个 frame-level labels。

| 模型 | Frame Acc | MAE | Acc@5 | Acc@10 | FB err | Large err | Opp err | Jitter | 参数量 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SDEL-DOA CRNN | 0.6639 | **7.34°** | 0.7862 | **0.9033** | **0.0395** | **0.0200** | **0.0128** | **2.61°** | ~400k |
| v7 litecueenc concat all cf80 gru80 | 0.6462 | 9.14° | **0.7723** | 0.8751 | 0.0600 | 0.0288 | 0.0169 | 3.20° | ~150k |
| v7 dual-cue value/reliability | 0.6511 | 9.22° | 0.7651 | 0.8767 | 0.0599 | 0.0289 | 0.0169 | 3.26° | ~187k |
| v7 dual-cue (target label) | 0.4706 | 8.34° | 0.7521 | 0.9033 | 0.0608 | 0.0277 | 0.0137 | 3.21° | ~187k |
| BiL-style GCC-PHAT CRN | 0.6050 | 14.58° | 0.7080 | 0.8191 | 0.1072 | 0.0632 | 0.0256 | 3.31° | ~600k |
| FAViT-style ILD/IPD | 0.3968 | 37.64° | 0.4744 | 0.5585 | 0.2628 | 0.1848 | 0.0737 | 40.69° | — |

按轨迹类型分组：

| 模型 | static MAE | linear MAE | piecewise MAE |
|---|---:|---:|---:|
| SDEL-DOA CRNN | 14.25° | **5.15°** | **7.74°** |
| v7 litecueenc | 18.72° | 6.39° | 8.76° |
| v7 dual-cue (rendered) | 16.47° | 6.69° | 10.38° |
| v7 dual-cue (target label) | 17.49° | **5.18°** | 9.69° |
| BiL-style GCC-PHAT CRN | 22.63° | 11.83° | 15.67° |
| FAViT-style | 38.39° | 37.40° | 37.67° |

移动结果解读：

- SDEL-DOA CRNN 是当前 moving rendered-label 协议下最强结果，MAE、Acc@10、front-back error、large error 和 jitter 均优于其他模型。提升主要来自双向乘性融合和更深的 2 层 BiGRU。
- litecueenc 和 dual-cue 非常接近，是当前 moving 任务中两条稳定的 v7 主线。litecueenc 的 Acc@5 最高，dual-cue (target label) 的 MAE 和 Acc@10 最优。
- **v7 dual-cue 使用 target label 训练的 MAE (8.34°) 和 Acc@10 (90.33%) 均优于 rendered label 版本**，说明细粒度的 72 类均匀标签比 CIPIC 量化标签更有利于模型学习精确角度映射。
- BiL-style 明显弱于两条 v7 主线，说明仅依赖 GCC-PHAT/时延峰在动态混响 BRIR 中不够稳。但其双向 GRU 保留了基本的轨迹跟踪能力 (linear 11.83° vs static 22.63°)。
- FAViT 在移动任务上失败特征明显：MAE 高、front/back error 高、jitter 极大，说明逐帧序列稳定性不足。
- 所有有效模型的 static trajectory MAE 仍高于 linear trajectory，说明静态片段中的前后混淆和 unseen-subject HRTF mismatch 仍是主要误差来源。

## 9. 静态与移动结果如何对比

静态和移动不能直接只比较 exact accuracy，因为任务定义不同：

- 静态是 2s clip-level 单标签分类。
- 移动是 4s frame-level 序列分类，每 100ms 一个标签。

建议论文中这样对比：

- 静态主表报告 `Accuracy / F1 / MAE / Acc@5 / Acc@10 / FB err / Large err`。
- 移动主表报告 `Frame Accuracy / Frame MAE / Acc@5 / Acc@10 / FB err / Large err / Jitter`。
- 静态与移动共同强调 `MAE / Acc@5 / Acc@10 / front-back / large error`，而不是单独比较 exact accuracy。

为什么移动的 MAE 看起来比静态更低：

- 移动模型输出 40 帧序列，GRU 可以利用轨迹连续性，天然减少孤立跳变。
- 当前移动数据的 linear 轨迹占 60%，轨迹连续性较强，模型可以从时间上下文获益。
- 静态测试是 2s clip-level，前后混淆会直接造成大角度尾部错误，拉高 MAE。

因此不能简单说移动任务"更容易"，而应解释为：移动任务在序列上下文和 rendered-label 策略下，局部帧级定位更稳定；但 static trajectory 分组 MAE 仍较高，说明前后混淆和 unseen-subject HRTF mismatch 仍是核心难点。

### 两个模型的跨任务表现对比

| 模型 | 静态 MAE | 移动 MAE | 跨任务趋势 |
|---|---|---|---|
| Lite Cue | 14.57° | 9.14° | 移动远优于静态（GRU 时序上下文极大帮助） |
| Dual Cue | 15.77° | 9.22° | 同上 |
| BiL-style GCC-PHAT | **18.43°** | **14.58°** | 移动优于静态，但 MAE 绝对值仍高于 v7 |

## 10. 当前结论

1. 正式静态和移动数据集的 360° 标签与 HRTF 渲染方向是可用的，没有后方大面积映射到 ±80° 的问题。
2. `hybrid_pathwise_hrtf_brir` 比简单 mono RIR -> HRTF 更合理，因为每条 early path 都按 arrival direction 独立选择 HRIR，同时 late tail 使用双耳扩散混响建模。
3. **静态任务上，Lite Cue 用最少参数 (~150k) 获得最低 MAE (14.57°)**，是对比中最优的静态主线。Dual Cue 在分类指标上最好。BiL GCC-PHAT (MAE=18.43°) 作为传统 baseline 证明了时延特征仍有竞争力。
4. **移动任务上，SDEL-DOA CRNN 最强（MAE 7.34°），得益于双向乘性融合 + 2 层 BiGRU。** Lite Cue 为 v7 最优 (9.14°)。Dual Cue target-label 版本 MAE=8.34°，证明 target label 优于 rendered label。
5. v7 模型在移动任务上相对 BiL 的优势明显 (9.14° vs 14.58°)，说明显式 ILD/IPD/coherence 特征在移动序列建模中优于单一 GCC-PHAT 特征。
6. FAViT 在静态和移动上均失败，证明纯 ILD/IPD Transformer 不适合强混响双耳 DOA。
7. **参数效率**：Lite Cue (~150k, MAE 14.57° 静态 / 9.14° 移动) 是所有模型中性价比最高的，适合部署场景。

## 11. BiL Official Backbone 变体分析

本章记录了一个未采用为主 baseline 的 BiL 变体及其弃用原因。

### 11.1 变体结构

参照 Yang et al. (ICASSP 2024) 开源 BiL 仓库的官方 backbone：3 层 128-channel CNN + 2 层单向 GRU (hidden=128) + PReLU MLP，865k 参数，输出头按本项目协议改为 72 类分类。

### 11.2 与主 BiL baseline 的差异

| 组件 | 主 BiL baseline (采用) | Official 变体 (未采用) |
|---|---|---|
| CNN channels | [32, 64, 96] 渐进式 | [128, 128, 128] 全宽 |
| GRU 方向 | **双向 (bidirectional=True)** | **单向 (bidirectional=False)** |
| GRU 层数/宽度 | 1 层 × hidden=96 | 2 层 × hidden=128 |
| Dropout | 0.1 | **0.0** |
| 参数量 | ~600k | 865k |
| Conv bias | False | True |

### 11.3 实验结果与弃用原因

| 指标 | 主 BiL (静态) | Official 变体 (静态) | 主 BiL (移动) | Official 变体 (移动) |
|---|---|---|---|---|
| MAE | 18.43° | 17.58° | **14.58°** | **21.94°** |
| Opposite error | 1.16% | **0.66%** | 2.56% | 3.77% |

Official 变体在静态上略优于主 BiL (MAE −0.85°)，opposite error 为所有模型中最低 (0.66%)，说明宽 CNN + 深 GRU 对静态 GCC-PHAT 特征有一定提升。但移动上 MAE 从 14.58° 退化到 21.94°，三种轨迹 MAE 几乎无差异 (~20-27°)。

**弃用原因：**

1. **单向 GRU 无法建模移动轨迹**：移动 DOA 的 40 帧序列需要看到前后帧来推断当前方向。单向 GRU 只能利用过去帧，失去未来信息，导致 linear/static/piecewise 三种轨迹 MAE 无差异——模型未学到轨迹模式。
2. **无 Dropout 导致过拟合**：865k 参数 + dropout=0.0 + 仅 36k 移动训练样本 → 严重过拟合。训练 loss 持续下降但验证 MAE 在 18.75° 附近停滞，test MAE 进一步退化至 21.94°。
3. **参数分配不合理**：移动任务更需要强时序建模而非强特征提取。Official 变体将大量参数投入 CNN（3×128），却削弱了 GRU 的时序能力（单向 128-dim vs 主 BiL 的双向 192-dim）。

**结论**：Official backbone 不适合本项目移动说话人序列任务，保留主 BiL ([32,64,96] + 双向 GRU) 作为正式 baseline。

## 12. 下一步方向

优先级从高到低：

1. 对移动任务做 subject/angle/trajectory 分组诊断，重点分析 static trajectory 为什么 MAE 高于 linear。
2. 在移动任务上尝试轻量 temporal smoothing 或 transition-aware loss，但只作为消融，不替代 CE baseline。
3. 增加轨迹可视化：随机抽取 test 样本画 target/rendered/pred 曲线，展示动态跟踪能力和跳变问题。
4. 写论文时把 target-vs-rendered mismatch 统计作为数据集 sanity check，说明当前 360° 标签物理一致。
5. 如果要进一步增强真实性，可以考虑用真实 measured BRIR 数据库做外部 test-only 泛化评估，但不建议替换当前主协议。
