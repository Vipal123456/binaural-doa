# CIPIC-Roomsim25 正式数据集生成与评测协议

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-02
- Verification Status: PARTIALLY VERIFIED
- Version Label: cipic_roomsim25_plan_v1

## 1. 最终决定

本项目的正式跨人头数据集固定为 **CIPIC-Roomsim25 v1**：

1. 使用 DP-RTF 作者发布的 Roomsim-CIPIC BRIR，不把当前 SofaMyRoom CIPIC pilot 扩展为正式主数据。
2. 只使用 CIPIC 水平面上 25 个真实测量方向，不插值分类标签。
3. 沿用 DP-RTF 的 30/6/9 人头划分和 10/2/3 房间划分。
4. 语音改用本项目已有的 LibriSpeech，噪声使用已有 DEMAND；所有模型共享完全相同的合成波形。
5. 主数据之外，固定增加 Direct-HRIR seen/unseen 人头诊断集和 Neidhardt 实测 BRIR 外部测试集。
6. SofaMyRoom `interp=1` 数据只作为渲染器敏感性测试，不进入主表，不用于选择模型。

这是一套针对当前项目的协议，不是对 DP-RTF 波形数据的逐样本复现。复用的是其 CIPIC 人头、房间、BRIR、方向、距离、RT60 和测试条件设计；语音、噪声、2 s 输入和模型评测口径按本项目统一。

## 2. 研究问题与各数据集角色

| 数据集 | 唯一用途 | 能支持的结论 | 不能支持的结论 |
|---|---|---|---|
| CIPIC-Roomsim25 main | 训练、验证和主测试 | 静态单声源在未见人头、未见仿真房间及噪声下的联合泛化 | 单独归因于人头变化；真实录音泛化 |
| Direct-HRIR25 diagnostic | 隔离人头变量 | 无混响、无噪声条件下 seen/unseen 人头的性能差距 | 混响鲁棒性或真实房间泛化 |
| Neidhardt-BRIR25 external | 测量域外部测试 | 从 Roomsim-CIPIC 到 KEMAR 实测 BRIR 的跨设备、跨房间迁移 | 现场真人语音录音泛化 |
| SofaMyRoom25 sensitivity | 检查渲染器依赖 | 更换 BRIR 渲染规则时的性能敏感性 | 哪个渲染器等价于真实房间 |

重要混杂因素：DP-RTF 的每个 CIPIC subject 只绑定一个 room type。主测试同时更换人头和房间，因此主测试不能单独证明“跨人头能力”。Direct-HRIR25 用于补上这一诊断缺口。

## 3. 已确认来源与当前可用状态

### 3.1 已确认来源

- DP-RTF 论文：B. Yang, H. Liu, and X. Li, “Learning Deep Direct-Path Relative Transfer Function for Binaural Sound Source Localization,” IEEE/ACM TASLP, 2021. DOI: https://doi.org/10.1109/TASLP.2021.3120641
- 作者代码：https://github.com/BingYang-20/DP-RTF-Learning
- Roomsim 工具：https://github.com/bingo-todd/Roomsim_Campbell
- 本地 DP-RTF 配置：`/disk2/bywang/project/DP-RTF-Learning-main/code/common/config.py`
- 作者 BRIR 下载说明：`/disk2/bywang/project/DP-RTF-Learning-main/data/RIR/CIPIC/README.md`
- CIPIC SOFA：`/disk2/bywang/data/HRTF/subject_*.sofa`，本地已确认共 45 个人头。

### 3.2 当前缺失项

截至 2026-08-02，本地 DP-RTF 的 `data/RIR/CIPIC/` 目录只有 README，尚无作者发布的 BRIR 文件。因此正式生成开始前，必须先完成下载、文件枚举和 SHA-256 清单。不能在下载失败时静默切换到 SofaMyRoom。

### 3.3 Roomsim-CIPIC 的已知近似

Roomsim 对每条传播路径选择最近的 CIPIC 实测 HRIR，并非连续 HRTF 插值；其 CIPIC 测量盲区按官方实现置零。25 个目标直达方向都是实际测量方向，但任意反射到达方向仍会发生最近邻量化，部分盲区反射会被丢弃。

因此论文只能写“采用固定的 Roomsim-CIPIC BRIR 协议控制渲染误差”，不能写“消除了 HRTF 插值/反射方向误差”。

## 4. 固定任务定义

- 任务：静态、单语音源、水平面方位分类。
- 采样率：16 kHz。
- 输入长度：2.0 s，即双通道 `[32000, 2]`。
- 类别数：25。
- 类别中心（顺序固定）：

```text
[-80, -65, -55, -45, -40, -35, -30, -25, -20, -15,
 -10,  -5,   0,   5,  10,  15,  20,  25,  30,  35,
  40,  45,  55,  65,  80]
```

- 类别索引：上表位置 `0..24`。禁止再使用 `(-180, 180)` 上均匀 5 度公式推导标签。
- 预测角度：`pred_angle = class_angles_deg[argmax(logits)]`。
- MAE：`mean(abs(pred_angle - target_angle))`。本协议只有前方 `[-80,80]`，线性绝对误差与圆周最短误差数值相同。
- 左右手性：以项目现有 KEMAR/CIPIC 检查约定为基准，生成前用 `+80/-80` 的直达段 ILD/ITD 确认通道和标签符号，不通过检查不得生成全量数据。

## 5. 人头与房间划分

### 5.1 精确 subject-room 分配

| Split | Room | CIPIC subjects |
|---|---|---|
| Train | 707040 | 010, 028, 124 |
| Train | 806536 | 011, 012, 165 |
| Train | 506028 | 044, 127, 156 |
| Train | 456031 | 015, 017, 018 |
| Train | 708050 | 048, 050, 051 |
| Train | 679046 | 058, 059, 060 |
| Train | 405530 | 134, 135, 137 |
| Train | 538038 | 147, 148, 152 |
| Train | 383025 | 153, 154, 155 |
| Train | 503229 | 158, 162, 163 |
| Validation | 606035 | 061, 065, 119 |
| Validation | 406032 | 126, 131, 133 |
| Test | 608038 | 008, 009, 033 |
| Test | 507030 | 021, 003, 040 |
| Test | 404027 | 019, 020, 027 |

三个 split 的 subject ID 必须完全不重叠。验证集只用于 early stopping 和 checkpoint 选择；test 不用于调超参数、选择 seed 或选择 `best_acc`/`best_mae`。

### 5.2 房间、RT60 与距离

房间编码按 DP-RTF 配置解释为三维尺寸，例如 `707040 = 7.0 x 7.0 x 4.0 m`。

| Split | Room size (m) | RT60 候选 (s) | 距离候选 (m) |
|---|---:|---|---|
| Train | 7.0 x 7.0 x 4.0 | 0, .18, .36, .54, .72, .90 | 1.0, 2.0, 3.0 |
| Train | 8.0 x 6.5 x 3.6 | 0, .27, .54, .81 | 1.5, 2.9 |
| Train | 5.0 x 6.0 x 2.8 | 0, .21, .42, .63, .84 | .5, 1.5, 2.5 |
| Train | 4.5 x 6.0 x 3.1 | 0, .26, .52, .78 | .8, 2.2 |
| Train | 7.0 x 8.0 x 5.0 | 0, .17, .34, .51, .68, .85 | 1.5, 2.0, 2.5, 3.0, 3.4 |
| Train | 6.7 x 9.0 x 4.6 | 0, .28, .56, .84 | 3.0, 3.6 |
| Train | 4.0 x 5.5 x 3.0 | 0, .25, .50, .75 | .5, 1.0 |
| Train | 5.3 x 8.0 x 3.8 | 0, .23, .46, .69, .92 | 1.8, 2.4 |
| Train | 3.8 x 3.0 x 2.5 | 0, .30, .60, .90 | .75, 1.25 |
| Train | 5.0 x 3.2 x 2.9 | 0, .19, .38, .57, .76, .95 | .6, 1.2 |
| Validation | 6.0 x 6.0 x 3.5 | 0, .22, .44, .66, .88 | 1.75, 2.25 |
| Validation | 4.0 x 6.0 x 3.2 | 0, .24, .48, .72 | .75, 1.25 |
| Test | 6.0 x 8.0 x 3.8 | 见测试条件表 | .6, 1.5, 2.4, 3.3 |
| Test | 5.0 x 7.0 x 3.0 | 见测试条件表 | .7, 1.4, 2.1 |
| Test | 4.0 x 4.0 x 2.7 | 见测试条件表 | .8, 1.3 |

### 5.3 测试 RT60-SNR 配对

主测试固定为以下 8 个条件，不做随机组合：

| Condition | RT60 (s) | SNR (dB) | 用途 |
|---:|---:|---:|---|
| 1 | .60 | 15 | 固定混响下 SNR 曲线 |
| 2 | .60 | 10 | 固定混响下 SNR 曲线 |
| 3 | .60 | 5 | 两条曲线的交点 |
| 4 | .60 | 0 | 固定混响下 SNR 曲线 |
| 5 | .60 | -5 | 固定混响下 SNR 曲线 |
| 6 | .20 | 5 | 固定 SNR 下 RT60 曲线 |
| 7 | .40 | 5 | 固定 SNR 下 RT60 曲线 |
| 8 | .80 | 5 | 固定 SNR 下 RT60 曲线 |

不能把全部 `SNR=5 dB` 条件与其他 SNR 直接比较，因为它额外包含四种 RT60。正式 by-SNR 表只取 `RT60=.60 s`；正式 by-RT60 表只取 `SNR=5 dB`。

## 6. 语音与噪声

### 6.1 语音划分

| Split | LibriSpeech root | 规则 |
|---|---|---|
| Train | `train-clean-100` | 只用于训练 |
| Validation | `dev-clean` | 只用于验证 |
| Test | `test-clean` | 只用于所有测试集 |

要求：

- train/validation/test speaker ID 交集必须为空。
- 每个 2 s crop 记录 `speaker/chapter/utterance/start_sample`。
- split 间禁止同一 utterance；split 内禁止重复的完整 crop key。
- 小于 2 s 的语音不做循环拼接；优先丢弃，必要时仅在末尾补零并记录。
- 使用语音活动比例门槛，2 s crop 中有效语音占比至少 70%。
- BRIR 卷积后截取时保留完整直达起点，并避免把卷积尾部静音当作有效输入。

### 6.2 DEMAND 噪声

固定场景：`OOFFICE, PCAFETER, TMETRO, TBUS, SPSQUARE, NPARK`。

固定同步通道对：`(1,2), (3,4), ..., (15,16)`。同一样本的左右噪声必须取同一场景、同一时刻的一个固定通道对，禁止左右耳独立随机裁剪。

每个 DEMAND 文件按时间划分：

- Train：前 60%。
- Validation：中间 20%。
- Test：最后 20%。
- 相邻分区边界各留 2 s guard interval。

这能阻止完全相同的噪声片段跨 split 泄漏，但 DEMAND 通道对是环境麦克风阵列录音，不是 CIPIC 假人头双耳噪声。该限制必须在论文中披露。

### 6.3 SNR 定义与归一化

先完成 clean speech 与 BRIR 的双耳卷积，再加噪。只允许用一个共同增益缩放双耳信号，禁止左右耳分别归一化，以免破坏 ILD。

```text
P_s = mean(s_left^2 + s_right^2) / 2
P_n = mean(n_left^2 + n_right^2) / 2
scale_n = sqrt(P_s / (P_n * 10^(SNR/10)))
y = s + scale_n * n
```

若峰值溢出，最终对双耳共同缩放。metadata 同时保存目标 SNR 和实际 SNR；实际误差必须小于 0.2 dB。

训练与验证 SNR 固定为 `[-5, 0, 5, 10, 15, 20] dB`，边际分布近似均匀。主测试使用第 5.3 节的固定配对，不额外加入 clean 或 -15 dB。

## 7. 样本数量与采样规则

### 7.1 主数据集

| Split | 样本数 | 时长 | 构成 |
|---|---:|---:|---|
| Train | 120,000 | 66.67 h | 30 subjects x 25 angles x 160 realizations |
| Validation | 24,000 | 13.33 h | 6 subjects x 25 angles x 160 realizations |
| Test | 324,000 | 180.00 h | 27 subject-distance pairs x 8 RT/SNR x 25 angles x 6 scenes x 10 realizations |

Train/Validation 中，每个 subject-angle 固定 160 条。对该 subject 绑定房间内可用的 RT60、距离、6 个 SNR、6 个场景及 8 个通道对做确定性分层轮转；每个变量的边际计数最大差不得超过 1。不要先完全随机采样再期待事后平衡。

Test 中每个最细条件使用 10 个固定 speech/noise realization。所有模型、所有 seed 必须读取同一批 WAV 和 metadata，保证逐样本配对比较。

预计仅 PCM16 WAV 约占 60 GB；manifest、BRIR 缓存和临时文件不计在内。生成前至少预留 100 GB。

### 7.2 Direct-HRIR25 seen/unseen 人头诊断集

- Seen 组：正式 train 的全部 30 个已见人头。
- Unseen 组：正式 test 的 9 个未见人头。
- 条件：25 个准确实测方向、无混响、无噪声。
- 每个 subject-angle 使用 20 个固定 test-clean 语音 crop。
- 25 个方向各自固定 20 个 crop；同一个 angle-realization 在全部 seen/unseen subjects 上复用，形成配对条件。
- Seen：`30 x 25 x 20 = 15,000` 条；Unseen：`9 x 25 x 20 = 4,500` 条；总计 19,500 条，10.83 h。
- 使用 CIPIC 原始 HRIR 直接卷积，不插值，不用于 checkpoint 选择。

核心量是 `unseen subject-macro - seen subject-macro` 的 Accuracy/MAE 差值，而不是只看 unseen 绝对分数。该结果与 main test 联合解释：Direct-HRIR 的 seen/unseen gap 小而 main 差，说明主要问题来自房间/噪声或其交互；Direct-HRIR 已有明显 gap，才支持“模型本身缺乏跨人头稳定性”的判断。

### 7.3 Neidhardt-BRIR25 外部测试

- 只保留与 25 类完全一致的前方方向。
- 使用 5 个位置、2 个扬声器文件；每条 BRIR 使用 2 个不重叠 test-clean crop。
- 条件：`clean, -10, -5, 0, 5, 10 dB`。
- 总计：`5 x 2 x 25 x 2 x 6 = 3,000` 条。
- Positions 1-4 共 2,400 条作为主结果；已有几何异常的 Position 5 共 600 条单独报告，不混入主均值。
- 这仍是“语音与实测 BRIR 卷积”，不是现场语音录音。

详细坐标和通道规则沿用 `docs/neidhardt_brir_test_plan.md`，但必须重新生成 25 类标签，不能复用旧 72 类 label。

### 7.4 SofaMyRoom 渲染器敏感性集

仅在 main 模型确定后生成。使用同一批 test subjects、25 个方向、语音、噪声和测试 RT60/SNR 条件；房间尺寸和距离对齐 Roomsim 配置，但由于两个渲染器的路径模型不同，不宣称逐 BRIR 等价。结果放补充材料或误差分析，不用于决定主方法。

## 8. 输出格式

```text
data/librispeech_cipic_roomsim25_v1/
├── manifest.json
├── brir_inventory.csv
├── quality_report.json
├── train_subjects/
│   ├── binaural_dev/
│   ├── metadata_dev/
│   └── split_index.csv
├── val_subjects/
│   ├── binaural_dev/
│   ├── metadata_dev/
│   └── split_index.csv
├── test_subjects_unseen/
│   ├── binaural_dev/
│   ├── metadata_dev/
│   └── split_index.csv
└── diagnostics/
    ├── direct_hrir25_unseen/
    ├── neidhardt_brir25/
    └── sofamyroom25_renderer_sensitivity/
```

每条 metadata 至少包含：

```json
{
  "file_id": "test_000000001",
  "wav_path": "binaural_dev/binaural000000001.wav",
  "class_index": 0,
  "azimuth_deg": -80.0,
  "subject_id": "008",
  "room_id": "608038",
  "rt60_s": 0.6,
  "distance_m": 0.6,
  "target_snr_db": -5.0,
  "achieved_snr_db": -5.01,
  "speech_path": "...",
  "speech_speaker_id": "...",
  "speech_start_sample": 0,
  "noise_scene": "TBUS",
  "noise_channels": [1, 2],
  "noise_start_sample": 0,
  "brir_path": "...",
  "brir_sha256": "...",
  "renderer": "Roomsim_Campbell",
  "seed": 42
}
```

`manifest.json` 必须记录生成脚本 git commit、Python/NumPy/SciPy/libsndfile 版本、所有输入根目录、随机种子、BRIR 下载来源和 BRIR 清单哈希。

## 9. 生成前后质量门槛

以下任一硬门槛失败，数据集不得进入训练：

1. 45 个 CIPIC SOFA 齐全；30/6/9 subject 集合互斥。
2. Roomsim BRIR inventory 与配置要求的 subject-room-RT60-distance-angle 组合一致；缺失文件数为 0。
3. 每条音频 16 kHz、双通道、32000 samples、无 NaN/Inf、非静音。
4. 双耳只使用共同增益；随机抽查卷积前后 ILD 不被归一化改变。
5. `+80/-80` 的直达段 ITD/ILD 与项目方位手性一致；通道反转数为 0。
6. 25 个 label 只能来自固定列表，class-index/angle 反查错误数为 0。
7. train/validation/test speech speaker 交集为 0；utterance 交集为 0。
8. DEMAND 时间分区无交叠；跨 split 完整 noise crop key 重复数为 0。
9. 实际 SNR 与目标 SNR 的绝对偏差不超过 0.2 dB。
10. Train/Validation 每个 subject-angle 恰为 160 条；分层变量边际最大计数差不超过 1。
11. Test 最细条件恰为 10 条；总数恰为 324,000。
12. 对非零 RT60 估计值生成统计报告；偏差超过 `max(0.1 s, 20%)` 的 BRIR 标为异常并人工检查，不能自动删除后继续训练。

附加诊断但不是自动通过条件：

- 绘制每个人头 25 方向的直达 ITD/ILD 曲线，检查方向单调性和左右镜像趋势。
- 随机试听每个 split 至少 20 条。
- 对每个 split 计算 peak、RMS、静音比例、裁剪比例和 SNR 误差分布。
- 检查 Roomsim 盲区置零是否造成异常短 BRIR，并在 manifest 中记录统计。

## 10. 训练与评测约束

- 主模型和 SDEL、DP-RTF、BiL、FN-SSL 等对比模型全部从头训练，不能复用 KEMAR checkpoint 作为正式结果。
- 固定 seeds `42, 43, 44`；第一轮可只跑 seed 42 排错，但正式表报告三 seed mean ± std，同时列出每个 seed。
- checkpoint 只按 validation `best_acc.pth` 选择；`best_mae.pth` 仅做补充分析。不得在 test 上挑单次最好 seed 作为论文主结果。
- 训练、验证和测试数据对所有模型完全一致；模型专属预处理只能改变特征，不能改变样本集合。
- 主要比较采用逐样本配对结果；多 seed 报告均值和标准差。

## 11. 指标与正式表格

主指标：

- Exact Accuracy：argmax 类别是否与标签类别完全相同。
- MAE：argmax 类别中心与真实类别中心的绝对角度误差。
- Acc@5 degrees、Acc@10 degrees。
- LargeErr：绝对误差大于 30 degrees 的比例。
- SideErr：真实角度非 0 时，预测落到相反左右半平面的比例。

分组结果：

- 9 个 test subject 分别报告，再做 subject-macro mean ± std。
- 25 个方向分别报告，并汇总 angle-macro 指标。
- by-SNR：只使用 `RT60=.60 s` 的五个 SNR。
- by-RT60：只使用 `SNR=5 dB` 的四个 RT60。
- 按距离和 DEMAND scene 分组。
- Direct-HRIR25 和 Neidhardt-BRIR25 单独成表，不与 main test 合并平均。

本任务没有后方类别，因此不要报告 front/back confusion；SideErr 只描述左右侧混淆。

## 12. 明确禁止的论文表述

- 禁止：“CIPIC 插值误差已被消除。”
- 禁止：“主测试单独证明了未见人头泛化。”
- 禁止：“Neidhardt 是真实现场录音测试。”
- 禁止：“25 类分类可以输出任意连续角度。”
- 禁止：“测试集上最好的单个 seed 是方法的代表性能。”
- 禁止把旧 KEMAR 72 类结果与新 CIPIC 25 类 Accuracy 直接比较高低。

允许的准确表述是：

> We evaluate static single-source localization on 25 measured frontal CIPIC directions. The main Roomsim-CIPIC test jointly changes listener, room, distance, reverberation, and noise. A direct-HRIR diagnostic isolates unseen-listener variation, while a measured-BRIR test evaluates external transfer to a different dummy head and a real measured room response.

## 13. 实施顺序与停止条件

1. 下载作者 BRIR，生成 `brir_inventory.csv` 和 SHA-256 清单。
2. 实现 BRIR 解析器，只生成每个 split 25 条 smoke set。
3. 通过通道、角度、长度、RT60 和 SNR 质检。
4. 生成 19,500 条 Direct-HRIR25 seen/unseen 配对诊断数据，但不用于选 checkpoint。
5. 生成主数据 Train 120k 和 Validation 24k，训练主模型 seed 42。
6. 用 validation 选出的主模型在 Direct-HRIR25 上比较 seen/unseen gap；先排除标签、通道和基础跨人头失败。
7. 只在验证和诊断流程正常后生成 Test 324k，并冻结 test manifest。
8. 主模型达到可用结果后，训练全部对比模型和另外两个 seeds。
9. 最后生成 Neidhardt-BRIR25 和可选 SofaMyRoom 敏感性集。

停止条件：若 Direct-HRIR25 上的 unseen 结果接近随机、出现大规模左右反转，或相对 seen 组发生极大退化，不应继续耗费空间生成 324k 主测试；先检查标签/通道，再判断是否需要修改模型。若作者 BRIR 文件无法取得或清单不完整，暂停正式主集，不得用当前 `interp=1` pilot 冒充替代。
