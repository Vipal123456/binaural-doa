# Paper Fact Sheet

> 状态：从零撰写论文的事实底稿，不是论文正文。  
> 当前主线：静态、单声源、双耳水平 DOA 分类；`KEMAR + SofaMyRoom + diffusefg`。  
> 代表方法：代码标识 `v7_dualcue_liteenc_v1`；正式论文名称待补充。  
> 证据标签：`[已确认]` 表示可由当前代码、配置、元数据或结果文件直接核对；`[合理推测]` 表示设计解释但尚无充分实验或文献证据；`[待补充]` 表示当前不能写成论文事实。

## Evidence scope and exclusions

- `[已确认]` 当前 README 明确将旧 `robust50h multisubject`、CIPIC subject-disjoint、移动声源以及重型 attention/gate/Mamba/LSTM 探索排除出当前论文主线。
- `[已确认]` 本 Fact Sheet 因此不采用 `PERFORMANCE_COMPARISON_TABLE.md`、`docs/GENERALIZATION_RESULTS_REPORT.md` 和 `docs/static_and_moving_hybrid_brir_summary.md` 中的旧协议结果。
- `[已确认]` 主结果只采用 7,776 条完整 KEMAR diffusefg 测试样本；`outputs/grouped_eval_runs_m15/` 中每次仅 648 条样本的结果不作为主结果。
- `[已确认]` 角度指标采用当前 `utils/angle.py` 的离散方向定义 `angle = -180 + 5 * bin`。旧结果把类别当作区间中心，导致正确分类也产生 2.5 度误差；README 中的 MAE/Acc@k 尚未同步，最终论文表必须统一重生成。

## 1. Research problem

- `[已确认]` 研究静态单说话人的双耳水平声源到达方向估计（binaural horizontal DOA estimation）。
- `[已确认]` 目标是在混响、加性场景噪声和双通道扩散噪声条件下，从左右耳信号估计完整 360 度方位角。
- `[合理推测]` 核心困难是 ILD、IPD 等空间线索会被噪声和混响扰动，且不同时间频率区域的线索可靠性不一致。
- `[待补充]` 应用场景、目标部署平台和实时性要求尚未定义。

## 2. Input and output

### Input

- `[已确认]` 原始输入为 2.0 s、16 kHz 的双通道音频，张量约定为 `[2, N]`，通道 0/1 分别为左/右耳。
- `[已确认]` STFT 参数为 `n_fft=512`、`win_length=400`、`hop_length=160`、Hann window，得到约 `T=201` 个时间帧和 `F=257` 个频点。
- `[已确认]` 主模型实际使用：左右耳 `log_mag_L/log_mag_R`、ILD、`sin(IPD)`、`cos(IPD)` 和局部时频 coherence。

### Output

- `[已确认]` 主输出为 `[B, 72]` 方位类别 logits，覆盖 360 度水平面，每 5 度一个类别。
- `[已确认]` 训练时另有 `[B, 2]` front/back 辅助分类输出；推理的主要任务输出仍为 72 类 DOA。
- `[已确认]` 当前是 clip-level 静态分类，不是连续角度回归，也不是逐帧移动轨迹估计。

## 3. Task setting

- `[已确认]` 静态、单声源、水平面、全方位 360 度、72 类闭集分类。
- `[已确认]` 语音使用 LibriSpeech `train-clean-100`，通过 KEMAR HRTF 和 SofaMyRoom 房间响应渲染为双耳语音，再加入 DEMAND 场景噪声生成的 diffuse-field 双通道背景。
- `[已确认]` 训练/验证 SNR 从 `Uniform[-10, 10] dB` 采样；测试包含 `clean, 10, 5, 0, -5, -10 dB`。
- `[已确认]` 测试集的噪声场景 `TBUS/NPARK/SPSQUARE` 与训练/验证的 `TMETRO/PCAFETER/OOFFICE` 不重合。
- `[已确认]` 当前不是 unseen-HRTF 或跨 dummy-head 泛化：训练、验证和测试均使用 KEMAR。
- `[待补充]` 是否将该设置命名为 noise-scene generalization、room generalization 或 in-domain robustness，需要在统计房间生成分布后严格界定。

## 4. Existing-method limitations

以下内容目前只能作为待验证的研究动机，不能写成已经被文献证明的事实：

- `[合理推测]` 在完整时频平面上进行重型时序/频率建模会带来较高计算量；当前复现的 FN-SSL 为 65.687 GFLOPs。
- `[合理推测]` 单独依赖手工或显式空间 cue 的模型在低 SNR 和混响下可能因 cue 污染而退化。
- `[合理推测]` 把内容表示和所有空间 cue 直接混合，可能缺少对“方向线索值”和“线索可靠性”的结构区分。
- `[待补充]` 必须通过正式文献检索和原论文核对，确认上述局限是否公平、是否已有方法解决；当前仓库没有可用于论文的引用库。

## 5. Proposed method

- `[已确认]` 当前推荐主模型是 `v7_dualcue_liteenc_v1`，配置类型为 `native_lite_v7_dual_cue_concat`。
- `[已确认]` 方法由三条信息路径组成：共享权重的左右耳内容编码、显式双耳内容关系分解、value/reliability 双分支空间 cue 编码。
- `[已确认]` 三路表示在时间维对齐后拼接，经 LayerNorm/Dropout、单层 BiGRU 和 attention pooling，输出 72 类 logits，并可输出 front/back 辅助 logits。
- `[待补充]` 正式模型名称。现有复杂度表使用过 `BRCNet`，但仓库没有给出该缩写的正式定义，不能直接用于论文。

## 6. Each module and its motivation

| 模块 | 已确认的实现 | 设计动机及证据状态 |
|---|---|---|
| Feature extractor | 左右耳 log magnitude、ILD、sin/cos IPD、局部 coherence | `[合理推测]` 显式提供内容与物理双耳线索，降低主干从波形隐式学习全部空间关系的负担。 |
| Shared `LightContentEncoderV1` | 左右耳共享同一轻量 CNN；通道 `[16,24,32]`，输出 64 维逐帧表示 | `[合理推测]` 共享权重使双耳表示可比较并降低参数量；尚无“非共享权重”对照。 |
| Content relation | `mean(F_L,F_R)`, `F_L-F_R`, `abs(F_L-F_R)` 拼接后投影至 80 维 | `[合理推测]` 同时表达共享内容、带符号差异和差异幅值；尚缺逐项消融，不能声称三项各自必要。 |
| Cue value branch | ILD、sin(IPD)、cos(IPD) 经 16-band pooling 和 temporal convolution，输出 24 维 | `[已确认]` 表示显式方向 cue；其作用与 reliability 分支分离。 |
| Cue reliability branch | coherence 经独立轻量 encoder，输出 8 维 | `[已确认]` 去除该分支后平均 Acc 下降约 0.55 个百分点、MAE 增加约 0.49 度；支持其在当前协议下有益，但尚无显著性检验。 |
| Late fusion | 80 维 content 与 32 维 cue 拼接为 112 维 | `[合理推测]` 保留两类信息的独立结构，再交给时序头整合；缺少与参数匹配 early-fusion 的当前协议对照。 |
| Temporal head | 单层 BiGRU，双向 hidden size 80；attention pooling | `[合理推测]` 聚合 2 s 内时序证据；缺少 mean pooling、无 GRU、单向 GRU 的参数匹配消融。 |
| Front/back auxiliary head | 二分类辅助头 | `[已确认]` 主配置权重为 0.3；尚无主模型三随机种子的 on/off 对照，不能声称它必然提升主模型。 |

## 7. Training objective

- `[已确认]` 主损失为带 label smoothing 的 72 类交叉熵，`label_smoothing=0.1`。
- `[已确认]` 总目标为 `L = L_DOA + 0.3 * L_front/back`。
- `[已确认]` 当前轻量主模型未启用 angular regression、anti-confusion loss、circular soft-label loss 或 front/back focus reweighting。
- `[已确认]` 优化器为 AdamW，初始学习率 `5e-4`，weight decay `1e-4`；cosine schedule，最多 100 epochs，最低学习率 `1e-5`，gradient clipping 1.0，early-stopping patience 15。
- `[已确认]` batch size 64；三次主实验使用 seeds 42/43/44。
- `[待补充]` checkpoint 选择准则、实际停止 epoch、训练硬件、训练时间、软件版本和确定性设置需要汇总。

## 8. Datasets and experimental protocol

| Split | 样本数 | 时长 | 房间 | 噪声/SNR |
|---|---:|---:|---|---|
| Train | 36,000 | 20 h | 随机 small/medium/large SofaMyRoom | `TMETRO/PCAFETER/OOFFICE`，diffusefg，SNR uniform `[-10,10]` dB |
| Validation | 7,200 | 4 h | 随机 small/medium/large SofaMyRoom | 与训练相同的三个 noise scene，SNR uniform `[-10,10]` dB |
| Test | 7,776 | 4.32 h | 6 个固定房间 `S1,S2,M1,M2,L1,L2` | clean 加三个未见 noise scenes；6 个 SNR 条件 |

- `[已确认]` 每条样本 2 s；训练每个角度 500 条，验证每个角度 100 条，测试每个角度 108 条，类别均衡。
- `[已确认]` 测试设计为 `6 rooms x 72 directions x 6 SNR conditions x 3 samples = 7,776`。
- `[已确认]` 目标 RT60 范围：训练/验证约 0.25--0.80 s，测试 0.30--0.80 s；声源距离约 1--2 m。
- `[已确认，高风险]` 语音文件不是 speaker/utterance disjoint：测试的 6,784 个唯一 speech paths 中有 4,868 个也出现在训练集，约 71.8%。这不等于已证明发生标签泄漏，但会削弱对内容泛化的主张。
- `[待补充]` 生成器版本、随机种子、BRIR/ANF 参数、数据许可和完整数据生成命令需要冻结到论文补充材料。

## 9. Baselines

当前可保留的对比模型：

- `[已确认]` SDEL：CNN + BiGRU 风格，0.926 M parameters / 1.402 GFLOPs。
- `[已确认]` FN-SSL：重型时频双向序列建模，0.659 M / 65.687 GFLOPs。
- `[已确认]` DP-RTF：显式 DP-RTF 表征路线，0.877 M / 8.215 GFLOPs。
- `[已确认]` BiL：GCC-PHAT CRN 路线，0.865 M / 3.117 GFLOPs。
- `[排除主表候选]` FAViT：当前结果较弱且训练/复杂度统计稳定性不足，可放附录或不报告。
- `[待补充]` 上述模型是否忠实复现原论文、是否做了等预算调参、原始引用及实现差异尚未审计。在完成该审计前，应称为 `*-style reproduction`，不应暗示是原论文官方结果。

## 10. Evaluation metrics

- `[已确认]` Accuracy：72 类 exact-bin accuracy。
- `[已确认]` MAE：预测离散角和真实角之间的 circular absolute error，范围 0--180 度。
- `[已确认]` Acc@5/Acc@10：圆周角误差不超过 5/10 度的样本比例。
- `[已确认]` Front/back half-plane error：预测与真值落在不同前/后半平面的比例；front 定义为 `|azimuth| <= 90 deg`。
- `[已确认]` Large error：角误差 `>=45 deg`；opposite error：当前 grouped evaluator 使用 `>150 deg`。
- `[待补充]` 论文主指标优先级、置信区间和配对显著性检验方案。

## 11. Main quantitative results

下表使用完整 7,776 样本测试集，并按当前修正后的角度映射从每个 run 的 `pred_bin` 重新核算。数值为多次运行的 mean +/- population std。

| Model | Runs | Acc (%) | MAE (deg) | Acc@5 (%) | Acc@10 (%) | Params (M) | FLOPs (G) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v7_dualcue_liteenc_v1` | 3 | 96.51 +/- 0.10 | 2.46 +/- 0.06 | 97.96 +/- 0.05 | 98.06 +/- 0.07 | 0.153 | 0.391 |
| SDEL (`nofb`) | 3 | **97.18 +/- 0.08** | **2.18 +/- 0.05** | **98.24 +/- 0.09** | **98.34 +/- 0.08** | 0.926 | 1.402 |
| FN-SSL | 3 | 96.75 +/- 0.10 | 2.33 +/- 0.15 | 98.23 +/- 0.14 | 98.32 +/- 0.17 | 0.659 | 65.687 |
| DP-RTF | 3 | 95.49 +/- 0.07 | 4.22 +/- 0.17 | 97.12 +/- 0.10 | 97.20 +/- 0.12 | 0.877 | 8.215 |
| BiL | 3 | 93.97 +/- 0.26 | 5.41 +/- 0.20 | 95.57 +/- 0.20 | 95.72 +/- 0.21 | 0.865 | 3.117 |

必须如实表述：

- `[已确认]` SDEL 在当前绝对定位指标上优于提出模型；不能声称提出模型达到 overall SOTA 或击败全部 baselines。
- `[已确认]` 提出模型的核心优势是效率：相对 SDEL 参数少约 83.5%、FLOPs 少约 72.1%，但 Acc 低约 0.67 个百分点、MAE 高约 0.27 度。
- `[已确认]` 相对 FN-SSL，提出模型参数少约 76.8%、FLOPs 少约 99.4%，指标接近但并未全面更优。
- `[待补充]` 上表中 FN-SSL 三次运行混合了 auxiliary on/off 设置；最终主表应补齐完全一致的三随机种子设置。

## 12. Ablation evidence

所有消融均为完整测试集、seeds 42/43/44、修正角度映射后的 mean +/- population std。

| Variant | Acc (%) | MAE (deg) | Acc@5 (%) | Acc@10 (%) | 结论边界 |
|---|---:|---:|---:|---:|---|
| Full model | **96.51 +/- 0.10** | **2.46 +/- 0.06** | **97.96 +/- 0.05** | **98.06 +/- 0.07** | 当前完整结构 |
| No content stream | 95.95 +/- 0.24 | 2.94 +/- 0.16 | 97.57 +/- 0.16 | 97.68 +/- 0.16 | content stream 在当前组合中有益，但 cue-only 仍然很强 |
| No reliability branch | 95.96 +/- 0.14 | 2.95 +/- 0.03 | 97.57 +/- 0.01 | 97.67 +/- 0.01 | coherence reliability branch 有益 |
| Merged cue branch | 96.16 +/- 0.18 | 3.05 +/- 0.19 | 97.58 +/- 0.13 | 97.67 +/- 0.15 | value/reliability 分支分离优于当前 merged 实现 |

- `[已确认]` 三类去除/合并变体的平均指标均低于完整模型。
- `[待补充]` 尚无 paired bootstrap、置信区间或显著性检验，因此不能把小差异写成统计显著。
- `[待补充]` 仍缺：共享/非共享 encoder、`mean/diff/absdiff` 各项、band pooling、BiGRU、attention pooling、front/back auxiliary、不同 cue 输入的参数匹配消融。

## 13. Supported contributions

当前证据最多支持以下贡献措辞：

1. `[已支持]` 提出一种紧凑的双流双耳 DOA 架构，将共享内容上下文、显式双耳关系和空间 cue 编码结合起来，在 KEMAR diffuse-field 测试协议上以 0.153 M 参数和 0.391 GFLOPs 获得接近更大基线的定位性能。
2. `[已支持]` 在当前实现和数据协议下，将 ILD/IPD cue value 与 coherence reliability 分支分离，优于去除 reliability 或合并 cue 的对照。
3. `[已支持]` 内容流在显式 cue 已很强的情况下仍提供一致的增益，说明内容上下文与空间 cue 在当前任务中具有互补性。
4. `[部分支持]` 构建了包含未见噪声场景、多个房间、混响和多 SNR 条件的 KEMAR diffuse-field 静态评测协议；其真实性与外部有效性仍需真实录音或外部 BRIR 数据验证。

当前不支持以下表述：

- 提出模型在绝对精度上优于所有 baselines。
- 模型已证明可泛化到 unseen HRTF、真实房间、真实设备或移动声源。
- 每个设计模块都不可替代或均有统计显著贡献。
- diffusefg 一定比旧噪声协议“更真实”；当前只有生成机制差异，没有感知或真实录音验证。

## 14. Missing information or experiments

### High priority before writing Abstract/Introduction claims

1. `[待补充]` 冻结唯一评测定义，并用当前 angle mapping 重生成 README、主表、by-SNR 表和所有 baselines；删除旧/新 MAE 混用风险。
2. `[待补充]` 建立 speech-disjoint 或 speaker-disjoint train/val/test，至少量化当前约 71.8% test speech-path overlap 对结果的影响。
3. `[待补充]` 完成 baseline fidelity audit：原论文引用、输入特征、模型改动、训练预算和调参范围。
4. `[待补充]` 给所有主模型使用一致的 seeds、辅助损失设置、checkpoint 选择和训练预算。
5. `[待补充]` 对主模型与 SDEL/FN-SSL、完整模型与消融模型做配对置信区间或显著性检验。

### High priority for contribution strength

6. `[待补充]` 外部有效性：真实双耳录音、独立 BRIR/HRTF 数据或至少 unseen dummy head/HRTF 测试。
7. `[待补充]` 当前主张所需的关键参数匹配消融，尤其是 content relation、shared encoder、temporal head 和 auxiliary loss。
8. `[待补充]` 报告运行时、峰值显存/内存和目标硬件延迟；FLOPs 不能单独证明可部署性。
9. `[待补充]` 报告 corrected per-SNR、per-room、front/back/side 和大误差结果，确认优势是否集中在特定条件。

### Reproducibility and writing inputs

10. `[待补充]` 论文题目、正式方法名、目标会议/期刊、作者与机构。
11. `[待补充]` 完整数据生成配置、软件/硬件版本、训练时长、实际 early-stop epochs、checkpoint hashes 和代码 commit。
12. `[待补充]` 数据与第三方实现许可。
13. `[待补充]` Related Work 文献检索与 BibTeX；当前 Fact Sheet 不包含任何未经核对的引用。
14. `[待补充]` 失败案例可视化和定性分析，特别是低 SNR、前后混淆和大角度错误。

## Evidence index

- 当前项目范围与汇总：`README.md`
- 代表配置：`configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_metricfix_seed42_g5.yaml`
- 特征：`dataset/feature_extractor.py`
- 标签与数据加载：`dataset/static_dataset.py`
- 主模型：`models/native_lite_v7.py`, `models/encoder.py`, `models/temporal_head.py`
- 损失与训练：`losses.py`, `engine/trainer.py`
- 指标定义：`utils/angle.py`, `metrics.py`, `tools/evaluate_kemar_grouped.py`
- 数据事实：三个当前数据目录下的 `metadata.csv`
- 主结果与消融：`outputs/grouped_eval_runs/*metricfix*/overall.json` 和 `per_sample.csv`
- 复杂度：`outputs/paper_diffusefg_macs.csv`

