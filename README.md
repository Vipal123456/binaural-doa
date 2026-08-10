# DOA-Net: directional-interference robust binaural localization

本仓库研究静态双耳水平声源定位：给定一段含混响目标语音和单个定向非语音干扰源的双通道音频，预测目标语音的水平到达方向（DOA）。当前实验统一使用 `librispeech_cipic_roomsim25_directional_dns_v4` 数据协议，并以 **CueFactorized-CPSD5（B2）** 作为推荐主线。

当前任务不是连续角度回归，而是在 25 个非均匀水平角上分类：

```text
[-80, -65, -55, -45, -40, -35, -30, -25, -20, -15, -10, -5,
   0,   5,  10,  15,  20,  25,  30,  35,  40,  45,  55,  65,  80]
```

## 数据协议

### 信号构造

- 目标内容：LibriSpeech，训练、验证、测试分别使用 `train-clean-100`、`dev-clean` 和 `test-clean`。
- 双耳渲染：目标和干扰分别通过 Roomsim-CIPIC BRIR 渲染；二者共享受试者、房间和混响时间，但独立选择声源距离和方位。
- 干扰内容：DNS Challenge 3 wideband noise。构建清单时通过 AudioSet 人声本体、文件名规则和 Silero VAD 排除人声相关片段。
- 干扰几何：每条样本包含一个定向点干扰源，与目标至少相隔 `20 deg`；角距离按 `20-40`、`45-80`、`85-160 deg` 三档采样。
- SIR 定义：BRIR 渲染后，在目标活跃采样点上计算双耳目标/干扰功率比。
- 幅度归一化：混合后对左右耳使用同一个缩放系数，避免破坏 ILD。
- 音频格式：`16 kHz`、双通道、每条 `2.0 s`。

DNS 原始 source ID、LibriSpeech 说话人、CIPIC 受试者和模拟房间均按 split 隔离。需要准确理解这一协议：每个 CIPIC 受试者只属于一个模拟房间，因此测试衡量的是**未见受试者与未见房间的联合泛化**，不能单独归因于跨人头泛化。

### 数据规模

| Split | 音频数 | 受试者数 | 房间数 | 每类样本数 | 无混响 R0 比例 |
|---|---:|---:|---:|---:|---:|
| Train | 120,000 | 30 | 10 | 4,800 | 10% |
| Validation | 12,000 | 6 | 2 | 480 | 10% |
| Test | 64,800 | 9 | 3 | 2,592 | 0% |

训练 SIR 在四个等数量连续区间内采样：`[-5,0)`、`[0,5)`、`[5,10)`、`[10,15] dB`。验证集使用 `-5/0/5/10/15 dB`。

测试集由两个受控扫描组成，每个条件包含同一组 `8,100` 个配对样本：

- SIR 扫描：固定 `RT60=0.6 s`，SIR 为 `-5/0/5/10/15 dB`。
- RT60 扫描：固定 `SIR=5 dB`，RT60 为 `0.2/0.4/0.6/0.8 s`。

两个扫描共享 `RT60=0.6 s, SIR=5 dB` 条件，所以测试集共有 8 个条件而不是 9 个。解释鲁棒性时应分别报告两个扫描，不能把联合平均值当作单一均匀分布。

### 数据审计

当前正式数据已通过完整审计：类别计数均衡、split 间 DNS source ID 和语音说话人无交集、最小角距离满足 `20 deg`，生成记录中的最大 SIR 误差小于 `5.3e-7 dB`。审计报告位于：

```text
data/librispeech_cipic_roomsim25_directional_dns_v4/manifest.json
data/librispeech_cipic_roomsim25_directional_dns_v4/quality_report.json
outputs/logs_cipic_roomsim25_directional_dns_v4_seven_model_queue/dataset_audit.json
```

## 推荐模型：CueFactorized-CPSD5

模型的出发点是：ILD 取决于稳定的双耳自功率比，而 IPD 取决于稳定的复互功率谱；定向干扰和混响对这两类统计量的破坏方式并不相同。因此，B2 不再用同一组时间权重估计所有空间线索，而是在每个时频点的 5 帧局部窗口中分别估计 ILD 和 IPD 的可靠性权重。

```text
双耳波形
  -> STFT (n_fft=512, win=400, hop=160)
  +-> 共享幅度编码器 -> 80D content feature -------------------+
  |                                                             |
  +-> 5-frame CueFactorized-CPSD                                 |
        +-> ILD-specific weights -> auto-power ratio -> ILD      |
        +-> IPD-specific weights -> complex CPSD -> sin/cos IPD  |
        +-> coherence                                            |
             -> ordered LocalTF32 -> 24D cue feature ------------+
                                                                  |
                         concat + LayerNorm -> BiGRU(80)
                             -> temporal attention -> 25 classes
```

局部窗口内首先构造相对能量、相对 pilot ILD 的一致性和相对 pilot phase 的一致性。随后使用两个独立、可学习且零初始化的评分器：

```text
ILD score = a_E * relative_energy + a_I * ILD_agreement
IPD score = b_E * relative_energy + b_P * phase_agreement
```

评分经过局部 softmax 后分别聚合自功率谱和复互谱，再计算 `ILD`、`sin(IPD)`、`cos(IPD)` 与 coherence。零初始化使训练起点严格退化为均匀 5 帧 CPSD，而不是随机门控。复互谱在求相位前完成加权，避免直接平均周期角度。

空间编码器保留 32 个有序频带，使用独立的 ILD 与 IPD Local-TF 卷积分支；coherence 作为两个分支的局部上下文，不再额外建立独立 coherence 输出支路。幅度内容特征和 24D 空间特征在进入单层双向 GRU 前拼接，最终通过时间注意力池化完成分类。

### 受控变体

| 名称 | 作用 |
|---|---|
| LocalTF32 | 不做局部 CPSD，检验有序 32 频带空间编码本身。 |
| RW-CPSD5 | ILD/IPD 共用一组 5 帧可靠性权重。 |
| B2 CueFactorized-CPSD5 | ILD/IPD 使用不同权重；当前推荐主线。 |
| B4 Target-Aware | 训练时用目标/干扰分量监督目标占优概率；用于机制和上限分析，不是 B2 的稳定改进。 |
| D2 PreCommon24 | 目标感知 CPSD 配合 24D 公共能量内容支路；用于参数量与精度权衡。 |

## 当前结果

以下结果来自同一个 64,800 条 directional-DNS 正式测试集，均使用各自按验证 MAE 选择的 checkpoint。`Acc` 是 25 类严格分类准确率；`Acc@5` 和 `Acc@10` 按预测角与真实角的绝对角误差阈值统计，因此不等同于分类准确率。

| 模型 | 可训练参数 | MAE (deg) | Acc | Acc@5 | Acc@10 |
|---|---:|---:|---:|---:|---:|
| **B2 CueFactorized-CPSD5** | 195,422 | **3.449** | **56.48%** | **85.96%** | 95.26% |
| B4 Target-Aware | 195,591 | 3.491 | 55.92% | 85.78% | 95.38% |
| RW-CPSD5 | 195,421 | 3.577 | 54.77% | 85.73% | 95.20% |
| D2 PreCommon24 | **147,695** | 3.600 | 54.05% | 85.75% | 95.19% |
| LocalTF32 | 195,418 | 3.680 | 54.27% | 85.35% | 94.76% |
| DP-RTF | 未统计 | 3.760 | 52.26% | 85.78% | **96.00%** |
| SDEL | 919,771 | 3.844 | 52.84% | 84.14% | 94.55% |
| BIL | 未统计 | 10.093 | 45.43% | 74.30% | 84.08% |

B2 相对 RW-CPSD5 的 MAE 改善为 `0.128 deg`，相对 LocalTF32 为 `0.231 deg`；这支持“线索专属时间可靠性优于共享权重”的当前假设，但差值仍小，不能仅凭单次实验宣称普遍优势。D2 减少约 24.4% 参数，但 MAE 比 B2 高 `0.151 deg`，更适合作为轻量化工作点。

B2 的分条件结果如下：

| SIR @ RT60=0.6 s | -5 dB | 0 dB | 5 dB | 10 dB | 15 dB |
|---|---:|---:|---:|---:|---:|
| MAE (deg) | 6.143 | 3.880 | 3.056 | 2.757 | 2.662 |

| RT60 @ SIR=5 dB | 0.2 s | 0.4 s | 0.6 s | 0.8 s |
|---|---:|---:|---:|---:|
| MAE (deg) | 2.883 | 2.886 | 3.056 | 3.326 |

## 训练与评测

安装依赖：

```bash
pip install -r requirements.txt
```

训练当前推荐模型：

```bash
python train.py \
  --config configs/train_cipic_roomsim25_directional_dns_v4_v7_localtf32_cuefactor_cpsd5_bestmae_seed42.yaml
```

常规测试：

```bash
python evaluate.py \
  --config configs/train_cipic_roomsim25_directional_dns_v4_v7_localtf32_cuefactor_cpsd5_bestmae_seed42.yaml \
  --checkpoint outputs/checkpoints_cipic_roomsim25_directional_dns_v4_v7_localtf32_cuefactor_cpsd5_bestmae_seed42/best_mae.pth
```

按 SIR、RT60、房间、受试者和角距离分组评测：

```bash
python tools/evaluate_cipic_compound_grouped.py \
  --config configs/train_cipic_roomsim25_directional_dns_v4_v7_localtf32_cuefactor_cpsd5_bestmae_seed42.yaml \
  --checkpoint outputs/checkpoints_cipic_roomsim25_directional_dns_v4_v7_localtf32_cuefactor_cpsd5_bestmae_seed42/best_mae.pth \
  --test_root data/librispeech_cipic_roomsim25_directional_dns_v4/test \
  --output_dir outputs/b2_directional_dns_grouped \
  --device cuda:0
```

重新审计数据集：

```bash
python tools/audit_cipic_roomsim25_directional_dns_v4.py \
  --dataset-root data/librispeech_cipic_roomsim25_directional_dns_v4 \
  --noise-inventory data/dns3_directional_v4_inventory/dns3_noise_inventory.csv \
  --report outputs/directional_dns_v4_audit.json \
  --workers 16
```

完整生成命令需要预先准备 LibriSpeech、Roomsim-CIPIC BRIR 和经过筛选的 DNS3 噪声清单。默认路径可在命令行显式覆盖：

```bash
python tools/generate_cipic_roomsim25_directional_dns_v4.py \
  --noise_inventory data/dns3_directional_v4_inventory/dns3_noise_inventory.csv \
  --output_root data/librispeech_cipic_roomsim25_directional_dns_v4 \
  --mode full \
  --workers 6 \
  --parallel_splits
```

## 代码结构

```text
configs/                 训练配置
dataset/                 特征提取与静态数据加载
models/native_lite_v7.py CPSD、LocalTF32 与主模型
models/temporal_head.py   BiGRU 与时间池化
engine/                  训练和评测流程
tools/                   数据生成、审计、诊断和分组评测
tests/                   数据协议与模型回归测试
```

## 结论边界

- 当前表格是 `seed=42` 的单 seed 结果，尚缺少多 seed 均值、方差和配对置信区间。
- 模型设计过程中多次查看了同一测试集，因此现阶段结果适合方法筛选，不应作为无偏的最终论文数字；最终结论需要冻结方法后使用独立测试集复核。
- 数据全部由模拟 BRIR 渲染，尚不能替代真实录制条件下的验证。
- 测试同时更换受试者和房间，不能从当前协议单独分离两种域偏移的贡献。
- `BiGRU + attention` 使用未来帧，模型属于轻量非因果定位器；在完成 MACs、实时系数和延迟测量前，不宣称实时运行。
- B2 对共享 RW-CPSD5 的增益有限。当前证据支持继续研究线索专属可靠性，但不足以证明更复杂的目标掩码或不确定性模块必然有效。
