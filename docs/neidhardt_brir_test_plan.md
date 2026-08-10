# Neidhardt 小型会议室实测 BRIR 外部测试方案 (v2)

本协议用于评估模型从 SofaMyRoom 仿真 BRIR 到真实房间实测 BRIR 的外部泛化。
最终语音由 LibriSpeech 与实测 BRIR 卷积生成，并非在会议室内直接录制的语音，
因此论文中统一称为 **measured-BRIR external generalization**，不称为
**real-recording generalization**。

## 1. 数据集概况

| 属性 | 值 |
|---|---|
| 来源 | Neidhardt et al., DAGA 2019 / Zenodo 2593714 |
| 人头 | KEMAR 45BA |
| 房间 | 小型会议室，SOFA `RoomDescription` 标注宽带 T60 = 0.63s |
| 方位角 | 72 个方向，5° 步长，360° 全覆盖 |
| 位置数 | 5 个（同一房间内不同听者位置） |
| 扬声器 | 2 个固定位置（0° 和 180°） |
| 扬声器条件 | Genelec 1030A，距听者 2.5m；朝向/背向听者 |
| 采样率 | 44100 Hz |
| BRIR 长度 | 1.0s |
| 文件数 | 10 个 SOFA（5 pos × 2 LS） |
| 总大小 | ~35 MB（已下载到 `/disk2/bywang/data/neidhardt_brir/`） |

## 2. 数据集结构与标签映射

### 2.1 文件结构

```
neidhardt_brir/
├── Pos1_LS_0.sofa    # 位置1，扬声器在0°
├── Pos1_LS_180.sofa  # 位置1，扬声器在180°
├── Pos2_LS_0.sofa
├── Pos2_LS_180.sofa
├── ...
└── Pos5_LS_180.sofa
```

每个 SOFA 内部：
```
Data.IR:  [72, 2, 1, 44100]
          M=72 个测量，R=2 个接收器（左/右耳），E=1 个发射器，N=44100 采样
```

### 2.2 标签映射（关键）

**SOFA 方位角与项目 KEMAR 方位角的左右手性相反，必须先转换坐标再生成标签。**

SOFA 几何坐标中，测量索引对应 `wrap(-5*M)`；项目训练协议中，正负方向约定与之
镜像，因此正式标签使用 `wrap(+5*M)`。这个符号不是根据模型分数选择的，而是通过
独立侧向通道检查确定：原 KEMAR 测试集 `+90°` 为 receiver/channel 1 占优，
Neidhardt 只有在 `M=18 -> +90°` 时呈现相同关系。

**LS_0 文件（扬声器在 0°）**：
```python
# 转换到项目 KEMAR 方位角手性
azimuth_deg = wrap_deg(float(5 * M))
# M=0  →   0° (正前)
# M=9  →  45°
# M=18 →  90°
# M=36 → 180° (正后) — 注意 wrap 到 ±180
# M=54 → -90°
# M=71 →  -5°
```

**LS_180 文件（扬声器在 180°）**：
```python
# 两种扬声器朝向使用相同的相对方位网格
azimuth_deg = wrap_deg(float(5 * M))
# M=0  →   0° (正前)
# M=9  →  45°
# M=18 →  90°
# M=36 → 180° (正后) — 注意 wrap 到 ±180
# M=54 → -90°
# M=71 →  -5°
```

**统一标签生成**：
```python
from utils.angle import wrap_deg

def angle_to_label(az_deg, num_classes=72):
    """实测方向位于 5 度网格，映射到最近的固定方向类别。"""
    wrapped = wrap_deg(az_deg)
    return int(np.rint((wrapped + 180.0) / 5.0)) % num_classes

# LS_0:
azimuth_deg = wrap_deg(float(5 * M))
label = angle_to_label(azimuth_deg)

# LS_180:
azimuth_deg = wrap_deg(float(5 * M))
label = angle_to_label(azimuth_deg)
```

### 2.3 标签映射验证表

| M | Head Rot | LS_0: 声源 Az | LS_0: Bin | LS_180: 声源 Az | LS_180: Bin | 物理方向 |
|---|---------|-------------|----------|----------------|-----------|---------|
| 0 | 0° | 0° | 36 | 0° | 36 | 正前 |
| 9 | 45° | 45° | 45 | 45° | 45 | 项目 +45° |
| 18 | 90° | 90° | 54 | 90° | 54 | 项目 +90° |
| 27 | 135° | 135° | 63 | 135° | 63 | 项目 +135° |
| 36 | 180° | 180° (-180°) | 0 | 180° (-180°) | 0 | 正后 |
| 45 | 225° | -135° | 9 | -135° | 9 | 项目 -135° |
| 54 | 270° | -90° | 18 | -90° | 18 | 项目 -90° |
| 63 | 315° | -45° | 27 | -45° | 27 | 项目 -45° |
| 71 | 355° | -5° | 35 | -5° | 35 | 项目 -5° |

### 2.4 Label Sanity Check（生成前必须运行）

生成第一个 segment 之前，对 8 个关键方向做 ILD/ITD 检查：

```python
test_angles = [0, 45, 90, 135, 180, -135, -90, -45]  # 8 个关键方向

对于 LS_0 文件:
  M = int((test_angle % 360) / 5)  # 反推 M
  取 BRIR 的前 3ms（直达声部分）
  计算 left_rms / right_rms 的 dB 差

验证:
  test_angle = +90° → channel 1 能量 > channel 0，与原 KEMAR 测试一致 ✅
  test_angle = -90° → channel 0 能量 > channel 1，与原 KEMAR 测试一致 ✅
  test_angle = 0° (正前)    → 左右耳能量接近       ✅
  test_angle = 180° (正后)  → 左右耳能量接近       ✅
```

如果检查失败（如 +90° 出现左耳能量更大的反转），则需要调整 `azimuth_deg` 公式中的符号。

## 3. 音频合成管道

### 3.1 语音源

`/disk2/bywang/data/LibriSpeech/LibriSpeech_test/test-clean`（2620 条 FLAC，363 MB），与训练 `train-clean-100` 话者完全不重叠。

### 3.2 Segment 生成规则

```
目标: 每条 BRIR → 固定 2 个 2s crop（不滑窗，避免高相关 segment）
例外: 语音太短（< 2s）→ 取 1 个 crop 或 padding

步骤:
  1. 从 test-clean 随机取 720 条不重复语音
  2. 每条语音与对应 BRIR 卷积
  3. 随机取 2 个不重叠的 2s 起始位置
     → 每条 BRIR 产生 2 个 [32000, 2] segment
     → 总 clean segment: 720 × 2 = 1440（不是之前的 ~2000）
  4. 按 6 个 SNR 条件分别加噪
     → 总 segment: 1440 × 6 = 8640
     → 每 SNR 约 1440 个 segment（约 0.8h）
     → 每 bin 约 120 个 segment
```

**不滑窗**：50% overlap 让相邻 segment 强相关，统计显著性计算会虚高。随机取不重叠 crop 更干净。

### 3.3 逐条 BRIR 合成流程

```
对每个 SOFA 文件 (10个):
  对每个 measurement M = 0..71 (720条 BRIR 总计):
    
    Step 1: 提取并降采样 BRIR
      brir_ch0 = IR[M, 0, 0, :]
      brir_ch1 = IR[M, 1, 0, :]
      brir_ch0_16k = resample_poly(brir_ch0, 16000, 44100)
      brir_ch1_16k = resample_poly(brir_ch1, 16000, 44100)
    
    Step 2: 确定标签
      az = wrap_deg(float(5 * M))
      label = angle_to_label(az, 72)
    
    Step 3: 卷积
      speech = load_mono(librispeech_file, 16000)
      channel0 = fftconvolve(speech, brir_ch0_16k)
      channel1 = fftconvolve(speech, brir_ch1_16k)
    
    Step 4: 固定 crop（不滑窗）
      从卷积结果中随机取 2 个不重叠的 2s 片段
      → 每 BRIR 固定 2 个 clean segment
    
    Step 5: 加 DEMAND 噪声（6 个条件）
      SNR = [clean, -10, -5, 0, +5, +10] dB
      对于 SNR ≠ "clean":
        同 scene 不同 channel，同时间片段
        全局归一化后按 SNR 混合
    
    Step 6: 保存
      WAV: binauralXXXXX_snrXX.wav
      metadata: azimuth_deg, label, M, sofa_file, listener_position, 
                ls_angle, snr_db, speech_path, brir_source
```

### 3.4 噪声方案（修正版）

| 属性 | 值 |
|---|---|
| 噪声源 | DEMAND（与训练一致） |
| 场景 | OOFFICE, PCAFETER, TMETRO, TBUS, SPSQUARE, NPARK |
| 左右耳 | **同 scene 不同 channel，同时间起始**（与训练一致） |
| 加噪时机 | BRIR 卷积 + 全局归一化后 |
| SNR 级别 | **clean, -10, -5, 0, +5, +10 dB** |
| clean 作用 | 拆分误差来源：clean→测混响+HRTF gap，noisy→测噪声鲁棒性 |

### 3.5 测试集规模

```
总 BRIR:      720 条 (10 文件 × 72 M)
Crop 策略:    固定 2 个不重叠 2s crop
Clean:        1,440 segments  (0.8h)
Per SNR:      1,440 segments  (0.8h)  × 5 noisy levels
总计:         1,440 × 6 = 8,640 segments  (~4.8h)
Per bin:      ~120 segments (6 SNR × 20)
评估时间:     ~20-30 分钟 (batch_size=64)
```

## 4. 目录结构与保存格式

对齐现有 `static_dataset` 结构：

```
data/librispeech_neidhardt_measured_brir_test_v2/
└── test_all/
    ├── binaural_dev/
    │   ├── binaural000001.wav
    │   ├── binaural000002.wav
    │   └── ...
    ├── metadata_dev/
    │   ├── metadata000001.json
    │   ├── metadata000002.json
    │   └── ...
    ├── mixing_report.csv
    └── manifest.json
```

Metadata 字段（每 segment）：
```json
{
  "azimuth_deg": 90.0,
  "azimuth_label": 54,
  "measurement_index": 54,
  "sofa_file": "Pos3_LS_0.sofa",
  "listener_position": 3,
  "ls_angle": 0,
  "snr_db": "clean",
  "speech_path": "/disk2/bywang/data/LibriSpeech/LibriSpeech_test/test-clean/...",
  "brir_source": "neidhardt_2019_zenodo_2593714",
  "t60_broadband_s": 0.63,
  "dummy_head": "KEMAR 45BA"
}
```

## 5. 评估指标

```
Per-segment:
  - MAE, Acc@5°, Acc@10°
  - Front/back error rate, Large error rate, Opposite error rate

分组报告:
  - Per-bin: 72 类 per-angle MAE 曲线
  - Per-SNR: 6 条 SNR-MAE 曲线 (clean, -10, -5, 0, +5, +10)
  - Per-position: 5 个听者位置的 MAE 对比
  - Per-LS: LS_0 vs LS_180 MAE 对比
```

## 6. 评估的模型

从当前静态 KEMAR 主实验中选择模型进行零样本评估，不需要重新训练。所有 checkpoint
只能根据原 KEMAR 仿真验证集选择；禁止根据 Neidhardt 结果挑选 checkpoint、seed、
解码方式或阈值。主线、对比模型和消融必须使用同一生成数据与同一评测实现。

## 7. 这个实验能回答的核心问题

| 问题 | 如何回答 |
|---|---|
| **仿真混响 → 真实混响的泛化** | 模型在 ISM 仿真上训练，在真实房间 BRIR 上测试 |
| **未见实测链路鲁棒性** | 仿真 MIT KEMAR/SofaMyRoom 训练，实测 KEMAR 45BA BRIR 测试 |
| **真实房间声学的 360° 定位** | 72 类 per-angle MAE 曲线 |
| **听者位置变化的鲁棒性** | 5 个位置的 per-position MAE 对比 |
| **SNR 对真实/仿真混响的不同影响** | clean/noisy SNR 曲线 vs 训练数据同 SNR 曲线 |
| **误差来源分解** | clean → 仿真到实测 BRIR 的域差异；noisy → 再叠加噪声影响 |

## 8. 与训练数据的关键差异（需在论文中明确报告）

| 差异 | 训练 | 测试 | 预期影响 |
|---|---|---|---|
| 混响类型 | SofaMyRoom 仿真 BRIR | **真实房间实测 BRIR** | 核心测试点 |
| 双耳系统 | MIT KEMAR 仿真链路 | KEMAR 45BA small ears 实测链路 | 测量链路/耳廓配置差异；不是跨多人头实验 |
| 噪声 | DEMAND，训练时加入 | DEMAND，测试时加入 | 一致 ✅ |
| SNR | Uniform[-10, 10] dB | {clean, -10, -5, 0, +5, +10} dB | 含 clean 用于误差分解 |
| 采样率 | 16kHz 原生 | 44.1kHz → 16kHz 降采样 | 轻微 |
| segment | 2s | 2s | 一致 ✅ |

## 9. 生成与验证 checklist

- [x] 跑侧向 label/channel sanity check（±90° 直达声能量验证）
- [x] 确认类别角映射固定为 `angle = -180 + 5 * class`，正确分类 MAE 为 0°
- [x] 确认 720 条语音从 test-clean 不重复选取
- [x] 确认同一基础样本的五档 noisy 条件复用同一 DEMAND 场景、通道和起点
- [x] 确认 metadata 格式兼容 `tools/evaluate_neidhardt_brir.py`
- [x] 生成后验证全部 8640 个 WAV，并独立抽查 5 个随机样本

正式 v2 已生成到 `data/librispeech_neidhardt_measured_brir_test_v2/test_all`。
生成器验证结果：72 类各 120 条、6 个 SNR 各 1440 条、5 个位置各
1728 条，两种扬声器朝向各 4320 条；标签网格最大偏差为 0°。
