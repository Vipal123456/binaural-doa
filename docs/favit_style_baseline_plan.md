# FAViT-Style Baseline 复现方案

## 1. 目标

本方案用于在当前静态双耳 DOA 任务上复现一个 **FAViT-style external deep baseline**，作为和当前主线模型：

- `dual cue value/reliability`
- `cf80_cue24_gru80`

进行公平对照的外部模型。

这里的“FAViT-style”指的是**保留论文的核心思想**：

1. 使用显式双耳 cue 作为输入
2. 用**沿频率方向切分的 patch**代替常规二维 patch
3. 通过轻量 Transformer 在频率优先的 token 序列上建模

而不是机械复刻原论文的全部超参数。  
原因是原论文任务和当前任务并不完全一致：

- 原论文：`180° / 37 classes / 5°`
- 当前任务：`360° / 72 classes / 5°`
- 当前任务还包含：
  - `unseen-subject`
  - `noise + reverb`
  - `unseen-noise`

因此需要做**协议适配后的公平复现**。

---

## 2. 对照目标

这条 baseline 主要回答：

> 如果采用一个频率优先、Transformer 风格、只使用显式 binaural cue 的外部深度学习结构，它在当前 72 类 / 360° 协议下能达到什么水平？

它希望和以下几条内部线形成有意义对比：

1. `content-only`
   - 验证显式 cue 是否必要
2. `early-fusion`
   - 验证单流显式 cue + 内容的效果
3. `cf80_cue24_gru80`
   - 验证轻量 cue encoder + GRU 的 compact 设计是否更划算
4. `dual cue value/reliability`
   - 验证显式 `value / reliability` 分解是否优于统一 cue Transformer 建模

---

## 3. 参考论文

### 论文

- Phokhinanan et al., **Binaural Sound Localization in Noisy Environments Using Frequency-Based Audio Vision Transformer**, Interspeech 2023
- 页面：<https://www.isca-archive.org/interspeech_2023/phokhinanan23_interspeech.html>
- PDF：<https://www.isca-archive.org/interspeech_2023/phokhinanan23_interspeech.pdf>

### 代码情况

作者提供了仓库链接：

- <https://github.com/Senzt/FAViT>

但当前仓库未提供可直接复现的训练代码，因此这里采用**论文复现式 baseline 方案**。

---

## 4. 当前任务下的复现原则

### 必须保持不变

为了和现有模型公平比较，以下条件必须保持一致：

- 数据集：
  - `train_subjects`
  - `val_subjects`
  - `test_subjects_unseen`
- 附加更严格测试：
  - `test_subjects_unseen_noiseheldout`
- 标签协议：
  - `72 classes`
  - `[-180°, 180°)`
  - `5°` 分箱
- 训练流程：
  - batch size、优化器、学习率调度尽量与现有主线统一
- 指标：
  - `Accuracy`
  - `F1-score`
  - `Acc@5°`
  - `Acc@10°`
  - `MAE`
  - `FB err`
  - `Opp err`
  - `Large err`

### 允许适配调整的部分

- 输入 cue 形式（`IPD` vs `sin/cos(IPD)`）
- patch 数和 embedding 维度
- Transformer 深度和头数
- 分类头结构

---

## 5. 输入特征设计

## 5.1 主版本建议：最贴近论文

### 输入

- `ILD`
- `IPD`

输入 tensor：

```text
X ∈ [B, 2, T, F]
```

### 为什么先用这个版本

因为这是最贴近论文原始设计的版本，作为外部 baseline 更有说服力。

---

## 5.2 备选版本（只在主版本过弱时再试）

### 输入

- `ILD`
- `sin(IPD)`
- `cos(IPD)`

输入 tensor：

```text
X ∈ [B, 3, T, F]
```

### 作用

这个版本不是第一优先，只在主版本明显受 `IPD` 周期性问题影响时再尝试。

---

## 6. 频率优先 patch 设计

这部分是 FAViT-style baseline 的核心。

### 关键思想

不采用标准 ViT 的方形 patch，而是沿频率轴切分 **vertical frequency patches**。

也就是说，每个 token 更像一个“频带片段”，而不是一个时频二维小块。

---

## 6.1 当前特征尺寸背景

按照现有特征参数，大致会得到：

- `T ≈ 201`
- `F ≈ 257`

如果直接在这个尺寸上跑 Transformer，token 数过多，不适合做轻量 baseline。

因此建议先做**时间维压缩**。

---

## 6.2 建议的最小复现 patch 流程

### Step 1: 时间维压缩

将输入时间维从 `T ≈ 201` 压到：

- `T' = 16`

实现方式可以后续选择：

- average pooling over time
- strided Conv2d on time axis
- adaptive pooling to fixed `T'`

### Step 2: 保留频率维

保持：

- `F = 257`

### Step 3: 频率分块

沿频率轴切成：

- `M = 16` 个 vertical patches

这样每个 patch 覆盖：

- 全部 `T' = 16` 个时间步
- 大约 `257 / 16 ≈ 16` 个频率 bin

最终：

```text
[B, C, 16, 257]
-> 16 个 frequency-oriented patches
-> 16 个 tokens
```

---

## 6.3 推荐 patch 设置

### 第一版固定方案

- `T' = 16`
- `num_patches = 16`

这套设置原因：

1. token 数足够小
2. 仍保留完整频率覆盖
3. 和论文“频率优先 token 化”的思想一致
4. 参数量容易控制

---

## 7. Patch embedding 设计

每个 patch 先 flatten，再线性映射到 embedding 维度：

```text
patch_i
-> flatten
-> Linear(patch_dim -> D)
```

### 推荐第一版 embedding 维度

- `D = 64`

### 为什么不是照抄论文的 `D=20`

因为当前任务明显更难：

- 360° 而不是 180°
- 72 类而不是 37 类
- 有 `unseen-subject`
- 有 `reverb`
- 有 `unseen-noise`

如果完全照抄 `D=20`，很可能 baseline 过弱，不利于公平比较。

因此建议保留“轻量 Transformer”思想，但适度提高容量。

---

## 8. Transformer encoder 设计

## 8.1 推荐第一版设置

- `depth = 6`
- `num_heads = 4`
- `embed_dim = 64`
- `mlp_ratio = 4`
- `dropout = 0.1`

### 说明

这比原论文略大，但仍然是一个轻量 Transformer baseline。

---

## 8.2 token 处理方式

建议采用标准 ViT 风格：

- prepend `cls token`
- add learned position embedding
- pass through transformer blocks

最终使用：

- `cls token`
  或
- 所有 tokens 的 mean pooling

作为整段表示。

### 第一版建议

先用：

- `cls token`

这样更贴近原始 ViT 叙事。

---

## 9. 输出头设计

### 当前任务必须改成

- `72-class classifier`

建议分类头：

```text
Linear(64 -> 256)
ReLU
Dropout
Linear(256 -> 72)
```

这就够了，不需要太复杂。

---

## 10. 模型整体结构建议

第一版可以写成：

```text
[ILD, IPD]                     # [B, 2, T, F]
-> time compression           # [B, 2, 16, F]
-> frequency-wise patch split
-> patch embedding            # [B, 16, 64]
-> add cls token + pos embed
-> Transformer encoder x 6
-> cls token
-> MLP head
-> 72-class logits
```

---

## 11. 训练协议

## 11.1 数据协议

使用当前完全相同的数据协议：

- train: `train_subjects`
- val: `val_subjects`
- test: `test_subjects_unseen`
- extra test: `test_subjects_unseen_noiseheldout`

## 11.2 优化器和调度器

尽量沿用现有主线，方便公平比较：

- optimizer: `AdamW`
- lr: 与当前主线相同量级
- scheduler: cosine
- label smoothing: 可沿用当前设置

## 11.3 批大小

先保持和现有主线一致：

- `batch_size = 64`

如果显存不够，再下调。

---

## 12. 评估指标

必须与当前主线完全一致：

- `Accuracy`
- `F1-score`
- `Acc@5°`
- `Acc@10°`
- `MAE`
- `front_back_halfplane_error_rate`
- `opposite_error_rate`
- `large_error_rate`

这点非常关键，因为你是要把它放进同一张主结果表里。

---

## 13. 论文中的角色定位

这条模型不应该被写成“我们的方法”，而应该明确写成：

## 外部深度学习对照

> A FAViT-style external baseline using explicit binaural cues and frequency-oriented transformer encoding.

它回答的是：

> 用频率优先的 Transformer 直接建模显式双耳 cue，和当前 dual-cue / cf80 主线相比怎么样？

---

## 14. 和当前主线的对照点

这条 baseline 和你当前主线的区别很清楚：

### FAViT-style

- 输入是显式 cue
- 不显式建模内容流
- 不显式区分 cue value / reliability
- Transformer 在 cue tokens 上做统一建模

### dual cue

- 有内容流
- 有 cue 流
- 显式区分 value / reliability
- 用轻量 GRU 只做聚合

### cf80

- 有内容流
- 有单分支 lite cue encoder
- 结构更紧凑

这样对照就很干净。

---

## 15. 风险点

## 风险 1：完全照论文超小模型会过弱

### 应对

embedding 维度不要太小，建议 `64` 起步。

## 风险 2：只用 `IPD` 可能不稳定

### 应对

先做最贴论文的 `ILD + IPD` 版本；若效果异常弱，再补 `ILD + sin/cos(IPD)` 版本。

## 风险 3：Transformer token 数过大

### 应对

先压缩时间维，再沿频率切 patch。

---

## 16. 推荐实施顺序

### 第一阶段（最小可行版本）

1. 输入：`ILD + IPD`
2. `T -> 16`
3. `16` 个频率 patch
4. `D=64, depth=6, heads=4`
5. `72-class head`

### 第二阶段（仅在必要时）

如果第一版明显过弱，再考虑：

1. `ILD + sin(IPD) + cos(IPD)`
2. `embed_dim = 96`
3. `depth = 8`

但不建议一开始就这么做。

---

## 17. 最终建议

当前最推荐的 FAViT-style baseline 复现方案是：

- **输入**：`ILD + IPD`
- **token 化**：`time compression + frequency-wise vertical patches`
- **Transformer**：`embed_dim=64, depth=6, heads=4`
- **输出**：`72-class classifier`
- **协议**：完全沿用当前静态 unseen-subject / unseen-noise 评估

这是一个：

- 足够贴近原论文思想
- 又能公平适配你当前任务
- 并且工作量可控

的外部深度学习 baseline 方案。

---

## 18. 下一步

如果后续决定实现，建议按下面顺序推进：

1. 新建 `favit_style` 模型类
2. 只实现主版本 `ILD + IPD`
3. 跑 `test_subjects_unseen`
4. 再跑 `test_subjects_unseen_noiseheldout`
5. 最后和：
   - `content-only`
   - `early-fusion`
   - `cf80`
   - `dual cue`
   放进同一张表
