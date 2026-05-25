# BiL-Style Baseline 复现方案

## 1. 目标

本方案用于在当前静态双耳 DOA 任务上复现一个 **BiL-style external lightweight baseline**，作为和当前主线模型：

- `dual cue value/reliability`
- `cf80_cue24_gru80`
- `early-fusion`

进行公平对照的外部模型。

这里的 “BiL-style” 指的是保留论文 **Binaural Localization Model for Speech in Noise** 的核心思路：

1. 使用 **GCC-PHAT** 作为显式双耳输入特征
2. 使用 **轻量卷积前端 + GRU** 的 CRN 结构
3. 将模型作为一个语音、噪声、混响条件下的轻量双耳定位网络

但不机械照搬其原始任务协议。原因是原论文和当前任务差异较大：

- 原论文：`frontal plane (-90° ~ +90°)`
- 原论文：更偏连续方向向量输出
- 当前任务：`360° / 72 classes / 5°`
- 当前任务还包含：
  - `unseen-subject`
  - `noise + reverb`
  - `unseen-noise`

因此需要做 **协议适配后的公平复现**。

---

## 2. 参考论文与代码

### 论文

- Vikas Tokala et al., **Binaural Localization Model for Speech in Noise**, 2025
- PDF：<https://uol.de/f/6/dept/mediphysik/ag/sigproc/download/papers/SP2024_23.pdf>
- arXiv：<https://arxiv.org/abs/2507.20027>

### 开源代码

- GitHub：<https://github.com/VikasTokala/BiL>

这篇和 FAViT 不同，作者确实放出了代码仓库。仓库里可见的核心文件包括：

- `models/`
- `binaural_dataset.py`
- `train_binaural.py`
- `test.py`
- `trainer.py`
- `loss.py`
- `metrics.py`
- `params.json`

---

## 3. 对照目标

这条 baseline 主要回答：

> 如果采用一个以 GCC-PHAT 为核心输入、采用轻量 CRN + GRU 的外部双耳定位结构，它在当前 72 类 / 360° / unseen-subject / unseen-noise 协议下能达到什么水平？

它希望和以下内部线形成有意义对比：

1. `content-only`
   - 验证显式双耳 cue 是否必要
2. `early-fusion`
   - 验证单流显式 cue 建模的水平
3. `cf80_cue24_gru80`
   - 验证轻量 cue encoder + GRU 是否比 GCC-PHAT + CRN 更划算
4. `dual cue value/reliability`
   - 验证显式 `value / reliability` 分解是否优于统一 GCC-PHAT 驱动的 CRN

---

## 4. 当前任务下的复现原则

### 必须保持不变

为了和现有主线公平比较，以下条件必须保持一致：

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
- 训练协议：
  - batch size 尽量与现有主线统一
  - optimizer / scheduler 尽量与现有主线统一
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

- 输出头：由原始方向向量回归改成 `72-class classification`
- GRU hidden size
- 卷积通道数
- 是否引入 front/back auxiliary
- 是否加入 internal ear noise 模拟

---

## 5. 输入特征设计

### 5.1 主版本：GCC-PHAT

根据论文，原始模型的核心输入特征是：

- `GCC-PHAT`

因此主复现版本建议保持这一点，不要第一版就改成 ILD/IPD。

### 输入张量建议

将 GCC-PHAT 组织成：

```text
X ∈ [B, 1, T, F_gcc]
```

其中：
- `T`：时间帧数
- `F_gcc`：GCC 特征维度（可等于 STFT 频点数，或由实现决定）

### 为什么主版本先用 GCC-PHAT

因为这是这篇 BiL-style baseline 与你现有双耳线最根本的区别：

- 你的主线：显式 `ILD / IPD / coherence`
- BiL-style：`GCC-PHAT + CRN`

如果输入也改成 ILD/IPD，就丢掉了它作为外部风格 baseline 的意义。

---

## 6. 网络骨架设计

## 6.1 论文核心结构

按论文描述，其网络由：

1. 一组卷积块
2. flatten 频率和通道维
3. GRU
4. 线性输出层

构成。

论文中还提到：

- 所有卷积 kernel 使用 `(3, 3)`
- 前两层有 max pooling
- 最后一层不池化
- 卷积部分激活函数用 `PReLU`

---

## 6.2 当前任务下的推荐最小复现结构

建议第一版用一个轻量但稳定的版本：

### 卷积前端

1. `ConvBlock1`
   - `1 -> 32`
   - `kernel = 3x3`
   - `PReLU`
   - `MaxPool`

2. `ConvBlock2`
   - `32 -> 64`
   - `kernel = 3x3`
   - `PReLU`
   - `MaxPool`

3. `ConvBlock3`
   - `64 -> 96`
   - `kernel = 3x3`
   - `PReLU`
   - 无池化

### 时序头

- `BiGRU`
- `hidden_size = 96`
- `num_layers = 1`

### 分类头

由于你当前任务是 72 类分类，建议改成：

```text
GRU output
-> pooled feature
-> MLP(192 -> 128 -> 72)
```

如果是双向 GRU，`192 = 96 x 2`。

---

## 6.3 输出形式

### 不建议照搬原始 2D vector 输出

原论文使用 2D 方向向量回归，并采用 cosine-style loss，而且特意弱化了 front-back ambiguity 的惩罚。

这和你当前 360°、72 类、需要显式区分 front/back 的任务并不一致。

### 建议主版本输出

- **72-class softmax**

这样能直接和：
- `dual cue`
- `cf80`
- `early-fusion`
- `content-only`

同表公平比较。

---

## 7. 是否引入 internal ear noise

### 结论

**第一版先不做。**

### 原因

1. 这是原论文里偏 psychoacoustic 的附加建模
2. 你当前最重要的是先做一个结构风格对照
3. 如果一开始把 internal ear noise 也带进来，变量太多

### 推荐顺序

#### 第一版
- `GCC-PHAT + CRN + GRU + 72类`

#### 第二版（只有第一版值得继续时再做）
- 再加入 internal ear noise 模拟

---

## 8. 推荐复现名称

建议模型命名为：

**`bil_style_gccphat_crn_72cls`**

这个名字能很清楚地表达：

- `bil_style`
- `gccphat`
- `crn`
- `72cls`

---

## 9. 训练协议建议

### 训练 / 验证 / 测试

统一沿用你当前静态主线：

- `train_subjects`
- `val_subjects`
- `test_subjects_unseen`
- `test_subjects_unseen_noiseheldout`

### 优化器与调度器

第一版建议尽量贴近你当前主线，减少协议差异：

- `Adam`
- cosine scheduler
- batch size `64`
- `num_workers = 8`

### loss

第一版直接用：

- `CrossEntropy`

如果后面发现 front/back 错误特别高，再考虑是否补 `front_back_auxiliary` 版本。

---

## 10. 和现有主线的公平对比方式

这条 baseline 最终要回答的是：

### 与 `dual cue`
- 明确建模 `value / reliability` 是否优于统一 GCC-PHAT CRN

### 与 `cf80`
- 轻量 cue encoder + GRU 是否比经典 GCC-PHAT CRN 更划算

### 与 `early-fusion`
- 单流显式 cue 内容融合 vs GCC-PHAT CRN 谁更强

### 与 `content-only`
- 显式双耳 cue 的必要性是否再次得到验证

---

## 11. 你实际需要怎么做

### 11.1 不建议的做法

不要直接把 `BiL` 仓库完整并进当前工程作为一个子项目跑。

原因：
- 数据协议不同
- 输出形式不同
- 训练脚本不同
- 很容易把当前项目的实验口径弄乱

### 11.2 推荐做法

可以把作者仓库 **clone 到临时目录做参考**，但不要直接集成。

真正需要重点阅读的源码部分：

1. `models/`
2. `params.json`
3. `loss.py`
4. `binaural_dataset.py`
5. `train_binaural.py`

### 你真正要做的事

不是“运行它的项目”，而是：

> 在你自己的训练/测试框架中，实现一个 **BiL-style baseline**

这样才能保证：
- 数据一致
- 指标一致
- 主表公平

---

## 12. 推荐的实施顺序

### 第一步
阅读并提炼以下内容：

- `models/`
- `params.json`
- `loss.py`
- `binaural_dataset.py`

### 第二步
在当前工程里实现：

- `bil_style_gccphat_crn_72cls`

### 第三步
训练并评估：

1. `test_subjects_unseen`
2. `test_subjects_unseen_noiseheldout`

### 第四步
与以下模型做主表对比：

- `dual cue value/reliability`
- `cf80_cue24_gru80`
- `early-fusion`
- `content-only`

---

## 13. 风险与预期

### 预期优势

- 作为外部轻量深度 baseline，非常贴题
- 结构风格与你当前轻量主线相近
- GCC-PHAT 路线是一个合理外部对照

### 风险

- 论文原始 frontal-plane 设置可能让它在 360° 任务上不天然占优
- 如果完全去掉原始 vector loss，它的人类感知建模优势会变弱
- 需要自己实现或对齐 GCC-PHAT 特征管线

---

## 14. 最终建议

如果只从静态论文补外部对照的价值来看，这条 BiL-style baseline **值得做**，而且它比很多更大更花的模型更贴合你当前任务。

推荐策略是：

1. 保留论文的 **GCC-PHAT + CRN + GRU** 核心
2. 改成你的 **72-class / 360°** 评估协议
3. 暂不引入 internal ear noise
4. 在统一数据协议下和 `dual cue / cf80 / early-fusion / content-only` 做公平比较

这样它会成为一个很有说服力的外部轻量深度 baseline。
