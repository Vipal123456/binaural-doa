"""时序建模（BiGRU）+ 分类头。

接收融合后的特征序列，输出逐帧或逐片段的方位角 logits。

输入 shape:  ``[B, T, D_fused]``
输出 shape:  ``[B, num_classes]``  （时间均值池化后的片段级预测）
"""

import torch
import torch.nn as nn


class TemporalHead(nn.Module):
    """BiGRU 时序编码器 + 线性分类头 + 回归头。

    参数
    ----------
    input_dim : int
        每个时间步的融合特征向量维度。
    gru_hidden_size : int
        每个 GRU 方向的隐藏层大小。
    gru_num_layers : int
        堆叠的 GRU 层数。
    num_classes : int
        方位角 bin 数量。
    gru_dropout : float
        GRU 层间 Dropout（仅在 ``gru_num_layers > 1`` 时生效）。
    dropout : float
        分类头前的 Dropout。
    use_regression : bool
        是否使用回归头输出连续角度值。
    """

    def __init__(
        self,
        input_dim: int,
        gru_hidden_size: int = 128,
        gru_num_layers: int = 2,
        num_classes: int = 72,
        gru_dropout: float = 0.1,
        dropout: float = 0.2,
        use_regression: bool = False,
    ):
        super().__init__()
        self.use_regression = use_regression

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout if gru_num_layers > 1 else 0.0,
        )

        # BiGRU 输出维度 = 2 * hidden_size
        gru_out_dim = 2 * gru_hidden_size

        self.dropout = nn.Dropout(dropout)

        # 分类头：输出离散的方位角类别
        self.classifier = nn.Linear(gru_out_dim, num_classes)

        # 回归头：输出连续的方位角值（弧度，范围 [-π, π]）
        if use_regression:
            self.regressor = nn.Sequential(
                nn.Linear(gru_out_dim, gru_out_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(gru_out_dim // 2, 1),
                nn.Tanh()  # 输出范围 [-1, 1]，需要乘以π得到弧度
            )

    def forward(self, x: torch.Tensor) -> dict:
        """
        参数:
            x: 融合特征 ``[B, T, D_fused]``。

        返回:
            dict: 包含以下键的字典：
              - ``"logits"``: ``[B, num_classes]`` — 片段级方位角分类 logits
              - ``"angle"``: ``[B]`` — 片段级回归角度（弧度，范围 [-π, π]），仅在 use_regression=True 时存在
        """
        # 输入 shape: [B, T, D_fused]
        gru_out, _ = self.gru(x)          # [B, T, 2*H]
        # 对时间维进行均值池化，得到片段级预测
        pooled = gru_out.mean(dim=1)      # [B, 2*H]
        pooled = self.dropout(pooled)     # [B, 2*H]

        # 分类输出
        logits = self.classifier(pooled)  # [B, num_classes]

        result = {"logits": logits}

        # 回归输出（如果启用）
        if self.use_regression:
            angle_normalized = self.regressor(pooled)  # [B, 1], 范围 [-1, 1]
            angle_rad = angle_normalized.squeeze(-1) * 3.14159265359  # [B], 范围 [-π, π]
            result["angle"] = angle_rad

        return result
