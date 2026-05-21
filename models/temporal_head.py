"""时序建模（BiGRU / BiLSTM）+ Attention Pooling + 分类/回归头。

接收融合后的特征序列，输出逐帧或逐片段的方位角 logits。

输入 shape:  ``[B, T, D_fused]``
输出 shape:  ``[B, num_classes]``  （时间池化后的片段级预测）
"""

import torch
import torch.nn as nn
import math

try:
    from mambapy.mamba import Mamba, MambaConfig
except Exception:  # pragma: no cover - optional dependency
    Mamba = None
    MambaConfig = None


class TemporalHead(nn.Module):
    """BiGRU / BiLSTM 时序编码器 + 线性分类头 + 回归头。

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
    use_attention_pooling : bool
        是否使用注意力池化（否则退化为时间均值池化）。
    """

    def __init__(
        self,
        input_dim: int,
        gru_hidden_size: int = 128,
        gru_num_layers: int = 2,
        temporal_encoder_type: str = "gru",
        mamba_num_layers: int = 2,
        mamba_state_dim: int = 16,
        mamba_expand_factor: int = 2,
        mamba_conv_kernel: int = 4,
        num_classes: int = 72,
        gru_dropout: float = 0.1,
        dropout: float = 0.2,
        use_regression: bool = False,
        use_pure_regression: bool = False,
        use_attention_pooling: bool = True,
        use_front_back_auxiliary: bool = False,
        azimuth_range = (-180.0, 180.0),
    ):
        super().__init__()
        self.use_regression = use_regression
        self.use_pure_regression = use_pure_regression
        self.use_attention_pooling = use_attention_pooling
        self.use_front_back_auxiliary = use_front_back_auxiliary
        self.num_classes = num_classes
        self.azimuth_range = tuple(azimuth_range)
        self.temporal_encoder_type = temporal_encoder_type

        rnn_dropout = gru_dropout if gru_num_layers > 1 else 0.0
        if temporal_encoder_type == "gru":
            self.temporal_encoder = nn.GRU(
                input_size=input_dim,
                hidden_size=gru_hidden_size,
                num_layers=gru_num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=rnn_dropout,
            )
            temporal_out_dim = 2 * gru_hidden_size
        elif temporal_encoder_type == "lstm":
            self.temporal_encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=gru_hidden_size,
                num_layers=gru_num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=rnn_dropout,
            )
            temporal_out_dim = 2 * gru_hidden_size
        elif temporal_encoder_type == "mamba":
            if Mamba is None or MambaConfig is None:
                raise ImportError(
                    "temporal_encoder_type='mamba' requires mambapy to be installed."
                )
            self.temporal_encoder = Mamba(
                MambaConfig(
                    d_model=input_dim,
                    n_layers=mamba_num_layers,
                    d_state=mamba_state_dim,
                    expand_factor=mamba_expand_factor,
                    d_conv=mamba_conv_kernel,
                    use_cuda=False,
                )
            )
            temporal_out_dim = input_dim
        else:
            raise ValueError(f"Unsupported temporal_encoder_type: {temporal_encoder_type}")

        self.dropout = nn.Dropout(dropout)

        # Attention Pooling: 对时间维进行可学习加权聚合
        if self.use_attention_pooling:
            attn_hidden = max(temporal_out_dim // 2, 32)
            self.attn_pool = nn.Sequential(
                nn.Linear(temporal_out_dim, attn_hidden),
                nn.Tanh(),
                nn.Linear(attn_hidden, 1),
            )

        # 分类头：输出离散的方位角类别
        if not self.use_pure_regression:
            self.classifier = nn.Linear(temporal_out_dim, num_classes)

        # front/back 辅助头：显式学习前后判别
        if self.use_front_back_auxiliary:
            self.front_back_classifier = nn.Linear(temporal_out_dim, 2)

        # 回归头：输出连续的方位角值（弧度，范围 [-π, π]）
        if use_regression and not self.use_pure_regression:
            self.regressor = nn.Sequential(
                nn.Linear(temporal_out_dim, temporal_out_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(temporal_out_dim // 2, 1),
                nn.Tanh()  # 输出范围 [-1, 1]，需要乘以π得到弧度
            )

        if self.use_pure_regression:
            self.vector_regressor = nn.Sequential(
                nn.Linear(temporal_out_dim, temporal_out_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(temporal_out_dim // 2, 2),
            )

            centers_deg = self.azimuth_range[0] + (torch.arange(num_classes).float() + 0.5) * (
                (self.azimuth_range[1] - self.azimuth_range[0]) / num_classes
            )
            self.register_buffer("bin_centers_deg", centers_deg, persistent=False)

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
        if self.temporal_encoder_type == "mamba":
            temporal_out = self.temporal_encoder(x)  # [B, T, D]
        else:
            temporal_out, _ = self.temporal_encoder(x)  # [B, T, 2*H]
        # 对时间维进行池化，得到片段级预测
        if self.use_attention_pooling:
            attn_logits = self.attn_pool(temporal_out)     # [B, T, 1]
            attn_weights = torch.softmax(attn_logits, dim=1)
            pooled = (attn_weights * temporal_out).sum(dim=1)
        else:
            pooled = temporal_out.mean(dim=1)
        pooled = self.dropout(pooled)

        if self.use_pure_regression:
            angle_vec = self.vector_regressor(pooled)  # [B, 2]
            angle_vec = nn.functional.normalize(angle_vec, dim=-1)
            pred_deg = torch.rad2deg(torch.atan2(angle_vec[:, 0], angle_vec[:, 1]))  # [B]

            diff = pred_deg.unsqueeze(-1) - self.bin_centers_deg.unsqueeze(0)  # [B, C]
            diff = torch.remainder(diff + 180.0, 360.0) - 180.0
            logits = -torch.abs(diff) / 5.0

            result = {
                "logits": logits,
                "angle_vec": angle_vec,
            }
        else:
            # 分类输出
            logits = self.classifier(pooled)  # [B, num_classes]
            result = {"logits": logits}

        if self.use_front_back_auxiliary:
            result["front_back_logits"] = self.front_back_classifier(pooled)

        # 回归输出（如果启用）
        if self.use_regression and not self.use_pure_regression:
            angle_normalized = self.regressor(pooled)  # [B, 1], 范围 [-1, 1]
            angle_rad = angle_normalized.squeeze(-1) * 3.14159265359  # [B], 范围 [-π, π]
            result["angle"] = angle_rad

        return result
