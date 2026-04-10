"""门控模块，用于差异先验引导的注意力调制。

门控计算：
  G = Sigmoid(Linear(D_feat))

调制：
  A_LR' = G * A_LR
  A_RL' = G * A_RL

所有张量均为 ``[B, T, D]``。
"""

import torch
import torch.nn as nn


class GatingModule(nn.Module):
    """由差异先验 ``D_feat`` 驱动的逐元素门控。

    参数
    ----------
    prior_dim : int
        ``D_feat`` 的维度（门控输入维度）。
    gate_dim : int
        门控输出维度（须与注意力输出维度一致）。
    """

    def __init__(self, prior_dim: int = 128, gate_dim: int = 128):
        super().__init__()
        self.gate_fc = nn.Linear(prior_dim, gate_dim)

    def forward(
        self,
        d_feat: torch.Tensor,
        a_lr: torch.Tensor,
        a_rl: torch.Tensor,
    ):
        """
        参数:
            d_feat: 差异先验 ``[B, T, prior_dim]``
            a_lr:   左耳查询右耳的注意力输出 ``[B, T, gate_dim]``
            a_rl:   右耳查询左耳的注意力输出 ``[B, T, gate_dim]``

        返回:
            gated_a_lr: ``[B, T, gate_dim]``
            gated_a_rl: ``[B, T, gate_dim]``
        """
        # 输入 shape: d_feat [B, T, prior_dim]
        g = torch.sigmoid(self.gate_fc(d_feat))  # [B, T, gate_dim]

        gated_a_lr = g * a_lr   # [B, T, gate_dim]
        gated_a_rl = g * a_rl   # [B, T, gate_dim]

        return gated_a_lr, gated_a_rl
