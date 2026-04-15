"""双向门控模块。

支持两种消融形态：

1. 共享门控：左右方向复用同一组参数。
2. 独立门控：左右方向分别学习门控。

同时支持两种输出方式：

1. 残差式：f + g * a
2. 非残差式：g * a

所有张量均为 ``[B, T, D]``。
"""

import torch
import torch.nn as nn


class GatingModule(nn.Module):
    """双向门控。

    参数
    ----------
    prior_dim : int
        ``D_feat`` 的维度（门控输入维度）。
    gate_dim : int
        门控输出维度（须与注意力输出维度一致）。
    use_independent_gating : bool
        是否为左右方向分别学习门控。
    use_residual_gating : bool
        是否使用残差式调制 ``f + g * a``。
    """

    def __init__(
        self,
        prior_dim: int = 128,
        gate_dim: int = 128,
        use_independent_gating: bool = True,
        use_residual_gating: bool = True,
    ):
        super().__init__()
        self.use_independent_gating = use_independent_gating
        self.use_residual_gating = use_residual_gating
        in_dim = prior_dim + gate_dim + gate_dim

        self.gate_shared = nn.Sequential(
            nn.Linear(in_dim, gate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gate_dim, gate_dim),
        )

        if self.use_independent_gating:
            self.gate_lr = nn.Sequential(
                nn.Linear(in_dim, gate_dim),
                nn.ReLU(inplace=True),
                nn.Linear(gate_dim, gate_dim),
            )
            self.gate_rl = nn.Sequential(
                nn.Linear(in_dim, gate_dim),
                nn.ReLU(inplace=True),
                nn.Linear(gate_dim, gate_dim),
            )
        else:
            self.gate_lr = self.gate_shared
            self.gate_rl = self.gate_shared

    def forward(
        self,
        d_feat: torch.Tensor,
        a_lr: torch.Tensor,
        a_rl: torch.Tensor,
        f_l: torch.Tensor,
        f_r: torch.Tensor,
    ):
        """
        参数:
            d_feat: 差异先验 ``[B, T, prior_dim]``
            a_lr:   左耳查询右耳的注意力输出 ``[B, T, gate_dim]``
            a_rl:   右耳查询左耳的注意力输出 ``[B, T, gate_dim]``
            f_l:    左耳编码特征 ``[B, T, gate_dim]``
            f_r:    右耳编码特征 ``[B, T, gate_dim]``

        返回:
            gated_a_lr: ``[B, T, gate_dim]``
            gated_a_rl: ``[B, T, gate_dim]``
        """
        # 左查右门控
        lr_in = torch.cat([d_feat, a_lr, f_l], dim=-1)
        g_lr = torch.sigmoid(self.gate_lr(lr_in))

        # 右查左门控
        rl_in = torch.cat([d_feat, a_rl, f_r], dim=-1)
        g_rl = torch.sigmoid(self.gate_rl(rl_in))

        # 输出形式
        if self.use_residual_gating:
            gated_a_lr = f_l + g_lr * a_lr
            gated_a_rl = f_r + g_rl * a_rl
        else:
            gated_a_lr = g_lr * a_lr
            gated_a_rl = g_rl * a_rl

        return gated_a_lr, gated_a_rl
