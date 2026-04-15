"""双向独立残差门控模块。

与早期“共享门控直接乘注意力”不同，本模块为左右两个方向分别学习门控：

    g_lr = sigmoid(MLP_lr([d_feat, a_lr, f_l]))
    g_rl = sigmoid(MLP_rl([d_feat, a_rl, f_r]))

并采用残差式调制：

    a_lr' = f_l + g_lr * a_lr
    a_rl' = f_r + g_rl * a_rl

所有张量均为 ``[B, T, D]``。
"""

import torch
import torch.nn as nn


class GatingModule(nn.Module):
    """双向独立残差门控。

    参数
    ----------
    prior_dim : int
        ``D_feat`` 的维度（门控输入维度）。
    gate_dim : int
        门控输出维度（须与注意力输出维度一致）。
    """

    def __init__(self, prior_dim: int = 128, gate_dim: int = 128):
        super().__init__()
        in_dim = prior_dim + gate_dim + gate_dim
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
        # 左查右门控（独立）
        lr_in = torch.cat([d_feat, a_lr, f_l], dim=-1)
        g_lr = torch.sigmoid(self.gate_lr(lr_in))

        # 右查左门控（独立）
        rl_in = torch.cat([d_feat, a_rl, f_r], dim=-1)
        g_rl = torch.sigmoid(self.gate_rl(rl_in))

        # 残差门控
        gated_a_lr = f_l + g_lr * a_lr
        gated_a_rl = f_r + g_rl * a_rl

        return gated_a_lr, gated_a_rl
