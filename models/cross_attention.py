"""双向交叉注意力模块。

实现两个方向的多头交叉注意力：
  - 左耳查询右耳  → A_LR   （左耳关注右耳信息）
  - 右耳查询左耳  → A_RL   （右耳关注左耳信息）

输入 / 输出 shape：均为 ``[B, T, D]``。
"""

import torch
import torch.nn as nn


class BidirectionalCrossAttention(nn.Module):
    """将两个方向的交叉注意力封装在一个模块中。

    参数
    ----------
    embed_dim : int
        输入特征维度（须与编码器输出维度一致）。
    num_heads : int
        注意力头数。
    dropout : float
        注意力 Dropout 概率。
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # 左耳查询右耳
        self.attn_lr = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # 右耳查询左耳
        self.attn_rl = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_lr = nn.LayerNorm(embed_dim)
        self.norm_rl = nn.LayerNorm(embed_dim)

    def forward(
        self,
        f_l: torch.Tensor,
        f_r: torch.Tensor,
    ):
        """
        参数:
            f_l: 左耳特征  ``[B, T, D]``
            f_r: 右耳特征  ``[B, T, D]``

        返回:
            a_lr: 左耳查询右耳的注意力输出 ``[B, T, D]``
            a_rl: 右耳查询左耳的注意力输出 ``[B, T, D]``
        """
        # 左耳关注右耳: Q=f_l, K=V=f_r
        a_lr, _ = self.attn_lr(query=f_l, key=f_r, value=f_r)  # [B, T, D]
        a_lr = self.norm_lr(a_lr + f_l)  # 残差连接 + 层归一化

        # 右耳关注左耳: Q=f_r, K=V=f_l
        a_rl, _ = self.attn_rl(query=f_r, key=f_l, value=f_l)  # [B, T, D]
        a_rl = self.norm_rl(a_rl + f_r)  # 残差连接 + 层归一化

        return a_lr, a_rl
