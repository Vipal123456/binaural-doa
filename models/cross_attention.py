"""双向交叉注意力模块（支持score bias注入）。

实现两个方向的多头交叉注意力：
  - 左耳查询右耳  → A_LR
  - 右耳查询左耳  → A_RL

可选注入来自 DifferencePrior 的注意力分数偏置：
  score = QK^T / sqrt(d) + bias

输入 / 输出 shape：均为 ``[B, T, D]``。
"""

import math

import torch
import torch.nn as nn


class _CrossAttentionWithBias(nn.Module):
    """单方向交叉注意力，支持可选 score bias。"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim({embed_dim}) must be divisible by num_heads({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> [B, H, T, Dh]
        bsz, t, _ = x.shape
        x = x.view(bsz, t, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, Dh] -> [B, T, D]
        bsz, _, t, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(bsz, t, self.embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_bias: torch.Tensor = None,
    ) -> torch.Tensor:
        # query/key_value: [B, T, D]
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key_value))
        v = self._split_heads(self.v_proj(key_value))

        # [B, H, Tq, Tk]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_bias is not None:
            # 支持 [B, T, T] 或 [B, H, T, T]
            if attn_bias.dim() == 3:
                attn_bias = attn_bias.unsqueeze(1)
            scores = scores + attn_bias

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)  # [B, H, T, Dh]
        out = self._merge_heads(out)  # [B, T, D]
        out = self.out_proj(out)
        return out


class BidirectionalCrossAttention(nn.Module):
    """双向交叉注意力（可选双向score bias）。"""

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn_lr = _CrossAttentionWithBias(embed_dim, num_heads, dropout)
        self.attn_rl = _CrossAttentionWithBias(embed_dim, num_heads, dropout)
        self.norm_lr = nn.LayerNorm(embed_dim)
        self.norm_rl = nn.LayerNorm(embed_dim)

    def forward(
        self,
        f_l: torch.Tensor,
        f_r: torch.Tensor,
        bias_lr: torch.Tensor = None,
        bias_rl: torch.Tensor = None,
    ):
        """
        参数:
            f_l: 左耳特征 ``[B, T, D]``
            f_r: 右耳特征 ``[B, T, D]``
            bias_lr: 左查右 score bias，可选 ``[B, T, T]`` 或 ``[B, H, T, T]``
            bias_rl: 右查左 score bias，可选 ``[B, T, T]`` 或 ``[B, H, T, T]``

        返回:
            a_lr: 左耳查询右耳注意力输出 ``[B, T, D]``
            a_rl: 右耳查询左耳注意力输出 ``[B, T, D]``
        """
        a_lr = self.attn_lr(query=f_l, key_value=f_r, attn_bias=bias_lr)
        a_lr = self.norm_lr(a_lr + f_l)

        a_rl = self.attn_rl(query=f_r, key_value=f_l, attn_bias=bias_rl)
        a_rl = self.norm_rl(a_rl + f_r)

        return a_lr, a_rl
