"""差异先验模块。

由以下 4 项构造差异先验特征 ``D_feat``：
  1. F_L - F_R          （有符号差异）
  2. |F_L - F_R|        （绝对差异）
  3. IPD_proj            （投影后的 IPD）
  4. ILD_proj            （投影后的 ILD）

四项拼接后通过小型 MLP 生成 ``D_feat``。

张量 shape（均为 ``[B, T, ·]``）：
  - F_L, F_R      : ``[B, T, D_enc]``   （编码器输出）
  - IPD, ILD 原始 : ``[B, T, F_bins]``   （来自特征提取器）
  - IPD_proj      : ``[B, T, proj_dim]`` （投影后）
  - ILD_proj      : ``[B, T, proj_dim]``
  - D_feat        : ``[B, T, prior_out_dim]``
"""

import torch
import torch.nn as nn


class IPDILDProjection(nn.Module):
    """将原始 IPD / ILD（频率 bin 维度）投影到与编码器输出匹配的低维空间。

    IPD 和 ILD 各有独立的线性投影层。

    参数
    ----------
    freq_bins : int
        频率 bin 数量（= n_fft // 2 + 1）。
    proj_dim : int
        目标投影维度。
    """

    def __init__(self, freq_bins: int, proj_dim: int):
        super().__init__()
        self.ipd_proj = nn.Linear(freq_bins, proj_dim)
        self.ild_proj = nn.Linear(freq_bins, proj_dim)

    def forward(self, ipd: torch.Tensor, ild: torch.Tensor):
        """
        参数:
            ipd: ``[B, T, F]``
            ild: ``[B, T, F]``

        返回:
            ipd_proj: ``[B, T, proj_dim]``
            ild_proj: ``[B, T, proj_dim]``
        """
        return self.ipd_proj(ipd), self.ild_proj(ild)


class DifferencePrior(nn.Module):
    """构建差异先验特征 ``D_feat``。

    将四项拼接后通过 MLP：
      concat(F_L - F_R, |F_L - F_R|, IPD_proj, ILD_proj) → MLP → D_feat

    参数
    ----------
    enc_dim : int
        编码器输出维度（F_L, F_R 的维度）。
    proj_dim : int
        投影后的 IPD / ILD 维度。
    hidden_dim : int
        MLP 隐藏层大小。
    out_dim : int
        输出 ``D_feat`` 维度。
    dropout : float
        MLP 内部的 Dropout。
    """

    def __init__(
        self,
        enc_dim: int = 128,
        proj_dim: int = 128,
        hidden_dim: int = 256,
        out_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        # MLP 输入: (F_L - F_R) + |F_L - F_R| + IPD_proj + ILD_proj
        mlp_in = enc_dim + enc_dim + proj_dim + proj_dim

        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        f_l: torch.Tensor,
        f_r: torch.Tensor,
        ipd_proj: torch.Tensor,
        ild_proj: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数:
            f_l:      ``[B, T, D_enc]``
            f_r:      ``[B, T, D_enc]``
            ipd_proj: ``[B, T, proj_dim]``
            ild_proj: ``[B, T, proj_dim]``

        返回:
            d_feat: ``[B, T, out_dim]``
        """
        diff = f_l - f_r                 # [B, T, D_enc]
        abs_diff = diff.abs()            # [B, T, D_enc]
        cat = torch.cat([diff, abs_diff, ipd_proj, ild_proj], dim=-1)  # [B, T, 4*D]
        d_feat = self.mlp(cat)           # [B, T, out_dim]
        return d_feat
