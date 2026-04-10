"""双耳 DOA-Net — 完整模型。

串联所有子模块：
  1. 共享编码器                → F_L, F_R
  2. IPD / ILD 投影            → IPD_proj, ILD_proj
  3. 差异先验                  → D_feat
  4. 双向交叉注意力            → A_LR, A_RL
  5. 门控                      → gated_A_LR, gated_A_RL
  6. 特征融合
  7. BiGRU + 分类器            → logits

Forward 输入键（来自 DataLoader 的 batch）：
  "log_mag_L"  [B, T, F]
  "log_mag_R"  [B, T, F]
  "ipd"        [B, T, F]
  "ild"        [B, T, F]

Forward 输出：
  包含至少 {"logits": Tensor [B, num_classes]} 的字典
"""

import torch
import torch.nn as nn
from typing import Dict

from models.encoder import BinauralEncoder
from models.difference_prior import IPDILDProjection, DifferencePrior
from models.cross_attention import BidirectionalCrossAttention
from models.gating import GatingModule
from models.temporal_head import TemporalHead


class BinauralDOANet(nn.Module):
    """完整的双耳 DOA 估计网络。

    所有超参数详见 ``configs/default.yaml``。
    """

    def __init__(
        self,
        # 特征
        freq_bins: int = 257,
        # 编码器
        encoder_channels=None,
        encoder_out_dim: int = 128,
        # IPD/ILD 投影
        proj_dim: int = 128,
        # 差异先验
        prior_hidden_dim: int = 256,
        prior_out_dim: int = 128,
        # 交叉注意力
        attention_dim: int = 128,
        num_heads: int = 4,
        # 门控
        gate_dim: int = 128,
        use_gating: bool = True,
        # 时序头
        gru_hidden_size: int = 128,
        gru_num_layers: int = 2,
        gru_dropout: float = 0.1,
        # 分类器
        num_classes: int = 72,
        # 通用
        dropout: float = 0.2,
        # 回归
        use_regression: bool = False,
    ):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [32, 64, 128]

        # --- 1. 共享编码器（左右耳使用同一实例） ---
        self.encoder = BinauralEncoder(
            in_channels=1,
            channels=encoder_channels,
            out_dim=encoder_out_dim,
            dropout=dropout,
        )

        # --- 2. IPD / ILD 投影 ---
        self.ipd_ild_proj = IPDILDProjection(
            freq_bins=freq_bins,
            proj_dim=proj_dim,
        )

        # --- 3. 差异先验 ---
        self.diff_prior = DifferencePrior(
            enc_dim=encoder_out_dim,
            proj_dim=proj_dim,
            hidden_dim=prior_hidden_dim,
            out_dim=prior_out_dim,
            dropout=dropout,
        )

        # --- 4. 双向交叉注意力 ---
        self.cross_attn = BidirectionalCrossAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # --- 5. 门控 ---
        self.use_gating = use_gating
        if self.use_gating:
            self.gating = GatingModule(
                prior_dim=prior_out_dim,
                gate_dim=gate_dim,
            )
        else:
            self.gating = None

        # --- 6+7. 时序头 ---
        # 融合特征 = [F_L, F_R, gated_A_LR, gated_A_RL, (F_L - F_R)]
        # 各维度:      D     D     D           D           D
        fused_dim = 5 * encoder_out_dim

        self.temporal_head = TemporalHead(
            input_dim=fused_dim,
            gru_hidden_size=gru_hidden_size,
            gru_num_layers=gru_num_layers,
            num_classes=num_classes,
            gru_dropout=gru_dropout,
            dropout=dropout,
            use_regression=use_regression,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        参数:
            batch: 包含键 ``"log_mag_L"``、``"log_mag_R"``、
                   ``"ipd"``、``"ild"`` 的字典 — 每个均为 ``[B, T, F]``。

        返回:
            包含以下键的字典：
              - ``"logits"``: ``[B, num_classes]``
              - ``"d_feat"``: ``[B, T, prior_out_dim]``  （调试用）
        """
        log_mag_L = batch["log_mag_L"]  # [B, T, F]
        log_mag_R = batch["log_mag_R"]  # [B, T, F]
        ipd = batch["ipd"]              # [B, T, F]
        ild = batch["ild"]              # [B, T, F]

        # ---- 第 1 步: 共享编码器 ----
        # 编码器期望输入 [B, 1, T, F]
        f_l = self.encoder(log_mag_L.unsqueeze(1))  # [B, T', D_enc]
        f_r = self.encoder(log_mag_R.unsqueeze(1))  # [B, T', D_enc]

        # ---- 对齐 IPD/ILD 的时间维度与编码器输出 ----
        # 编码器可能因卷积步幅改变 T。
        # 我们的编码器在时间轴步幅为 1，所以 T' == T。但为安全起见做截断：
        T_enc = f_l.shape[1]
        ipd_aligned = ipd[:, :T_enc, :]   # [B, T', F]
        ild_aligned = ild[:, :T_enc, :]   # [B, T', F]

        # ---- 第 2 步: IPD / ILD 投影 ----
        ipd_proj, ild_proj = self.ipd_ild_proj(ipd_aligned, ild_aligned)
        # ipd_proj: [B, T', proj_dim],  ild_proj: [B, T', proj_dim]

        # ---- 第 3 步: 差异先验 ----
        d_feat = self.diff_prior(f_l, f_r, ipd_proj, ild_proj)  # [B, T', prior_out_dim]

        # ---- 第 4 步: 双向交叉注意力 ----
        a_lr, a_rl = self.cross_attn(f_l, f_r)  # 各 [B, T', D_enc]

        # ---- 第 5 步: 门控 ----
        if self.use_gating:
            gated_a_lr, gated_a_rl = self.gating(d_feat, a_lr, a_rl)
            # 各 [B, T', gate_dim]
        else:
            # 门控消融：直接使用注意力输出，不做调制
            gated_a_lr, gated_a_rl = a_lr, a_rl

        # ---- 第 6 步: 融合 ----
        fused = torch.cat([
            f_l,                # [B, T', D_enc]
            f_r,                # [B, T', D_enc]
            gated_a_lr,         # [B, T', D_enc]
            gated_a_rl,         # [B, T', D_enc]
            f_l - f_r,          # [B, T', D_enc]
        ], dim=-1)              # [B, T', 5 * D_enc]

        # ---- 第 7 步: 时序建模 + 分类器 + 回归器 ----
        outputs = self.temporal_head(fused)  # dict with "logits" and optionally "angle"

        # 添加调试用的中间特征
        outputs["d_feat"] = d_feat

        return outputs


def build_model(cfg) -> BinauralDOANet:
    """根据配置对象构建 :class:`BinauralDOANet`。

    参数:
        cfg: 包含 ``model`` 和 ``feature`` 子配置的 Config 对象。

    返回:
        实例化的模型（尚未移到设备上）。
    """
    m = cfg.model
    f = cfg.feature

    freq_bins = f.n_fft // 2 + 1

    # 检查是否启用回归
    use_regression = getattr(m, 'use_regression', False)
    use_gating = getattr(m, 'use_gating', True)

    return BinauralDOANet(
        freq_bins=freq_bins,
        encoder_channels=m.encoder_channels,
        encoder_out_dim=m.encoder_out_dim,
        proj_dim=m.proj_dim,
        prior_hidden_dim=m.prior_hidden_dim,
        prior_out_dim=m.prior_out_dim,
        attention_dim=m.attention_dim,
        num_heads=m.num_heads,
        gate_dim=m.gate_dim,
        use_gating=use_gating,
        gru_hidden_size=m.gru_hidden_size,
        gru_num_layers=m.gru_num_layers,
        gru_dropout=m.gru_dropout,
        num_classes=m.num_classes,
        dropout=m.dropout,
        use_regression=use_regression,
    )
