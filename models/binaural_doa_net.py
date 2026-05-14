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
import math

from models.encoder import BinauralEncoder
from models.difference_prior import IPDILDProjection, DifferencePrior
from models.cross_attention import BidirectionalCrossAttention
from models.gating import GatingModule
from models.temporal_head import TemporalHead
from models.native_lite_v7 import (
    NativeLiteDOANet,
    NativeLiteCueConcatDOANet,
    NativeLiteLiteCueConcatDOANet,
)
from models.sdel_crnn_baseline import SDELCRNNBaseline


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
        use_independent_gating: bool = True,
        use_residual_gating: bool = True,
        # 时序头
        gru_hidden_size: int = 128,
        gru_num_layers: int = 2,
        gru_dropout: float = 0.1,
        # 分类器
        num_classes: int = 72,
        azimuth_range = (-180.0, 180.0),
        # 通用
        dropout: float = 0.2,
        # 回归
        use_regression: bool = False,
        use_pure_regression: bool = False,
        # 增强双耳特征
        use_enhanced_binaural_features: bool = False,
        # Attention bias
        use_attention_bias: bool = True,
        attention_bias_rank: int = 16,
        # 时序池化
        use_attention_pooling: bool = True,
        # 前后辅助任务
        use_front_back_auxiliary: bool = False,
        # 简化版 native 主线
        use_simple_binaural_fusion: bool = False,
        use_simple_cross_attention: bool = False,
        use_raw_ipd: bool = True,
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

        self.use_enhanced_binaural_features = use_enhanced_binaural_features
        self.use_simple_binaural_fusion = use_simple_binaural_fusion
        self.use_simple_cross_attention = use_simple_cross_attention
        self.use_raw_ipd = use_raw_ipd

        if self.use_simple_binaural_fusion:
            cue_in_bins = freq_bins * 4 if self.use_enhanced_binaural_features else freq_bins * 2
            self.simple_cue_proj = nn.Linear(cue_in_bins, proj_dim)
            self.ipd_ild_proj = None
            self.diff_prior = None
            if self.use_simple_cross_attention:
                self.cross_attn = BidirectionalCrossAttention(
                    embed_dim=encoder_out_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            else:
                self.cross_attn = None
            self.use_attention_bias = False
            self.attention_bias_rank = 0
            self.gating = None
            self.use_gating = False
            # 简化主线：保留左右耳内容、显式差异和轻量双耳线索投影。
            fused_dim = 4 * encoder_out_dim + proj_dim
            if self.use_simple_cross_attention:
                fused_dim += 2 * encoder_out_dim
        else:
            # 增强特征开启时：
            # ipd 输入 = [ipd, sin(ipd), cos(ipd), coherence] -> 4F
            # ild 输入 = [ild, coherence] -> 2F
            ipd_in_bins = freq_bins * 4 if self.use_enhanced_binaural_features else freq_bins
            ild_in_bins = freq_bins * 2 if self.use_enhanced_binaural_features else freq_bins

            # --- 2. IPD / ILD 投影 ---
            self.ipd_ild_proj = IPDILDProjection(
                freq_bins=freq_bins,
                proj_dim=proj_dim,
                ipd_in_bins=ipd_in_bins,
                ild_in_bins=ild_in_bins,
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

            # --- 4.1 差异先验引导的双向attention bias ---
            self.use_attention_bias = use_attention_bias
            self.attention_bias_rank = max(int(attention_bias_rank), 1)
            if self.use_attention_bias:
                r = self.attention_bias_rank
                self.bias_u_lr = nn.Linear(prior_out_dim, r)
                self.bias_v_lr = nn.Linear(prior_out_dim, r)
                self.bias_u_rl = nn.Linear(prior_out_dim, r)
                self.bias_v_rl = nn.Linear(prior_out_dim, r)

            # --- 5. 门控 ---
            self.use_gating = use_gating
            if self.use_gating:
                self.gating = GatingModule(
                    prior_dim=prior_out_dim,
                    gate_dim=gate_dim,
                    use_independent_gating=use_independent_gating,
                    use_residual_gating=use_residual_gating,
                )
            else:
                self.gating = None

            # --- 6+7. 时序头 ---
            # 融合特征 = [F_L, F_R, gated_A_LR, gated_A_RL, (F_L - F_R)]
            fused_dim = 5 * encoder_out_dim

        self.temporal_head = TemporalHead(
            input_dim=fused_dim,
            gru_hidden_size=gru_hidden_size,
            gru_num_layers=gru_num_layers,
            num_classes=num_classes,
            gru_dropout=gru_dropout,
            dropout=dropout,
            use_regression=use_regression,
            use_pure_regression=use_pure_regression,
            use_attention_pooling=use_attention_pooling,
            use_front_back_auxiliary=use_front_back_auxiliary,
            azimuth_range=tuple(azimuth_range),
        )

    def _build_attention_bias(self, d_feat: torch.Tensor) -> tuple:
        """由差异先验生成双向低秩attention score bias。"""
        # d_feat: [B, T, Dp]
        r = float(self.attention_bias_rank)

        u_lr = self.bias_u_lr(d_feat)  # [B, T, r]
        v_lr = self.bias_v_lr(d_feat)  # [B, T, r]
        bias_lr = torch.matmul(u_lr, v_lr.transpose(1, 2)) / math.sqrt(r)

        u_rl = self.bias_u_rl(d_feat)  # [B, T, r]
        v_rl = self.bias_v_rl(d_feat)  # [B, T, r]
        bias_rl = torch.matmul(u_rl, v_rl.transpose(1, 2)) / math.sqrt(r)

        return bias_lr, bias_rl

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

        ipd_sin = batch.get("ipd_sin")
        ipd_cos = batch.get("ipd_cos")
        coh = batch.get("coherence")

        if self.use_enhanced_binaural_features:
            # 若batch中未提供增强特征，则退化为在线构造以保证兼容。
            if coh is None:
                coh = torch.ones_like(ipd_aligned)
            else:
                coh = coh[:, :T_enc, :]

            if ipd_sin is None:
                ipd_sin = torch.sin(ipd_aligned)
            else:
                ipd_sin = ipd_sin[:, :T_enc, :]

            if ipd_cos is None:
                ipd_cos = torch.cos(ipd_aligned)
            else:
                ipd_cos = ipd_cos[:, :T_enc, :]

            if not self.use_simple_binaural_fusion or self.use_raw_ipd:
                ipd_aligned = torch.cat([ipd_aligned, ipd_sin, ipd_cos, coh], dim=-1)
            ild_aligned = torch.cat([ild_aligned, coh], dim=-1)

        if self.use_simple_binaural_fusion:
            diff = f_l - f_r
            abs_diff = diff.abs()
            if self.use_enhanced_binaural_features:
                cue_parts = [ild[:, :T_enc, :]]
                if self.use_raw_ipd:
                    cue_parts.append(ipd[:, :T_enc, :])
                cue_parts.extend([ipd_sin, ipd_cos, coh])
                cue_in = torch.cat(cue_parts, dim=-1)
            else:
                cue_in = torch.cat([ild_aligned, ipd[:, :T_enc, :]], dim=-1)
            cue_proj = self.simple_cue_proj(cue_in)
            fused_parts = [f_l, f_r]
            if self.use_simple_cross_attention and self.cross_attn is not None:
                a_lr, a_rl = self.cross_attn(f_l, f_r, bias_lr=None, bias_rl=None)
                fused_parts.extend([a_lr, a_rl])
            fused_parts.extend([diff, abs_diff, cue_proj])
            fused = torch.cat(fused_parts, dim=-1)
            d_feat = cue_proj
        else:
            # ---- 第 2 步: IPD / ILD 投影 ----
            ipd_proj, ild_proj = self.ipd_ild_proj(ipd_aligned, ild_aligned)
            # ---- 第 3 步: 差异先验 ----
            d_feat = self.diff_prior(f_l, f_r, ipd_proj, ild_proj)

            # ---- 第 4 步: 双向交叉注意力（可选score bias）----
            if self.use_attention_bias:
                bias_lr, bias_rl = self._build_attention_bias(d_feat)
            else:
                bias_lr, bias_rl = None, None

            a_lr, a_rl = self.cross_attn(f_l, f_r, bias_lr=bias_lr, bias_rl=bias_rl)

            # ---- 第 5 步: 门控 ----
            if self.use_gating:
                gated_a_lr, gated_a_rl = self.gating(d_feat, a_lr, a_rl, f_l, f_r)
            else:
                gated_a_lr, gated_a_rl = a_lr, a_rl

            # ---- 第 6 步: 融合 ----
            fused = torch.cat([
                f_l,
                f_r,
                gated_a_lr,
                gated_a_rl,
                f_l - f_r,
            ], dim=-1)

        # ---- 第 7 步: 时序建模 + 分类器 + 回归器 ----
        outputs = self.temporal_head(fused)  # dict with "logits" and optionally "angle"

        # 添加调试用的中间特征
        outputs["d_feat"] = d_feat

        return outputs


def build_model(cfg):
    """根据配置对象构建 :class:`BinauralDOANet`。

    参数:
        cfg: 包含 ``model`` 和 ``feature`` 子配置的 Config 对象。

    返回:
        实例化的模型（尚未移到设备上）。
    """
    m = cfg.model
    f = cfg.feature

    freq_bins = f.n_fft // 2 + 1
    model_type = getattr(m, "type", "binaural_doa_net")

    if model_type in {"sdel_doa_reg", "sdel_doa_cls"}:
        return SDELCRNNBaseline(
            freq_bins=freq_bins,
            cnn_channels=getattr(m, "sdel_cnn_channels", [32, 64, 128]),
            f_pool_size=getattr(m, "sdel_f_pool_size", [4, 4, 4]),
            t_pool_size=getattr(m, "sdel_t_pool_size", [1, 1, 1]),
            kernel_size=tuple(getattr(m, "sdel_kernel_size", [3, 3])),
            dropout=m.dropout,
            gru_hidden_size=m.gru_hidden_size,
            gru_num_layers=m.gru_num_layers,
            fnn_size=getattr(m, "sdel_fnn_size", 128),
            num_fnn_layers=getattr(m, "sdel_num_fnn_layers", 2),
            num_classes=m.num_classes,
            azimuth_range=tuple(m.azimuth_range),
            use_front_back_auxiliary=getattr(m, "use_front_back_auxiliary", False),
            output_mode="reg" if model_type == "sdel_doa_reg" else "cls",
        )

    if model_type == "native_lite_v7":
        return NativeLiteDOANet(
            freq_bins=freq_bins,
            encoder_channels=getattr(m, "encoder_channels", [24, 48, 96]),
            encoder_out_dim=getattr(m, "encoder_out_dim", 96),
            encoder_variant=getattr(m, "encoder_variant", "v1"),
            content_input_mode=getattr(m, "content_input_mode", "logmag"),
            use_cue_stream=getattr(m, "use_cue_stream", True),
            cue_feature_mode=getattr(m, "cue_feature_mode", "all"),
            use_cross_ear_interaction=getattr(m, "use_cross_ear_interaction", False),
            cue_bands=getattr(m, "cue_bands", 32),
            cue_hidden_dim=getattr(m, "cue_hidden_dim", 64),
            fusion_dim=getattr(m, "fusion_dim", 160),
            gru_hidden_size=m.gru_hidden_size,
            gru_num_layers=m.gru_num_layers,
            gru_dropout=m.gru_dropout,
            num_classes=m.num_classes,
            azimuth_range=tuple(m.azimuth_range),
            dropout=m.dropout,
            use_attention_pooling=getattr(m, "use_attention_pooling", True),
            use_front_back_auxiliary=getattr(m, "use_front_back_auxiliary", True),
            use_regression=getattr(m, "use_regression", False),
            use_pure_regression=getattr(m, "use_pure_regression", False),
        )

    if model_type == "native_lite_v7_cue_concat":
        return NativeLiteCueConcatDOANet(
            freq_bins=freq_bins,
            encoder_channels=getattr(m, "encoder_channels", [24, 40, 64]),
            encoder_out_dim=getattr(m, "encoder_out_dim", 96),
            encoder_variant=getattr(m, "encoder_variant", "v2_balanced"),
            content_input_mode=getattr(m, "content_input_mode", "logmag"),
            cue_feature_mode=getattr(m, "cue_feature_mode", "ild_phase"),
            cue_encoder_channels=getattr(m, "cue_encoder_channels", [8, 16, 24]),
            cue_encoder_out_dim=getattr(m, "cue_encoder_out_dim", 32),
            content_fusion_dim=getattr(m, "content_fusion_dim", 96),
            use_cross_ear_interaction=getattr(m, "use_cross_ear_interaction", False),
            gru_hidden_size=m.gru_hidden_size,
            gru_num_layers=m.gru_num_layers,
            gru_dropout=m.gru_dropout,
            num_classes=m.num_classes,
            azimuth_range=tuple(m.azimuth_range),
            dropout=m.dropout,
            use_attention_pooling=getattr(m, "use_attention_pooling", True),
            use_front_back_auxiliary=getattr(m, "use_front_back_auxiliary", True),
            use_regression=getattr(m, "use_regression", False),
            use_pure_regression=getattr(m, "use_pure_regression", False),
        )

    if model_type == "native_lite_v7_lite_cue_concat":
        return NativeLiteLiteCueConcatDOANet(
            freq_bins=freq_bins,
            encoder_channels=getattr(m, "encoder_channels", [24, 40, 64]),
            encoder_out_dim=getattr(m, "encoder_out_dim", 96),
            encoder_variant=getattr(m, "encoder_variant", "v2_balanced"),
            content_input_mode=getattr(m, "content_input_mode", "logmag"),
            cue_feature_mode=getattr(m, "cue_feature_mode", "ild_phase"),
            content_fusion_dim=getattr(m, "content_fusion_dim", 96),
            lite_cue_bands=getattr(m, "lite_cue_bands", 16),
            lite_cue_hidden_dim=getattr(m, "lite_cue_hidden_dim", 48),
            cue_encoder_out_dim=getattr(m, "cue_encoder_out_dim", 32),
            lite_cue_kernel_size=getattr(m, "lite_cue_kernel_size", 3),
            use_cross_ear_interaction=getattr(m, "use_cross_ear_interaction", False),
            gru_hidden_size=m.gru_hidden_size,
            gru_num_layers=m.gru_num_layers,
            gru_dropout=m.gru_dropout,
            num_classes=m.num_classes,
            azimuth_range=tuple(m.azimuth_range),
            dropout=m.dropout,
            use_attention_pooling=getattr(m, "use_attention_pooling", True),
            use_front_back_auxiliary=getattr(m, "use_front_back_auxiliary", True),
            use_regression=getattr(m, "use_regression", False),
            use_pure_regression=getattr(m, "use_pure_regression", False),
        )

    # 检查是否启用回归
    use_regression = getattr(m, 'use_regression', False)
    use_pure_regression = getattr(m, 'use_pure_regression', False)
    use_gating = getattr(m, 'use_gating', True)
    use_independent_gating = getattr(m, 'use_independent_gating', True)
    use_residual_gating = getattr(m, 'use_residual_gating', True)
    use_enhanced_binaural_features = getattr(m, 'use_enhanced_binaural_features', False)
    use_attention_bias = getattr(m, 'use_attention_bias', True)
    attention_bias_rank = getattr(m, 'attention_bias_rank', 16)
    use_attention_pooling = getattr(m, 'use_attention_pooling', True)
    use_front_back_auxiliary = getattr(m, 'use_front_back_auxiliary', False)
    use_simple_binaural_fusion = getattr(m, 'use_simple_binaural_fusion', False)
    use_simple_cross_attention = getattr(m, 'use_simple_cross_attention', False)
    use_raw_ipd = getattr(m, 'use_raw_ipd', True)

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
        use_independent_gating=use_independent_gating,
        use_residual_gating=use_residual_gating,
        gru_hidden_size=m.gru_hidden_size,
        gru_num_layers=m.gru_num_layers,
        gru_dropout=m.gru_dropout,
        num_classes=m.num_classes,
        dropout=m.dropout,
        use_regression=use_regression,
        use_pure_regression=use_pure_regression,
        use_enhanced_binaural_features=use_enhanced_binaural_features,
        use_attention_bias=use_attention_bias,
        attention_bias_rank=attention_bias_rank,
        use_attention_pooling=use_attention_pooling,
        use_front_back_auxiliary=use_front_back_auxiliary,
        use_simple_binaural_fusion=use_simple_binaural_fusion,
        use_simple_cross_attention=use_simple_cross_attention,
        use_raw_ipd=use_raw_ipd,
    )
