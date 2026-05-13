"""更干净的 native 双耳 DOA 主线。

设计目标：
1. 将“内容信息”和“双耳空间线索”解耦；
2. 避免把大量原始特征直接拼接到时序头，减小 GRU 参数量；
3. 保留 front/back 辅助任务，弱化复杂 prior / bias / gating 叙事。

结构：
    content stream:
        log_mag_L / log_mag_R -> shared encoder -> F_L, F_R

    cue stream:
        [ILD, sin(IPD), cos(IPD), coherence]
        -> band pooling -> small MLP -> cue_feat

    fusion:
        mean(F_L, F_R), diff(F_L, F_R), |diff|, cue_feat
        -> low-dimensional bottleneck fusion

    temporal:
        light BiGRU (+ optional attention pooling)

    heads:
        DOA classifier + optional front/back auxiliary classifier
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import BinauralEncoder, BinauralEncoderV2Balanced
from models.temporal_head import TemporalHead


class NativeLiteDOANet(nn.Module):
    """内容流 + 双耳线索流 + 低维融合 的轻量 native DOA 模型。"""

    def __init__(
        self,
        freq_bins: int = 257,
        encoder_channels=None,
        encoder_out_dim: int = 96,
        encoder_variant: str = "v1",
        content_input_mode: str = "logmag",
        use_cue_stream: bool = True,
        cue_feature_mode: str = "all",
        use_cross_ear_interaction: bool = False,
        cue_bands: int = 32,
        cue_hidden_dim: int = 64,
        fusion_dim: int = 160,
        gru_hidden_size: int = 96,
        gru_num_layers: int = 1,
        gru_dropout: float = 0.1,
        num_classes: int = 72,
        azimuth_range=(-180.0, 180.0),
        dropout: float = 0.2,
        use_attention_pooling: bool = True,
        use_front_back_auxiliary: bool = True,
        use_regression: bool = False,
        use_pure_regression: bool = False,
    ):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [24, 48, 96]

        if content_input_mode not in {"logmag", "complex_ri"}:
            raise ValueError(f"Unsupported content_input_mode: {content_input_mode}")
        if cue_feature_mode not in {"all", "phase_only"}:
            raise ValueError(f"Unsupported cue_feature_mode: {cue_feature_mode}")
        self.content_input_mode = content_input_mode
        self.use_cue_stream = use_cue_stream
        self.cue_feature_mode = cue_feature_mode
        self.use_cross_ear_interaction = use_cross_ear_interaction
        content_in_channels = 1 if content_input_mode == "logmag" else 2

        if encoder_variant == "v1":
            encoder_cls = BinauralEncoder
        elif encoder_variant == "v2_balanced":
            encoder_cls = BinauralEncoderV2Balanced
        else:
            raise ValueError(f"Unsupported encoder_variant: {encoder_variant}")

        self.encoder = encoder_cls(
            in_channels=content_in_channels,
            channels=encoder_channels,
            out_dim=encoder_out_dim,
            dropout=dropout,
        )

        self.cue_bands = cue_bands

        num_cues = 4 if cue_feature_mode == "all" else 2
        cue_flat_dim = num_cues * cue_bands
        if self.use_cue_stream:
            self.cue_mlp = nn.Sequential(
                nn.Linear(cue_flat_dim, cue_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(cue_hidden_dim, cue_hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.cue_proj = nn.Linear(cue_hidden_dim, fusion_dim)
        else:
            self.cue_mlp = None
            self.cue_proj = None

        self.mean_proj = nn.Linear(encoder_out_dim, fusion_dim)
        self.diff_proj = nn.Linear(encoder_out_dim, fusion_dim)
        self.abs_diff_proj = nn.Linear(encoder_out_dim, fusion_dim)

        if self.use_cross_ear_interaction:
            # Lightweight cross-ear interaction: each ear receives a residual
            # projection from the opposite ear without reintroducing large
            # attention or gating modules.
            self.cross_rl = nn.Linear(encoder_out_dim, encoder_out_dim)
            self.cross_lr = nn.Linear(encoder_out_dim, encoder_out_dim)
            self.cross_norm_l = nn.LayerNorm(encoder_out_dim)
            self.cross_norm_r = nn.LayerNorm(encoder_out_dim)
        else:
            self.cross_rl = None
            self.cross_lr = None
            self.cross_norm_l = None
            self.cross_norm_r = None

        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.fusion_dropout = nn.Dropout(dropout)

        self.temporal_head = TemporalHead(
            input_dim=fusion_dim,
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

    def _pool_cues(self, cue_tensor: torch.Tensor) -> torch.Tensor:
        """对双耳线索做频带级压缩。

        参数:
            cue_tensor: [B, T, 4, F]
        返回:
            [B, T, 4*cue_bands]
        """
        bsz, t, num_cues, f = cue_tensor.shape
        x = cue_tensor.reshape(bsz * t, num_cues, f)  # [B*T, 4, F]
        x = F.adaptive_avg_pool1d(x, self.cue_bands)  # [B*T, 4, BANDS]
        x = x.reshape(bsz, t, num_cues * self.cue_bands)
        return x

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        log_mag_L = batch["log_mag_L"]  # [B, T, F]
        log_mag_R = batch["log_mag_R"]  # [B, T, F]
        ild = batch["ild"]              # [B, T, F]
        ipd = batch["ipd"]              # [B, T, F]

        if self.content_input_mode == "logmag":
            left_content = log_mag_L.unsqueeze(1)  # [B, 1, T, F]
            right_content = log_mag_R.unsqueeze(1)
        else:
            left_content = torch.stack(
                [batch["spec_real_L"], batch["spec_imag_L"]],
                dim=1,
            )  # [B, 2, T, F]
            right_content = torch.stack(
                [batch["spec_real_R"], batch["spec_imag_R"]],
                dim=1,
            )  # [B, 2, T, F]

        f_l = self.encoder(left_content)     # [B, T', D]
        f_r = self.encoder(right_content)    # [B, T', D]

        if self.use_cross_ear_interaction:
            cross_l = self.cross_norm_l(self.cross_rl(f_r))
            cross_r = self.cross_norm_r(self.cross_lr(f_l))
            f_l = f_l + cross_l
            f_r = f_r + cross_r

        t_enc = f_l.shape[1]
        ild = ild[:, :t_enc, :]

        ipd_sin = batch.get("ipd_sin")
        ipd_cos = batch.get("ipd_cos")
        coherence = batch.get("coherence")

        if ipd_sin is None:
            ipd_sin = torch.sin(ipd[:, :t_enc, :])
        else:
            ipd_sin = ipd_sin[:, :t_enc, :]

        if ipd_cos is None:
            ipd_cos = torch.cos(ipd[:, :t_enc, :])
        else:
            ipd_cos = ipd_cos[:, :t_enc, :]

        if coherence is None:
            coherence = torch.ones_like(ild)
        else:
            coherence = coherence[:, :t_enc, :]

        mean_feat = 0.5 * (f_l + f_r)
        diff_feat = f_l - f_r
        abs_diff_feat = diff_feat.abs()

        fused = (
            self.mean_proj(mean_feat)
            + self.diff_proj(diff_feat)
            + self.abs_diff_proj(abs_diff_feat)
        )

        cue_feat = None
        if self.use_cue_stream:
            if self.cue_feature_mode == "phase_only":
                cue_tensor = torch.stack([ipd_sin, ipd_cos], dim=2)  # [B, T, 2, F]
            else:
                cue_tensor = torch.stack([ild, ipd_sin, ipd_cos, coherence], dim=2)  # [B, T, 4, F]
            cue_pooled = self._pool_cues(cue_tensor)                              # [B, T, 4*BANDS]
            cue_feat = self.cue_mlp(cue_pooled)                                   # [B, T, C]
            fused = fused + self.cue_proj(cue_feat)

        fused = self.fusion_norm(fused)
        fused = self.fusion_dropout(fused)

        outputs = self.temporal_head(fused)
        if cue_feat is not None:
            outputs["cue_feat"] = cue_feat
        outputs["fused_feat"] = fused
        return outputs
