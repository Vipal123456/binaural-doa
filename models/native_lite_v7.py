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


class LiteCueEncoder(nn.Module):
    """轻量 cue encoder：先做频带压缩，再做时间维 1D 卷积。"""

    def __init__(
        self,
        in_channels: int,
        cue_bands: int = 16,
        temporal_hidden_dim: int = 48,
        out_dim: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.2,
        encoder_type: str = "temporal_conv",
    ):
        super().__init__()
        self.cue_bands = cue_bands
        self.encoder_type = encoder_type
        flat_dim = in_channels * cue_bands
        if encoder_type == "temporal_conv":
            padding = kernel_size // 2
            self.temporal_net = nn.Sequential(
                nn.Conv1d(flat_dim, temporal_hidden_dim, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(temporal_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(temporal_hidden_dim, out_dim, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
            )
        elif encoder_type == "temporal_conv_bandattn":
            padding = kernel_size // 2
            self.band_gate = nn.Sequential(
                nn.Linear(in_channels * cue_bands, cue_bands),
                nn.ReLU(inplace=True),
                nn.Linear(cue_bands, cue_bands),
            )
            self.temporal_net = nn.Sequential(
                nn.Conv1d(flat_dim, temporal_hidden_dim, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(temporal_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(temporal_hidden_dim, out_dim, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
            )
        elif encoder_type == "temporal_conv_ms":
            padding3 = 3 // 2
            padding5 = 5 // 2
            branch_out_dim = max(temporal_hidden_dim // 2, 8)
            self.temporal_branch_k3 = nn.Sequential(
                nn.Conv1d(flat_dim, branch_out_dim, kernel_size=3, padding=padding3),
                nn.BatchNorm1d(branch_out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.temporal_branch_k5 = nn.Sequential(
                nn.Conv1d(flat_dim, branch_out_dim, kernel_size=5, padding=padding5),
                nn.BatchNorm1d(branch_out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.temporal_fuse = nn.Sequential(
                nn.Conv1d(branch_out_dim * 2, out_dim, kernel_size=1),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
            )
        elif encoder_type == "mlp":
            self.temporal_net = nn.Sequential(
                nn.Linear(flat_dim, temporal_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(temporal_hidden_dim, out_dim),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"Unsupported LiteCueEncoder encoder_type: {encoder_type}")

    def forward(self, cue_tensor: torch.Tensor) -> torch.Tensor:
        # cue_tensor: [B, C, T, F]
        bsz, num_cues, time_steps, freq_bins = cue_tensor.shape
        x = cue_tensor.reshape(bsz * num_cues * time_steps, 1, freq_bins)
        x = F.adaptive_avg_pool1d(x, self.cue_bands)
        x = x.reshape(bsz, num_cues, time_steps, self.cue_bands)
        x = x.permute(0, 2, 1, 3)  # [B, T, C, bands]
        if self.encoder_type == "temporal_conv_bandattn":
            x_flat = x.reshape(bsz, time_steps, num_cues * self.cue_bands)
            band_logits = self.band_gate(x_flat)  # [B, T, bands]
            band_weight = torch.softmax(band_logits, dim=-1).unsqueeze(2)  # [B, T, 1, bands]
            x = x * band_weight
        x = x.reshape(bsz, time_steps, num_cues * self.cue_bands)
        if self.encoder_type == "temporal_conv":
            x = x.transpose(1, 2)  # [B, C*bands, T]
            x = self.temporal_net(x)
            return x.transpose(1, 2)  # [B, T, out_dim]
        if self.encoder_type == "temporal_conv_bandattn":
            x = x.transpose(1, 2)  # [B, C*bands, T]
            x = self.temporal_net(x)
            return x.transpose(1, 2)  # [B, T, out_dim]
        if self.encoder_type == "temporal_conv_ms":
            x = x.transpose(1, 2)  # [B, C*bands, T]
            x3 = self.temporal_branch_k3(x)
            x5 = self.temporal_branch_k5(x)
            x = torch.cat([x3, x5], dim=1)
            x = self.temporal_fuse(x)
            return x.transpose(1, 2)  # [B, T, out_dim]
        return self.temporal_net(x)  # [B, T, out_dim]


class DualBranchCueEncoder(nn.Module):
    """双分支 cue encoder：
    - value branch 处理 ILD / sin(IPD) / cos(IPD)
    - reliability branch 处理 coherence
    第一版先用 concat 融合，保留更强的可解释性。
    """

    def __init__(
        self,
        cue_bands: int = 16,
        temporal_hidden_dim: int = 48,
        value_out_dim: int = 24,
        reliability_out_dim: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.2,
        encoder_type: str = "temporal_conv",
        fusion_mode: str = "concat",
    ):
        super().__init__()
        if fusion_mode not in {"concat", "gate"}:
            raise ValueError(f"Unsupported DualBranchCueEncoder fusion_mode: {fusion_mode}")

        self.fusion_mode = fusion_mode
        self.value_encoder = LiteCueEncoder(
            in_channels=3,
            cue_bands=cue_bands,
            temporal_hidden_dim=temporal_hidden_dim,
            out_dim=value_out_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            encoder_type=encoder_type,
        )
        self.reliability_encoder = LiteCueEncoder(
            in_channels=1,
            cue_bands=cue_bands,
            temporal_hidden_dim=max(temporal_hidden_dim // 2, 8),
            out_dim=reliability_out_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            encoder_type=encoder_type,
        )
        if fusion_mode == "gate":
            self.rel_to_gate = nn.Sequential(
                nn.Linear(reliability_out_dim, value_out_dim),
                nn.Sigmoid(),
            )
        else:
            self.rel_to_gate = None

    @property
    def out_dim(self) -> int:
        if self.fusion_mode == "gate":
            return self.value_encoder.temporal_net[-2].num_features if isinstance(self.value_encoder.temporal_net, nn.Sequential) and hasattr(self.value_encoder.temporal_net[-2], "num_features") else None
        return (
            self.value_encoder.temporal_net[-2].num_features + self.reliability_encoder.temporal_net[-2].num_features
            if isinstance(self.value_encoder.temporal_net, nn.Sequential)
            and hasattr(self.value_encoder.temporal_net[-2], "num_features")
            and isinstance(self.reliability_encoder.temporal_net, nn.Sequential)
            and hasattr(self.reliability_encoder.temporal_net[-2], "num_features")
            else None
        )

    def forward(
        self,
        value_tensor: torch.Tensor,
        reliability_tensor: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        value_feat = self.value_encoder(value_tensor)
        reliability_feat = self.reliability_encoder(reliability_tensor)
        if self.fusion_mode == "gate":
            gate = self.rel_to_gate(reliability_feat)
            cue_feat = value_feat * gate
        else:
            gate = None
            cue_feat = torch.cat([value_feat, reliability_feat], dim=-1)
        return {
            "cue_feat": cue_feat,
            "cue_value_feat": value_feat,
            "cue_reliability_feat": reliability_feat,
            "cue_gate": gate,
        }


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


class NativeLiteCueConcatDOANet(nn.Module):
    """内容流保持不变，cue 流单独编码，再拼接送入 GRU 的轻量 native 模型。"""

    def __init__(
        self,
        freq_bins: int = 257,
        encoder_channels=None,
        encoder_out_dim: int = 96,
        encoder_variant: str = "v2_balanced",
        content_input_mode: str = "logmag",
        cue_feature_mode: str = "ild_phase",
        cue_encoder_channels=None,
        cue_encoder_out_dim: int = 32,
        content_fusion_dim: int = 96,
        use_cross_ear_interaction: bool = False,
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
            encoder_channels = [24, 40, 64]
        if cue_encoder_channels is None:
            cue_encoder_channels = [8, 16, 24]

        if content_input_mode not in {"logmag", "complex_ri"}:
            raise ValueError(f"Unsupported content_input_mode: {content_input_mode}")
        if cue_feature_mode not in {"all", "phase_only", "ild_phase"}:
            raise ValueError(f"Unsupported cue_feature_mode: {cue_feature_mode}")

        self.content_input_mode = content_input_mode
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

        if cue_feature_mode == "all":
            cue_in_channels = 4
        elif cue_feature_mode == "phase_only":
            cue_in_channels = 2
        else:
            cue_in_channels = 3

        self.cue_encoder = BinauralEncoderV2Balanced(
            in_channels=cue_in_channels,
            channels=cue_encoder_channels,
            out_dim=cue_encoder_out_dim,
            dropout=dropout,
        )

        self.content_fusion = nn.Sequential(
            nn.Linear(encoder_out_dim * 3, content_fusion_dim),
            nn.LayerNorm(content_fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        if self.use_cross_ear_interaction:
            self.cross_rl = nn.Linear(encoder_out_dim, encoder_out_dim)
            self.cross_lr = nn.Linear(encoder_out_dim, encoder_out_dim)
            self.cross_norm_l = nn.LayerNorm(encoder_out_dim)
            self.cross_norm_r = nn.LayerNorm(encoder_out_dim)
        else:
            self.cross_rl = None
            self.cross_lr = None
            self.cross_norm_l = None
            self.cross_norm_r = None

        temporal_input_dim = content_fusion_dim + cue_encoder_out_dim
        self.fusion_norm = nn.LayerNorm(temporal_input_dim)
        self.fusion_dropout = nn.Dropout(dropout)

        self.temporal_head = TemporalHead(
            input_dim=temporal_input_dim,
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

    def _build_cue_tensor(
        self,
        ild: torch.Tensor,
        ipd_sin: torch.Tensor,
        ipd_cos: torch.Tensor,
        coherence: torch.Tensor,
    ) -> torch.Tensor:
        if self.cue_feature_mode == "phase_only":
            return torch.stack([ipd_sin, ipd_cos], dim=1)
        if self.cue_feature_mode == "ild_phase":
            return torch.stack([ild, ipd_sin, ipd_cos], dim=1)
        return torch.stack([ild, ipd_sin, ipd_cos, coherence], dim=1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        log_mag_L = batch["log_mag_L"]
        log_mag_R = batch["log_mag_R"]
        ild = batch["ild"]
        ipd = batch["ipd"]

        if self.content_input_mode == "logmag":
            left_content = log_mag_L.unsqueeze(1)
            right_content = log_mag_R.unsqueeze(1)
        else:
            left_content = torch.stack([batch["spec_real_L"], batch["spec_imag_L"]], dim=1)
            right_content = torch.stack([batch["spec_real_R"], batch["spec_imag_R"]], dim=1)

        f_l = self.encoder(left_content)
        f_r = self.encoder(right_content)

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
        content_feat = torch.cat([mean_feat, diff_feat, abs_diff_feat], dim=-1)
        content_feat = self.content_fusion(content_feat)

        cue_tensor = self._build_cue_tensor(ild, ipd_sin, ipd_cos, coherence)
        cue_feat = self.cue_encoder(cue_tensor)

        fused = torch.cat([content_feat, cue_feat], dim=-1)
        fused = self.fusion_norm(fused)
        fused = self.fusion_dropout(fused)

        outputs = self.temporal_head(fused)
        outputs["cue_feat"] = cue_feat
        outputs["fused_feat"] = fused
        outputs["content_feat"] = content_feat
        return outputs


class NativeLiteLiteCueConcatDOANet(nn.Module):
    """内容流保留 encoder v2，cue 流改为轻量 band-pool + temporal conv。"""

    def __init__(
        self,
        freq_bins: int = 257,
        encoder_channels=None,
        encoder_out_dim: int = 96,
        encoder_variant: str = "v2_balanced",
        content_input_mode: str = "logmag",
        cue_feature_mode: str = "ild_phase",
        content_relation_mode: str = "mean_diff_absdiff",
        content_fusion_dim: int = 96,
        lite_cue_bands: int = 16,
        lite_cue_hidden_dim: int = 48,
        cue_encoder_out_dim: int = 32,
        lite_cue_kernel_size: int = 3,
        lite_cue_encoder_type: str = "temporal_conv",
        use_cross_ear_interaction: bool = False,
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
            encoder_channels = [24, 40, 64]

        if content_input_mode not in {"logmag", "complex_ri"}:
            raise ValueError(f"Unsupported content_input_mode: {content_input_mode}")
        if cue_feature_mode not in {"all", "phase_only", "ild_phase"}:
            raise ValueError(f"Unsupported cue_feature_mode: {cue_feature_mode}")
        if content_relation_mode not in {"mean_diff_absdiff", "mean_diff", "diff_only"}:
            raise ValueError(f"Unsupported content_relation_mode: {content_relation_mode}")

        self.content_input_mode = content_input_mode
        self.cue_feature_mode = cue_feature_mode
        self.content_relation_mode = content_relation_mode
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

        if cue_feature_mode == "all":
            cue_in_channels = 4
        elif cue_feature_mode == "phase_only":
            cue_in_channels = 2
        else:
            cue_in_channels = 3

        self.cue_encoder = LiteCueEncoder(
            in_channels=cue_in_channels,
            cue_bands=lite_cue_bands,
            temporal_hidden_dim=lite_cue_hidden_dim,
            out_dim=cue_encoder_out_dim,
            kernel_size=lite_cue_kernel_size,
            dropout=dropout,
            encoder_type=lite_cue_encoder_type,
        )

        if content_relation_mode == "mean_diff_absdiff":
            content_relation_dim = encoder_out_dim * 3
        elif content_relation_mode == "mean_diff":
            content_relation_dim = encoder_out_dim * 2
        else:
            content_relation_dim = encoder_out_dim
        self.content_fusion = nn.Sequential(
            nn.Linear(content_relation_dim, content_fusion_dim),
            nn.LayerNorm(content_fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        if self.use_cross_ear_interaction:
            self.cross_rl = nn.Linear(encoder_out_dim, encoder_out_dim)
            self.cross_lr = nn.Linear(encoder_out_dim, encoder_out_dim)
            self.cross_norm_l = nn.LayerNorm(encoder_out_dim)
            self.cross_norm_r = nn.LayerNorm(encoder_out_dim)
        else:
            self.cross_rl = None
            self.cross_lr = None
            self.cross_norm_l = None
            self.cross_norm_r = None

        temporal_input_dim = content_fusion_dim + cue_encoder_out_dim
        self.fusion_norm = nn.LayerNorm(temporal_input_dim)
        self.fusion_dropout = nn.Dropout(dropout)

        self.temporal_head = TemporalHead(
            input_dim=temporal_input_dim,
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

    def _build_cue_tensor(
        self,
        ild: torch.Tensor,
        ipd_sin: torch.Tensor,
        ipd_cos: torch.Tensor,
        coherence: torch.Tensor,
    ) -> torch.Tensor:
        if self.cue_feature_mode == "phase_only":
            return torch.stack([ipd_sin, ipd_cos], dim=1)
        if self.cue_feature_mode == "ild_phase":
            return torch.stack([ild, ipd_sin, ipd_cos], dim=1)
        return torch.stack([ild, ipd_sin, ipd_cos, coherence], dim=1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        log_mag_L = batch["log_mag_L"]
        log_mag_R = batch["log_mag_R"]
        ild = batch["ild"]
        ipd = batch["ipd"]

        if self.content_input_mode == "logmag":
            left_content = log_mag_L.unsqueeze(1)
            right_content = log_mag_R.unsqueeze(1)
        else:
            left_content = torch.stack([batch["spec_real_L"], batch["spec_imag_L"]], dim=1)
            right_content = torch.stack([batch["spec_real_R"], batch["spec_imag_R"]], dim=1)

        f_l = self.encoder(left_content)
        f_r = self.encoder(right_content)

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
        if self.content_relation_mode == "mean_diff_absdiff":
            abs_diff_feat = diff_feat.abs()
            content_feat = torch.cat([mean_feat, diff_feat, abs_diff_feat], dim=-1)
        elif self.content_relation_mode == "mean_diff":
            content_feat = torch.cat([mean_feat, diff_feat], dim=-1)
        else:
            content_feat = diff_feat
        content_feat = self.content_fusion(content_feat)

        cue_tensor = self._build_cue_tensor(ild, ipd_sin, ipd_cos, coherence)
        cue_feat = self.cue_encoder(cue_tensor)

        fused = torch.cat([content_feat, cue_feat], dim=-1)
        fused = self.fusion_norm(fused)
        fused = self.fusion_dropout(fused)

        outputs = self.temporal_head(fused)
        outputs["cue_feat"] = cue_feat
        outputs["fused_feat"] = fused
        outputs["content_feat"] = content_feat
        return outputs


class NativeLiteDualCueConcatDOANet(nn.Module):
    """内容流保持 encoder v2，cue 流拆成 value/reliability 双分支。"""

    def __init__(
        self,
        freq_bins: int = 257,
        encoder_channels=None,
        encoder_out_dim: int = 96,
        encoder_variant: str = "v2_balanced",
        content_input_mode: str = "logmag",
        content_relation_mode: str = "mean_diff_absdiff",
        content_fusion_dim: int = 80,
        lite_cue_bands: int = 16,
        lite_cue_hidden_dim: int = 48,
        cue_value_out_dim: int = 24,
        cue_reliability_out_dim: int = 8,
        lite_cue_kernel_size: int = 3,
        lite_cue_encoder_type: str = "temporal_conv",
        dual_cue_fusion_mode: str = "concat",
        use_cross_ear_interaction: bool = False,
        gru_hidden_size: int = 80,
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
            encoder_channels = [24, 40, 64]
        if content_input_mode not in {"logmag", "complex_ri"}:
            raise ValueError(f"Unsupported content_input_mode: {content_input_mode}")
        if content_relation_mode not in {"mean_diff_absdiff", "mean_diff", "diff_only"}:
            raise ValueError(f"Unsupported content_relation_mode: {content_relation_mode}")

        self.content_input_mode = content_input_mode
        self.content_relation_mode = content_relation_mode
        self.use_cross_ear_interaction = use_cross_ear_interaction
        self.dual_cue_fusion_mode = dual_cue_fusion_mode

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

        self.cue_encoder = DualBranchCueEncoder(
            cue_bands=lite_cue_bands,
            temporal_hidden_dim=lite_cue_hidden_dim,
            value_out_dim=cue_value_out_dim,
            reliability_out_dim=cue_reliability_out_dim,
            kernel_size=lite_cue_kernel_size,
            dropout=dropout,
            encoder_type=lite_cue_encoder_type,
            fusion_mode=dual_cue_fusion_mode,
        )
        cue_encoder_out_dim = cue_value_out_dim if dual_cue_fusion_mode == "gate" else cue_value_out_dim + cue_reliability_out_dim

        if content_relation_mode == "mean_diff_absdiff":
            content_relation_dim = encoder_out_dim * 3
        elif content_relation_mode == "mean_diff":
            content_relation_dim = encoder_out_dim * 2
        else:
            content_relation_dim = encoder_out_dim
        self.content_fusion = nn.Sequential(
            nn.Linear(content_relation_dim, content_fusion_dim),
            nn.LayerNorm(content_fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        if self.use_cross_ear_interaction:
            self.cross_rl = nn.Linear(encoder_out_dim, encoder_out_dim)
            self.cross_lr = nn.Linear(encoder_out_dim, encoder_out_dim)
            self.cross_norm_l = nn.LayerNorm(encoder_out_dim)
            self.cross_norm_r = nn.LayerNorm(encoder_out_dim)
        else:
            self.cross_rl = None
            self.cross_lr = None
            self.cross_norm_l = None
            self.cross_norm_r = None

        temporal_input_dim = content_fusion_dim + cue_encoder_out_dim
        self.fusion_norm = nn.LayerNorm(temporal_input_dim)
        self.fusion_dropout = nn.Dropout(dropout)

        self.temporal_head = TemporalHead(
            input_dim=temporal_input_dim,
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

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        log_mag_L = batch["log_mag_L"]
        log_mag_R = batch["log_mag_R"]
        ild = batch["ild"]
        ipd = batch["ipd"]

        if self.content_input_mode == "logmag":
            left_content = log_mag_L.unsqueeze(1)
            right_content = log_mag_R.unsqueeze(1)
        else:
            left_content = torch.stack([batch["spec_real_L"], batch["spec_imag_L"]], dim=1)
            right_content = torch.stack([batch["spec_real_R"], batch["spec_imag_R"]], dim=1)

        f_l = self.encoder(left_content)
        f_r = self.encoder(right_content)

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
        if self.content_relation_mode == "mean_diff_absdiff":
            abs_diff_feat = diff_feat.abs()
            content_feat = torch.cat([mean_feat, diff_feat, abs_diff_feat], dim=-1)
        elif self.content_relation_mode == "mean_diff":
            content_feat = torch.cat([mean_feat, diff_feat], dim=-1)
        else:
            content_feat = diff_feat
        content_feat = self.content_fusion(content_feat)

        value_tensor = torch.stack([ild, ipd_sin, ipd_cos], dim=1)
        reliability_tensor = coherence.unsqueeze(1)
        cue_outputs = self.cue_encoder(value_tensor, reliability_tensor)
        cue_feat = cue_outputs["cue_feat"]

        fused = torch.cat([content_feat, cue_feat], dim=-1)
        fused = self.fusion_norm(fused)
        fused = self.fusion_dropout(fused)

        outputs = self.temporal_head(fused)
        outputs["cue_feat"] = cue_feat
        outputs["fused_feat"] = fused
        outputs["content_feat"] = content_feat
        outputs["cue_value_feat"] = cue_outputs["cue_value_feat"]
        outputs["cue_reliability_feat"] = cue_outputs["cue_reliability_feat"]
        if cue_outputs["cue_gate"] is not None:
            outputs["cue_gate"] = cue_outputs["cue_gate"]
        return outputs
