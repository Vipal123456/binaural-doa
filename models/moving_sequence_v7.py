"""V7 moving DOA sequence models."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import BinauralEncoder, BinauralEncoderV2Balanced
from models.native_lite_v7 import DualBranchCueEncoder, LiteCueEncoder


class SequenceTemporalHead(nn.Module):
    """Temporal encoder that keeps the label-time axis and predicts per step."""

    def __init__(
        self,
        input_dim: int,
        label_steps: int = 40,
        gru_hidden_size: int = 80,
        gru_num_layers: int = 1,
        gru_dropout: float = 0.1,
        num_classes: int = 72,
        dropout: float = 0.2,
        use_front_back_auxiliary: bool = False,
    ):
        super().__init__()
        self.label_steps = int(label_steps)
        self.use_front_back_auxiliary = bool(use_front_back_auxiliary)
        self.temporal_pool = nn.AdaptiveAvgPool1d(self.label_steps)
        rnn_dropout = gru_dropout if gru_num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=rnn_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * gru_hidden_size, num_classes)
        if self.use_front_back_auxiliary:
            self.front_back_classifier = nn.Linear(2 * gru_hidden_size, 2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B, T, D] -> [B, 40, D]
        pooled = self.temporal_pool(x.transpose(1, 2)).transpose(1, 2)
        temporal_out, _ = self.gru(pooled)
        feat = self.dropout(temporal_out)
        logits = self.classifier(feat)
        outputs = {"doa_logits": logits, "logits": logits, "sequence_feat": temporal_out}
        if self.use_front_back_auxiliary:
            outputs["front_back_logits"] = self.front_back_classifier(feat)
        return outputs


class SequenceTemporalHeadBeforePoolMul(nn.Module):
    """Run the BiGRU before label-step pooling, then multiply directions.

    This mirrors the SDEL sequence head more closely:
    [B, T, D] -> BiGRU -> tanh(fwd) * tanh(bwd) -> pool -> MLP -> classifier.
    """

    def __init__(
        self,
        input_dim: int,
        label_steps: int = 40,
        gru_hidden_size: int = 128,
        gru_num_layers: int = 2,
        gru_dropout: float = 0.2,
        num_classes: int = 72,
        dropout: float = 0.2,
        mlp_hidden_dim: int = 128,
        mlp_num_layers: int = 2,
        use_front_back_auxiliary: bool = False,
    ):
        super().__init__()
        self.label_steps = int(label_steps)
        self.use_front_back_auxiliary = bool(use_front_back_auxiliary)
        rnn_dropout = gru_dropout if gru_num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=rnn_dropout,
        )
        self.temporal_pool = nn.AdaptiveAvgPool1d(self.label_steps)

        mlp_layers = []
        in_dim = gru_hidden_size
        for _ in range(max(1, int(mlp_num_layers))):
            mlp_layers.append(nn.Linear(in_dim, mlp_hidden_dim))
            mlp_layers.append(nn.ReLU(inplace=True))
            mlp_layers.append(nn.Dropout(dropout))
            in_dim = mlp_hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)
        self.classifier = nn.Linear(in_dim, num_classes)
        if self.use_front_back_auxiliary:
            self.front_back_classifier = nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        temporal_out, _ = self.gru(x)
        h = temporal_out.shape[-1] // 2
        mul_feat = torch.tanh(temporal_out[:, :, :h]) * torch.tanh(temporal_out[:, :, h:])
        pooled = self.temporal_pool(mul_feat.transpose(1, 2)).transpose(1, 2)
        feat = self.mlp(pooled)
        logits = self.classifier(feat)
        outputs = {"doa_logits": logits, "logits": logits, "sequence_feat": feat}
        if self.use_front_back_auxiliary:
            outputs["front_back_logits"] = self.front_back_classifier(feat)
        return outputs


class MovingDualCueSequenceDOANet(nn.Module):
    """Moving version of the v7 content + cue value/reliability architecture."""

    def __init__(
        self,
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
        label_steps: int = 40,
        gru_hidden_size: int = 80,
        gru_num_layers: int = 1,
        gru_dropout: float = 0.1,
        num_classes: int = 72,
        dropout: float = 0.2,
        temporal_head_type: str = "pool_before_gru",
        temporal_mlp_hidden_dim: int = 128,
        temporal_mlp_num_layers: int = 2,
        use_front_back_auxiliary: bool = False,
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

        encoder_cls = BinauralEncoderV2Balanced if encoder_variant == "v2_balanced" else BinauralEncoder
        content_in_channels = 1 if content_input_mode == "logmag" else 2
        self.encoder = encoder_cls(
            in_channels=content_in_channels,
            channels=encoder_channels,
            out_dim=encoder_out_dim,
            dropout=dropout,
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
        cue_dim = cue_value_out_dim if dual_cue_fusion_mode == "gate" else cue_value_out_dim + cue_reliability_out_dim
        self.fusion_norm = nn.LayerNorm(content_fusion_dim + cue_dim)
        self.fusion_dropout = nn.Dropout(dropout)
        temporal_input_dim = content_fusion_dim + cue_dim
        if temporal_head_type == "gru_before_pool_mul":
            self.temporal_head = SequenceTemporalHeadBeforePoolMul(
                input_dim=temporal_input_dim,
                label_steps=label_steps,
                gru_hidden_size=gru_hidden_size,
                gru_num_layers=gru_num_layers,
                gru_dropout=gru_dropout,
                num_classes=num_classes,
                dropout=dropout,
                mlp_hidden_dim=temporal_mlp_hidden_dim,
                mlp_num_layers=temporal_mlp_num_layers,
                use_front_back_auxiliary=use_front_back_auxiliary,
            )
        elif temporal_head_type == "pool_before_gru":
            self.temporal_head = SequenceTemporalHead(
                input_dim=temporal_input_dim,
                label_steps=label_steps,
                gru_hidden_size=gru_hidden_size,
                gru_num_layers=gru_num_layers,
                gru_dropout=gru_dropout,
                num_classes=num_classes,
                dropout=dropout,
                use_front_back_auxiliary=use_front_back_auxiliary,
            )
        else:
            raise ValueError(f"Unsupported temporal_head_type: {temporal_head_type}")

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
        t_enc = f_l.shape[1]
        ild = ild[:, :t_enc, :]
        ipd_sin = batch.get("ipd_sin")
        ipd_cos = batch.get("ipd_cos")
        coherence = batch.get("coherence")
        ipd_sin = torch.sin(ipd[:, :t_enc, :]) if ipd_sin is None else ipd_sin[:, :t_enc, :]
        ipd_cos = torch.cos(ipd[:, :t_enc, :]) if ipd_cos is None else ipd_cos[:, :t_enc, :]
        coherence = torch.ones_like(ild) if coherence is None else coherence[:, :t_enc, :]

        mean_feat = 0.5 * (f_l + f_r)
        diff_feat = f_l - f_r
        if self.content_relation_mode == "mean_diff_absdiff":
            content_rel = torch.cat([mean_feat, diff_feat, diff_feat.abs()], dim=-1)
        elif self.content_relation_mode == "mean_diff":
            content_rel = torch.cat([mean_feat, diff_feat], dim=-1)
        else:
            content_rel = diff_feat
        content_feat = self.content_fusion(content_rel)

        value_tensor = torch.stack([ild, ipd_sin, ipd_cos], dim=1)
        reliability_tensor = coherence.unsqueeze(1)
        cue_outputs = self.cue_encoder(value_tensor, reliability_tensor)
        fused = torch.cat([content_feat, cue_outputs["cue_feat"]], dim=-1)
        fused = self.fusion_dropout(self.fusion_norm(fused))
        outputs = self.temporal_head(fused)
        outputs["fused_feat"] = fused
        outputs["content_feat"] = content_feat
        outputs["cue_feat"] = cue_outputs["cue_feat"]
        return outputs


class MovingLiteCueSequenceDOANet(nn.Module):
    """Moving version of the lite cue concat architecture."""

    def __init__(
        self,
        encoder_channels=None,
        encoder_out_dim: int = 96,
        encoder_variant: str = "v2_balanced",
        content_input_mode: str = "logmag",
        cue_feature_mode: str = "all",
        content_relation_mode: str = "mean_diff_absdiff",
        content_fusion_dim: int = 80,
        lite_cue_bands: int = 16,
        lite_cue_hidden_dim: int = 48,
        cue_encoder_out_dim: int = 32,
        lite_cue_kernel_size: int = 3,
        lite_cue_encoder_type: str = "temporal_conv",
        use_cross_ear_interaction: bool = False,
        label_steps: int = 40,
        gru_hidden_size: int = 80,
        gru_num_layers: int = 1,
        gru_dropout: float = 0.1,
        num_classes: int = 72,
        dropout: float = 0.2,
        temporal_head_type: str = "pool_before_gru",
        temporal_mlp_hidden_dim: int = 128,
        temporal_mlp_num_layers: int = 2,
        use_front_back_auxiliary: bool = False,
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

        encoder_cls = BinauralEncoderV2Balanced if encoder_variant == "v2_balanced" else BinauralEncoder
        content_in_channels = 1 if content_input_mode == "logmag" else 2
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
        if temporal_head_type == "gru_before_pool_mul":
            self.temporal_head = SequenceTemporalHeadBeforePoolMul(
                input_dim=temporal_input_dim,
                label_steps=label_steps,
                gru_hidden_size=gru_hidden_size,
                gru_num_layers=gru_num_layers,
                gru_dropout=gru_dropout,
                num_classes=num_classes,
                dropout=dropout,
                mlp_hidden_dim=temporal_mlp_hidden_dim,
                mlp_num_layers=temporal_mlp_num_layers,
                use_front_back_auxiliary=use_front_back_auxiliary,
            )
        elif temporal_head_type == "pool_before_gru":
            self.temporal_head = SequenceTemporalHead(
                input_dim=temporal_input_dim,
                label_steps=label_steps,
                gru_hidden_size=gru_hidden_size,
                gru_num_layers=gru_num_layers,
                gru_dropout=gru_dropout,
                num_classes=num_classes,
                dropout=dropout,
                use_front_back_auxiliary=use_front_back_auxiliary,
            )
        else:
            raise ValueError(f"Unsupported temporal_head_type: {temporal_head_type}")

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
        ipd_sin = torch.sin(ipd[:, :t_enc, :]) if ipd_sin is None else ipd_sin[:, :t_enc, :]
        ipd_cos = torch.cos(ipd[:, :t_enc, :]) if ipd_cos is None else ipd_cos[:, :t_enc, :]
        coherence = torch.ones_like(ild) if coherence is None else coherence[:, :t_enc, :]

        mean_feat = 0.5 * (f_l + f_r)
        diff_feat = f_l - f_r
        if self.content_relation_mode == "mean_diff_absdiff":
            content_feat = torch.cat([mean_feat, diff_feat, diff_feat.abs()], dim=-1)
        elif self.content_relation_mode == "mean_diff":
            content_feat = torch.cat([mean_feat, diff_feat], dim=-1)
        else:
            content_feat = diff_feat
        content_feat = self.content_fusion(content_feat)

        cue_tensor = self._build_cue_tensor(ild, ipd_sin, ipd_cos, coherence)
        cue_feat = self.cue_encoder(cue_tensor)

        fused = torch.cat([content_feat, cue_feat], dim=-1)
        fused = self.fusion_dropout(self.fusion_norm(fused))
        outputs = self.temporal_head(fused)
        outputs["cue_feat"] = cue_feat
        outputs["fused_feat"] = fused
        outputs["content_feat"] = content_feat
        return outputs
