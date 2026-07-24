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

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import (
    BinauralEncoder,
    BinauralEncoderV2Balanced,
    BandwiseBinauralEncoderV2,
    LightContentEncoderV1,
)
from models.temporal_head import TemporalHead, TemporalHeadMulMLP


def _hz_to_erb(freq_hz: torch.Tensor) -> torch.Tensor:
    return 21.4 * torch.log10(1.0 + 0.00437 * freq_hz)


def _erb_to_hz(erb: torch.Tensor) -> torch.Tensor:
    return (torch.pow(10.0, erb / 21.4) - 1.0) / 0.00437


def _erb_centers(num_bands: int, low_hz: float, high_hz: float) -> torch.Tensor:
    low = torch.tensor(float(low_hz))
    high = torch.tensor(float(high_hz))
    erb_points = torch.linspace(_hz_to_erb(low), _hz_to_erb(high), steps=num_bands)
    return _erb_to_hz(erb_points)


def _piecewise_erb_centers(parts) -> torch.Tensor:
    centers = []
    for idx, (num_bands, low_hz, high_hz) in enumerate(parts):
        part = _erb_centers(int(num_bands), float(low_hz), float(high_hz))
        if idx > 0 and part.numel() > 1:
            part = part[1:]
        centers.append(part)
    return torch.cat(centers, dim=0)


def _triangular_filterbank(
    centers_hz: torch.Tensor,
    freq_bins: int,
    sample_rate: int,
) -> torch.Tensor:
    freqs = torch.linspace(0.0, sample_rate / 2.0, steps=freq_bins)
    centers = torch.sort(centers_hz.float().clamp(0.0, sample_rate / 2.0))[0]
    if centers.numel() < 1:
        raise ValueError("At least one auditory band center is required")
    if centers.numel() == 1:
        edges = torch.tensor([0.0, sample_rate / 2.0])
    else:
        mids = 0.5 * (centers[:-1] + centers[1:])
        edges = torch.cat([torch.tensor([0.0]), mids, torch.tensor([sample_rate / 2.0])])

    filters = []
    for idx, center in enumerate(centers):
        left = edges[idx]
        right = edges[idx + 1]
        filt = torch.zeros_like(freqs)
        if center > left:
            left_mask = (freqs >= left) & (freqs <= center)
            filt[left_mask] = (freqs[left_mask] - left) / (center - left).clamp_min(1e-6)
        if right > center:
            right_mask = (freqs >= center) & (freqs <= right)
            filt[right_mask] = (right - freqs[right_mask]) / (right - center).clamp_min(1e-6)
        if filt.sum() <= 0:
            nearest = torch.argmin(torch.abs(freqs - center))
            filt[nearest] = 1.0
        filters.append(filt / filt.sum().clamp_min(1e-6))
    return torch.stack(filters, dim=0)


def _build_auditory_filterbank(
    mode: str,
    num_bands: int,
    freq_bins: int,
    sample_rate: int,
    in_channels: int,
) -> torch.Tensor:
    if mode == "erb":
        centers = _erb_centers(num_bands, 50.0, sample_rate / 2.0)
        return _triangular_filterbank(centers, freq_bins, sample_rate)
    if mode == "cue_specific_value":
        if in_channels != 3:
            raise ValueError("cue_specific_value expects value cues [ILD, sin(IPD), cos(IPD)]")
        ild_centers = _piecewise_erb_centers([
            (5, 50.0, 1000.0),
            (20, 1000.0, sample_rate / 2.0),
        ])[:num_bands]
        ipd_centers = _piecewise_erb_centers([
            (19, 50.0, 1500.0),
            (6, 1500.0, sample_rate / 2.0),
        ])[:num_bands]
        ild_fb = _triangular_filterbank(ild_centers, freq_bins, sample_rate)
        ipd_fb = _triangular_filterbank(ipd_centers, freq_bins, sample_rate)
        return torch.stack([ild_fb, ipd_fb, ipd_fb], dim=0)
    if mode == "cue_specific_reliability":
        if in_channels != 1:
            raise ValueError("cue_specific_reliability expects a single coherence channel")
        centers = _erb_centers(num_bands, 50.0, sample_rate / 2.0)
        return _triangular_filterbank(centers, freq_bins, sample_rate)
    raise ValueError(f"Unsupported auditory filterbank mode: {mode}")


def _init_learnable_filterbank_logits(filterbank: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Initialize learnable band projection logits from a fixed filterbank."""
    return torch.log(filterbank.float().clamp_min(eps))


class LiteCueEncoder(nn.Module):
    """轻量 cue encoder：先做频带压缩，再做时间维 1D 卷积。"""

    def __init__(
        self,
        in_channels: int,
        cue_bands: int = 16,
        freq_bins: int = 257,
        sample_rate: int = 16000,
        band_mode: str = "uniform",
        temporal_hidden_dim: int = 48,
        out_dim: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.2,
        encoder_type: str = "temporal_conv",
    ):
        super().__init__()
        self.cue_bands = cue_bands
        self.encoder_type = encoder_type
        self.band_mode = band_mode
        self.freq_bins = freq_bins
        self.learnable_band_projection = band_mode.startswith("learnable_")
        fixed_band_mode = band_mode[len("learnable_"):] if self.learnable_band_projection else band_mode
        if fixed_band_mode == "uniform":
            self.register_buffer("band_filterbank", None, persistent=False)
        else:
            band_filterbank = _build_auditory_filterbank(
                fixed_band_mode,
                cue_bands,
                freq_bins,
                sample_rate,
                in_channels,
            )
            if self.learnable_band_projection:
                self.band_filterbank_logits = nn.Parameter(
                    _init_learnable_filterbank_logits(band_filterbank)
                )
                self.register_buffer("band_filterbank", None, persistent=False)
            else:
                self.register_buffer("band_filterbank", band_filterbank, persistent=False)
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
        elif encoder_type == "temporal_conv_bandmix":
            padding = kernel_size // 2
            bandmix_hidden = max(temporal_hidden_dim // 2, 8)
            self.bandmix_net = nn.Sequential(
                nn.Conv1d(in_channels, bandmix_hidden, kernel_size=3, padding=1),
                nn.BatchNorm1d(bandmix_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(bandmix_hidden, in_channels, kernel_size=1),
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
        elif encoder_type == "temporal_conv_dwbandmix":
            padding = kernel_size // 2
            self.bandmix_net = nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels, in_channels, kernel_size=1),
                nn.Dropout(dropout),
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
        elif encoder_type == "temporal_conv_res":
            padding = kernel_size // 2
            self.residual_proj = nn.Conv1d(flat_dim, out_dim, kernel_size=1)
            self.temporal_net = nn.Sequential(
                nn.Conv1d(flat_dim, temporal_hidden_dim, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(temporal_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(temporal_hidden_dim, out_dim, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_dim),
            )
            self.residual_activation = nn.ReLU(inplace=True)
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
        if self.band_mode == "uniform":
            x = cue_tensor.reshape(bsz * num_cues * time_steps, 1, freq_bins)
            x = F.adaptive_avg_pool1d(x, self.cue_bands)
            x = x.reshape(bsz, num_cues, time_steps, self.cue_bands)
        elif self.learnable_band_projection:
            if freq_bins != self.freq_bins:
                raise ValueError(
                    f"Learnable cue encoder expected {self.freq_bins} frequency bins, got {freq_bins}"
                )
            band_weight = torch.softmax(self.band_filterbank_logits, dim=-1).to(
                device=cue_tensor.device,
                dtype=cue_tensor.dtype,
            )
            if band_weight.dim() == 2:
                x = torch.einsum("bctf,kf->bctk", cue_tensor, band_weight)
            else:
                x = torch.einsum("bctf,ckf->bctk", cue_tensor, band_weight)
        else:
            if freq_bins != self.freq_bins:
                raise ValueError(
                    f"Auditory cue encoder expected {self.freq_bins} frequency bins, got {freq_bins}"
                )
            band_filterbank = self.band_filterbank.to(device=cue_tensor.device, dtype=cue_tensor.dtype)
            if band_filterbank.dim() == 2:
                x = torch.einsum("bctf,kf->bctk", cue_tensor, band_filterbank)
            else:
                x = torch.einsum("bctf,ckf->bctk", cue_tensor, band_filterbank)
        x = x.permute(0, 2, 1, 3)  # [B, T, C, bands]
        if self.encoder_type == "temporal_conv_bandattn":
            x_flat = x.reshape(bsz, time_steps, num_cues * self.cue_bands)
            band_logits = self.band_gate(x_flat)  # [B, T, bands]
            band_weight = torch.softmax(band_logits, dim=-1).unsqueeze(2)  # [B, T, 1, bands]
            x = x * band_weight
        if self.encoder_type in {"temporal_conv_bandmix", "temporal_conv_dwbandmix"}:
            x_band = x.reshape(bsz * time_steps, num_cues, self.cue_bands)
            x_band = x_band + self.bandmix_net(x_band)
            x = x_band.reshape(bsz, time_steps, num_cues, self.cue_bands)
        x = x.reshape(bsz, time_steps, num_cues * self.cue_bands)
        if self.encoder_type == "temporal_conv":
            x = x.transpose(1, 2)  # [B, C*bands, T]
            x = self.temporal_net(x)
            return x.transpose(1, 2)  # [B, T, out_dim]
        if self.encoder_type == "temporal_conv_bandattn":
            x = x.transpose(1, 2)  # [B, C*bands, T]
            x = self.temporal_net(x)
            return x.transpose(1, 2)  # [B, T, out_dim]
        if self.encoder_type == "temporal_conv_bandmix":
            x = x.transpose(1, 2)  # [B, C*bands, T]
            x = self.temporal_net(x)
            return x.transpose(1, 2)  # [B, T, out_dim]
        if self.encoder_type == "temporal_conv_dwbandmix":
            x = x.transpose(1, 2)  # [B, C*bands, T]
            x = self.temporal_net(x)
            return x.transpose(1, 2)  # [B, T, out_dim]
        if self.encoder_type == "temporal_conv_res":
            x = x.transpose(1, 2)  # [B, C*bands, T]
            residual = self.residual_proj(x)
            x = self.temporal_net(x)
            x = self.residual_activation(x + residual)
            return x.transpose(1, 2)  # [B, T, out_dim]
        if self.encoder_type == "temporal_conv_ms":
            x = x.transpose(1, 2)  # [B, C*bands, T]
            x3 = self.temporal_branch_k3(x)
            x5 = self.temporal_branch_k5(x)
            x = torch.cat([x3, x5], dim=1)
            x = self.temporal_fuse(x)
            return x.transpose(1, 2)  # [B, T, out_dim]
        return self.temporal_net(x)  # [B, T, out_dim]


class LocalTFCueEncoder(nn.Module):
    """SDEL-inspired local time-frequency cue encoder.

    It keeps the cue map on the STFT time-frequency grid, learns local T-F
    patterns with small 2D convolutions, pools only along frequency, and emits a
    compact per-frame cue representation for the existing temporal head.
    """

    def __init__(
        self,
        freq_bins: int,
        out_dim: int = 32,
        cnn_channels: Sequence[int] | None = None,
        f_pool_size: Sequence[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [16, 24, 32]
        if f_pool_size is None:
            f_pool_size = [4, 4, 4]
        cnn_channels = list(cnn_channels)
        f_pool_size = list(f_pool_size)
        if len(cnn_channels) != len(f_pool_size):
            raise ValueError("cnn_channels and f_pool_size must have the same length")

        padding = kernel_size // 2
        in_ch = 4
        reduced_freq_bins = int(freq_bins)
        blocks = []
        for out_ch, f_pool in zip(cnn_channels, f_pool_size):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=(1, int(f_pool))),
                    nn.Dropout2d(dropout),
                )
            )
            reduced_freq_bins = max(1, reduced_freq_bins // int(f_pool))
            in_ch = out_ch
        self.cnn = nn.Sequential(*blocks)
        self.proj = nn.Sequential(
            nn.Linear(cnn_channels[-1] * reduced_freq_bins, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, cue_tensor: torch.Tensor) -> torch.Tensor:
        # cue_tensor: [B, 4, T, F]
        x = self.cnn(cue_tensor)
        x = x.permute(0, 2, 1, 3).contiguous()
        bsz, time_steps, channels, freq_bins = x.shape
        x = x.view(bsz, time_steps, channels * freq_bins)
        return self.proj(x)


class DualBranchCueEncoder(nn.Module):
    """双分支 cue encoder：
    - value branch 处理 ILD / sin(IPD) / cos(IPD)
    - reliability branch 处理 coherence
    第一版先用 concat 融合，保留更强的可解释性。
    """

    def __init__(
        self,
        cue_bands: int = 16,
        cue_freq_bins: int = 257,
        cue_sample_rate: int = 16000,
        cue_band_mode: str = "uniform",
        temporal_hidden_dim: int = 48,
        value_out_dim: int = 24,
        reliability_out_dim: int = 8,
        cue_ild_bands: int = 16,
        cue_ipd_bands: int = 32,
        cue_coherence_bands: int = 16,
        cue_ild_out_dim: int = 8,
        cue_ipd_out_dim: int = 16,
        kernel_size: int = 3,
        dropout: float = 0.2,
        encoder_type: str = "temporal_conv",
        value_encoder_type: str | None = None,
        reliability_encoder_type: str | None = None,
        fusion_mode: str = "concat",
        reliability_weight_scale: float = 0.5,
        branch_mode: str = "dual",
        disable_reliability_branch: bool = False,
        use_tf_mask: bool = False,
        tf_mask_hidden_channels: int = 8,
        tf_mask_residual_scale: float = 1.0,
    ):
        super().__init__()
        if fusion_mode not in {"concat", "gate", "reliability_weighted_concat", "rel_film_value"}:
            raise ValueError(f"Unsupported DualBranchCueEncoder fusion_mode: {fusion_mode}")
        if branch_mode not in {
            "dual",
            "merged",
            "cue_specific_resolution",
            "local_tf",
            "dual_local_tf",
            "dual_local_tf_gate",
        }:
            raise ValueError(f"Unsupported DualBranchCueEncoder branch_mode: {branch_mode}")
        if cue_band_mode not in {"uniform", "erb", "cue_specific", "learnable_cue_specific"}:
            raise ValueError(f"Unsupported cue_band_mode: {cue_band_mode}")
        if branch_mode == "merged" and fusion_mode == "gate":
            raise ValueError("branch_mode='merged' does not support fusion_mode='gate'")
        if disable_reliability_branch and fusion_mode == "gate":
            raise ValueError("disable_reliability_branch=True does not support fusion_mode='gate'")
        if cue_band_mode in {"cue_specific", "learnable_cue_specific"} and branch_mode in {"merged", "local_tf", "dual_local_tf", "dual_local_tf_gate"}:
            raise ValueError(f"cue_band_mode='{cue_band_mode}' requires dual cue branches")
        if branch_mode == "cue_specific_resolution" and fusion_mode != "concat":
            raise ValueError("branch_mode='cue_specific_resolution' only supports fusion_mode='concat'")
        if branch_mode == "local_tf" and fusion_mode != "concat":
            raise ValueError("branch_mode='local_tf' only supports fusion_mode='concat'")
        if fusion_mode == "reliability_weighted_concat" and disable_reliability_branch:
            raise ValueError("reliability_weighted_concat requires reliability branch")
        if fusion_mode == "rel_film_value" and disable_reliability_branch:
            raise ValueError("rel_film_value requires reliability branch")

        self.fusion_mode = fusion_mode
        self.reliability_weight_scale = reliability_weight_scale
        self.branch_mode = branch_mode
        self.disable_reliability_branch = disable_reliability_branch
        self.cue_band_mode = cue_band_mode
        self.use_tf_mask = use_tf_mask
        self.tf_mask_residual_scale = tf_mask_residual_scale
        self.cue_ild_bands = cue_ild_bands
        self.cue_ipd_bands = cue_ipd_bands
        self.cue_coherence_bands = cue_coherence_bands
        self.cue_ild_out_dim = cue_ild_out_dim
        self.cue_ipd_out_dim = cue_ipd_out_dim
        self.value_out_dim = value_out_dim
        self.reliability_out_dim = reliability_out_dim
        value_encoder_type = value_encoder_type or encoder_type
        reliability_encoder_type = reliability_encoder_type or encoder_type
        value_band_mode = {
            "uniform": "uniform",
            "erb": "erb",
            "cue_specific": "cue_specific_value",
            "learnable_cue_specific": "learnable_cue_specific_value",
        }[cue_band_mode]
        reliability_band_mode = {
            "uniform": "uniform",
            "erb": "erb",
            "cue_specific": "cue_specific_reliability",
            "learnable_cue_specific": "learnable_cue_specific_reliability",
        }[cue_band_mode]
        if use_tf_mask:
            self.tf_mask_net = nn.Sequential(
                nn.Conv2d(4, tf_mask_hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(tf_mask_hidden_channels, 1, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.tf_mask_net = None
        if branch_mode == "local_tf":
            self.local_tf_encoder = LocalTFCueEncoder(
                freq_bins=cue_freq_bins,
                out_dim=value_out_dim + reliability_out_dim,
                cnn_channels=[16, 24, 32],
                f_pool_size=[4, 4, 4],
                kernel_size=kernel_size,
                dropout=dropout,
            )
            self.merged_encoder = None
            self.value_encoder = None
            self.reliability_encoder = None
            self.ild_encoder = None
            self.ipd_encoder = None
        elif branch_mode in {"dual_local_tf", "dual_local_tf_gate"}:
            dual_out_dim = value_out_dim if disable_reliability_branch or fusion_mode == "gate" else value_out_dim + reliability_out_dim
            self.local_tf_encoder = LocalTFCueEncoder(
                freq_bins=cue_freq_bins,
                out_dim=value_out_dim + reliability_out_dim if branch_mode == "dual_local_tf" else dual_out_dim,
                cnn_channels=[16, 24, 32] if branch_mode == "dual_local_tf" else [8, 12, 16],
                f_pool_size=[4, 4, 4],
                kernel_size=kernel_size,
                dropout=dropout,
            )
            self.merged_encoder = None
            self.ild_encoder = None
            self.ipd_encoder = None
            self.value_encoder = LiteCueEncoder(
                in_channels=3,
                cue_bands=cue_bands,
                freq_bins=cue_freq_bins,
                sample_rate=cue_sample_rate,
                band_mode=value_band_mode,
                temporal_hidden_dim=temporal_hidden_dim,
                out_dim=value_out_dim,
                kernel_size=kernel_size,
                dropout=dropout,
                encoder_type=value_encoder_type,
            )
            self.reliability_encoder = None if disable_reliability_branch else LiteCueEncoder(
                in_channels=1,
                cue_bands=cue_bands,
                freq_bins=cue_freq_bins,
                sample_rate=cue_sample_rate,
                band_mode=reliability_band_mode,
                temporal_hidden_dim=max(temporal_hidden_dim // 2, 8),
                out_dim=reliability_out_dim,
                kernel_size=kernel_size,
                dropout=dropout,
                encoder_type=reliability_encoder_type,
            )
        elif branch_mode == "merged":
            self.local_tf_encoder = None
            merged_out_dim = value_out_dim + reliability_out_dim
            self.merged_encoder = LiteCueEncoder(
                in_channels=4,
                cue_bands=cue_bands,
                freq_bins=cue_freq_bins,
                sample_rate=cue_sample_rate,
                band_mode="erb" if cue_band_mode == "erb" else "uniform",
                temporal_hidden_dim=temporal_hidden_dim,
                out_dim=merged_out_dim,
                kernel_size=kernel_size,
                dropout=dropout,
                encoder_type=encoder_type,
            )
            self.value_encoder = None
            self.reliability_encoder = None
            self.ild_encoder = None
            self.ipd_encoder = None
        elif branch_mode == "cue_specific_resolution":
            self.local_tf_encoder = None
            self.merged_encoder = None
            self.value_encoder = None
            self.ild_encoder = LiteCueEncoder(
                in_channels=1,
                cue_bands=cue_ild_bands,
                freq_bins=cue_freq_bins,
                sample_rate=cue_sample_rate,
                band_mode="uniform",
                temporal_hidden_dim=max(temporal_hidden_dim // 2, 8),
                out_dim=cue_ild_out_dim,
                kernel_size=kernel_size,
                dropout=dropout,
                encoder_type=encoder_type,
            )
            self.ipd_encoder = LiteCueEncoder(
                in_channels=2,
                cue_bands=cue_ipd_bands,
                freq_bins=cue_freq_bins,
                sample_rate=cue_sample_rate,
                band_mode="uniform",
                temporal_hidden_dim=temporal_hidden_dim,
                out_dim=cue_ipd_out_dim,
                kernel_size=kernel_size,
                dropout=dropout,
                encoder_type=value_encoder_type,
            )
            self.reliability_encoder = None if disable_reliability_branch else LiteCueEncoder(
                in_channels=1,
                cue_bands=cue_coherence_bands,
                freq_bins=cue_freq_bins,
                sample_rate=cue_sample_rate,
                band_mode="uniform",
                temporal_hidden_dim=max(temporal_hidden_dim // 2, 8),
                out_dim=reliability_out_dim,
                kernel_size=kernel_size,
                dropout=dropout,
                encoder_type=reliability_encoder_type,
            )
        else:
            self.local_tf_encoder = None
            self.merged_encoder = None
            self.ild_encoder = None
            self.ipd_encoder = None
            self.value_encoder = LiteCueEncoder(
                in_channels=3,
                cue_bands=cue_bands,
                freq_bins=cue_freq_bins,
                sample_rate=cue_sample_rate,
                band_mode=value_band_mode,
                temporal_hidden_dim=temporal_hidden_dim,
                out_dim=value_out_dim,
                kernel_size=kernel_size,
                dropout=dropout,
                encoder_type=value_encoder_type,
            )
            self.reliability_encoder = None if disable_reliability_branch else LiteCueEncoder(
                in_channels=1,
                cue_bands=cue_bands,
                freq_bins=cue_freq_bins,
                sample_rate=cue_sample_rate,
                band_mode=reliability_band_mode,
                temporal_hidden_dim=max(temporal_hidden_dim // 2, 8),
                out_dim=reliability_out_dim,
                kernel_size=kernel_size,
                dropout=dropout,
                encoder_type=reliability_encoder_type,
            )
        if fusion_mode == "gate" and not disable_reliability_branch:
            self.rel_to_gate = nn.Sequential(
                nn.Linear(reliability_out_dim, value_out_dim),
                nn.Sigmoid(),
            )
        else:
            self.rel_to_gate = None
        if fusion_mode == "reliability_weighted_concat" and not disable_reliability_branch:
            self.rel_to_value_weight = nn.Sequential(
                nn.Linear(reliability_out_dim, value_out_dim),
                nn.Sigmoid(),
            )
        else:
            self.rel_to_value_weight = None
        if fusion_mode == "rel_film_value" and not disable_reliability_branch:
            self.rel_to_film = nn.Linear(reliability_out_dim, value_out_dim * 2)
        else:
            self.rel_to_film = None

    @property
    def out_dim(self) -> int:
        if self.branch_mode == "merged":
            return self.merged_encoder.temporal_net[-2].num_features if isinstance(self.merged_encoder.temporal_net, nn.Sequential) and hasattr(self.merged_encoder.temporal_net[-2], "num_features") else None
        if self.branch_mode == "local_tf":
            return self.local_tf_encoder.out_dim
        if self.branch_mode == "dual_local_tf":
            dual_dim = self.value_encoder.temporal_net[-2].num_features if isinstance(self.value_encoder.temporal_net, nn.Sequential) and hasattr(self.value_encoder.temporal_net[-2], "num_features") else 0
            if not self.disable_reliability_branch:
                dual_dim += self.reliability_encoder.temporal_net[-2].num_features if isinstance(self.reliability_encoder.temporal_net, nn.Sequential) and hasattr(self.reliability_encoder.temporal_net[-2], "num_features") else 0
            return dual_dim + self.local_tf_encoder.out_dim
        if self.branch_mode == "dual_local_tf_gate":
            if self.disable_reliability_branch or self.fusion_mode == "gate":
                return self.value_encoder.temporal_net[-2].num_features if isinstance(self.value_encoder.temporal_net, nn.Sequential) and hasattr(self.value_encoder.temporal_net[-2], "num_features") else None
            return (
                self.value_encoder.temporal_net[-2].num_features + self.reliability_encoder.temporal_net[-2].num_features
                if isinstance(self.value_encoder.temporal_net, nn.Sequential)
                and hasattr(self.value_encoder.temporal_net[-2], "num_features")
                and isinstance(self.reliability_encoder.temporal_net, nn.Sequential)
                and hasattr(self.reliability_encoder.temporal_net[-2], "num_features")
                else None
            )
        if self.disable_reliability_branch:
            return self.value_encoder.temporal_net[-2].num_features if isinstance(self.value_encoder.temporal_net, nn.Sequential) and hasattr(self.value_encoder.temporal_net[-2], "num_features") else None
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
        if self.branch_mode == "merged":
            merged_tensor = torch.cat([value_tensor, reliability_tensor], dim=1)
            cue_feat = self.merged_encoder(merged_tensor)
            return {
                "cue_feat": cue_feat,
                "cue_value_feat": cue_feat,
                "cue_reliability_feat": None,
                "cue_gate": None,
                "cue_tf_mask": None,
            }
        if self.branch_mode == "local_tf":
            cue_tensor = torch.cat([value_tensor, reliability_tensor], dim=1)
            cue_feat = self.local_tf_encoder(cue_tensor)
            return {
                "cue_feat": cue_feat,
                "cue_value_feat": cue_feat,
                "cue_reliability_feat": None,
                "cue_gate": None,
                "cue_tf_mask": None,
            }

        if self.use_tf_mask and not self.disable_reliability_branch:
            mask_input = torch.cat([value_tensor, reliability_tensor], dim=1)
            tf_mask = self.tf_mask_net(mask_input)
            value_tensor = value_tensor * (1.0 + self.tf_mask_residual_scale * tf_mask)
        else:
            tf_mask = None
        if self.branch_mode == "cue_specific_resolution":
            ild_tensor = value_tensor[:, 0:1]
            ipd_tensor = value_tensor[:, 1:3]
            ild_feat = self.ild_encoder(ild_tensor)
            ipd_feat = self.ipd_encoder(ipd_tensor)
            value_feat = torch.cat([ild_feat, ipd_feat], dim=-1)
            if self.disable_reliability_branch:
                reliability_feat = None
                cue_feat = value_feat
            else:
                reliability_feat = self.reliability_encoder(reliability_tensor)
                cue_feat = torch.cat([value_feat, reliability_feat], dim=-1)
            return {
                "cue_feat": cue_feat,
                "cue_value_feat": value_feat,
                "cue_reliability_feat": reliability_feat,
                "cue_gate": None,
                "cue_tf_mask": tf_mask,
            }
        value_feat = self.value_encoder(value_tensor)
        if self.disable_reliability_branch:
            reliability_feat = None
            gate = None
            cue_feat = value_feat
        else:
            reliability_feat = self.reliability_encoder(reliability_tensor)
            if self.fusion_mode == "gate":
                gate = self.rel_to_gate(reliability_feat)
                cue_feat = value_feat * gate
            elif self.fusion_mode == "reliability_weighted_concat":
                gate = self.rel_to_value_weight(reliability_feat)
                value_scale = 1.0 + self.reliability_weight_scale * (2.0 * gate - 1.0)
                cue_feat = torch.cat([value_feat * value_scale, reliability_feat], dim=-1)
            elif self.fusion_mode == "rel_film_value":
                film = self.rel_to_film(reliability_feat)
                scale, bias = film.chunk(2, dim=-1)
                scale = 0.1 * torch.tanh(scale)
                bias = 0.1 * torch.tanh(bias)
                value_feat = value_feat * (1.0 + scale) + bias
                gate = scale
                cue_feat = torch.cat([value_feat, reliability_feat], dim=-1)
            else:
                gate = None
                cue_feat = torch.cat([value_feat, reliability_feat], dim=-1)
        if self.branch_mode == "dual_local_tf":
            cue_tensor = torch.cat([value_tensor, reliability_tensor], dim=1)
            local_tf_feat = self.local_tf_encoder(cue_tensor)
            cue_feat = torch.cat([cue_feat, local_tf_feat], dim=-1)
        elif self.branch_mode == "dual_local_tf_gate":
            cue_tensor = torch.cat([value_tensor, reliability_tensor], dim=1)
            local_tf_gate = torch.sigmoid(self.local_tf_encoder(cue_tensor))
            cue_feat = cue_feat * (0.75 + 0.5 * local_tf_gate)
            gate = local_tf_gate
        return {
            "cue_feat": cue_feat,
            "cue_value_feat": value_feat,
            "cue_reliability_feat": reliability_feat,
            "cue_gate": gate,
            "cue_tf_mask": tf_mask,
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
        temporal_encoder_type: str = "gru",
        mamba_num_layers: int = 2,
        mamba_state_dim: int = 16,
        mamba_expand_factor: int = 2,
        mamba_conv_kernel: int = 4,
        gru_dropout: float = 0.1,
        num_classes: int = 72,
        azimuth_range=(-180.0, 180.0),
        dropout: float = 0.2,
        use_attention_pooling: bool = True,
        attention_pooling_variant: str = "default",
        use_front_back_auxiliary: bool = True,
        use_regression: bool = False,
        use_pure_regression: bool = False,
        temporal_head_type: str = "default",
        temporal_mlp_hidden_dim: int = 128,
        temporal_mlp_num_layers: int = 2,
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
            temporal_encoder_type=temporal_encoder_type,
            mamba_num_layers=mamba_num_layers,
            mamba_state_dim=mamba_state_dim,
            mamba_expand_factor=mamba_expand_factor,
            mamba_conv_kernel=mamba_conv_kernel,
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
        temporal_encoder_type: str = "gru",
        mamba_num_layers: int = 2,
        mamba_state_dim: int = 16,
        mamba_expand_factor: int = 2,
        mamba_conv_kernel: int = 4,
        gru_dropout: float = 0.1,
        num_classes: int = 72,
        azimuth_range=(-180.0, 180.0),
        dropout: float = 0.2,
        use_attention_pooling: bool = True,
        use_front_back_auxiliary: bool = True,
        use_regression: bool = False,
        use_pure_regression: bool = False,
        temporal_head_type: str = "default",
        temporal_mlp_hidden_dim: int = 128,
        temporal_mlp_num_layers: int = 2,
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

        if temporal_head_type == "gru_mul_mlp":
            self.temporal_head = TemporalHeadMulMLP(
                input_dim=temporal_input_dim,
                gru_hidden_size=gru_hidden_size,
                gru_num_layers=gru_num_layers,
                num_classes=num_classes,
                gru_dropout=gru_dropout,
                dropout=dropout,
                mlp_hidden_dim=temporal_mlp_hidden_dim,
                mlp_num_layers=temporal_mlp_num_layers,
                use_front_back_auxiliary=use_front_back_auxiliary,
            )
        elif temporal_head_type == "default":
            self.temporal_head = TemporalHead(
                input_dim=temporal_input_dim,
                gru_hidden_size=gru_hidden_size,
                gru_num_layers=gru_num_layers,
                temporal_encoder_type=temporal_encoder_type,
                mamba_num_layers=mamba_num_layers,
                mamba_state_dim=mamba_state_dim,
                mamba_expand_factor=mamba_expand_factor,
                mamba_conv_kernel=mamba_conv_kernel,
                num_classes=num_classes,
                gru_dropout=gru_dropout,
                dropout=dropout,
                use_regression=use_regression,
                use_pure_regression=use_pure_regression,
                use_attention_pooling=use_attention_pooling,
                attention_pooling_variant=attention_pooling_variant,
                use_front_back_auxiliary=use_front_back_auxiliary,
                azimuth_range=tuple(azimuth_range),
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
        content_encoder_type: str = "shared_2dcnn",
        content_encoder_num_bands: int = 4,
        content_encoder_band_out_dim: int = 24,
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
        temporal_encoder_type: str = "gru",
        mamba_num_layers: int = 2,
        mamba_state_dim: int = 16,
        mamba_expand_factor: int = 2,
        mamba_conv_kernel: int = 4,
        gru_dropout: float = 0.1,
        num_classes: int = 72,
        azimuth_range=(-180.0, 180.0),
        dropout: float = 0.2,
        use_attention_pooling: bool = True,
        attention_pooling_variant: str = "default",
        use_front_back_auxiliary: bool = True,
        use_regression: bool = False,
        use_pure_regression: bool = False,
        temporal_head_type: str = "default",
        temporal_mlp_hidden_dim: int = 128,
        temporal_mlp_num_layers: int = 2,
    ):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [24, 40, 64]

        if content_input_mode not in {"logmag", "complex_ri"}:
            raise ValueError(f"Unsupported content_input_mode: {content_input_mode}")
        if cue_feature_mode not in {"all", "phase_only", "ild_phase"}:
            raise ValueError(f"Unsupported cue_feature_mode: {cue_feature_mode}")
        if content_relation_mode not in {"mean_diff_absdiff", "mean_diff", "diff_only", "raw_concat", "learned_cross_attn"}:
            raise ValueError(f"Unsupported content_relation_mode: {content_relation_mode}")

        self.content_input_mode = content_input_mode
        self.cue_feature_mode = cue_feature_mode
        self.content_relation_mode = content_relation_mode
        self.content_encoder_type = content_encoder_type
        self.use_cross_ear_interaction = use_cross_ear_interaction

        content_in_channels = 1 if content_input_mode == "logmag" else 2
        if content_encoder_type == "shared_2dcnn":
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
        elif content_encoder_type == "lite_v1":
            self.encoder = LightContentEncoderV1(
                in_channels=content_in_channels,
                channels=encoder_channels,
                out_dim=encoder_out_dim,
                dropout=dropout,
            )
        elif content_encoder_type == "bandwise_v2":
            if encoder_variant != "v2_balanced":
                raise ValueError(
                    "content_encoder_type=bandwise_v2 currently requires encoder_variant=v2_balanced"
                )
            self.encoder = BandwiseBinauralEncoderV2(
                in_channels=content_in_channels,
                channels=encoder_channels,
                out_dim=encoder_out_dim,
                dropout=dropout,
                num_bands=content_encoder_num_bands,
                band_out_dim=content_encoder_band_out_dim,
            )
        else:
            raise ValueError(f"Unsupported content_encoder_type: {content_encoder_type}")

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
        elif content_relation_mode == "raw_concat":
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

        if temporal_head_type == "gru_mul_mlp":
            self.temporal_head = TemporalHeadMulMLP(
                input_dim=temporal_input_dim,
                gru_hidden_size=gru_hidden_size,
                gru_num_layers=gru_num_layers,
                num_classes=num_classes,
                gru_dropout=gru_dropout,
                dropout=dropout,
                mlp_hidden_dim=temporal_mlp_hidden_dim,
                mlp_num_layers=temporal_mlp_num_layers,
                use_front_back_auxiliary=use_front_back_auxiliary,
            )
        elif temporal_head_type == "default":
            self.temporal_head = TemporalHead(
                input_dim=temporal_input_dim,
                gru_hidden_size=gru_hidden_size,
                gru_num_layers=gru_num_layers,
                temporal_encoder_type=temporal_encoder_type,
                mamba_num_layers=mamba_num_layers,
                mamba_state_dim=mamba_state_dim,
                mamba_expand_factor=mamba_expand_factor,
                mamba_conv_kernel=mamba_conv_kernel,
                num_classes=num_classes,
                gru_dropout=gru_dropout,
                dropout=dropout,
                use_regression=use_regression,
                use_pure_regression=use_pure_regression,
                use_attention_pooling=use_attention_pooling,
                attention_pooling_variant=attention_pooling_variant,
                use_front_back_auxiliary=use_front_back_auxiliary,
                azimuth_range=tuple(azimuth_range),
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
        elif self.content_relation_mode == "raw_concat":
            content_feat = torch.cat([f_l, f_r], dim=-1)
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
        content_encoder_type: str = "shared_2dcnn",
        content_encoder_num_bands: int = 4,
        content_encoder_band_out_dim: int = 24,
        content_input_mode: str = "logmag",
        content_relation_mode: str = "mean_diff_absdiff",
        content_fusion_dim: int = 80,
        lite_cue_bands: int = 16,
        cue_band_mode: str = "uniform",
        cue_sample_rate: int = 16000,
        lite_cue_hidden_dim: int = 48,
        cue_value_out_dim: int = 24,
        cue_reliability_out_dim: int = 8,
        cue_ild_bands: int = 16,
        cue_ipd_bands: int = 32,
        cue_coherence_bands: int = 16,
        cue_ild_out_dim: int = 8,
        cue_ipd_out_dim: int = 16,
        lite_cue_kernel_size: int = 3,
        lite_cue_encoder_type: str = "temporal_conv",
        cue_value_encoder_type: str | None = None,
        cue_reliability_encoder_type: str | None = None,
        dual_cue_fusion_mode: str = "concat",
        dual_cue_reliability_weight_scale: float = 0.5,
        cue_branch_mode: str = "dual",
        disable_reliability_branch: bool = False,
        dual_cue_use_tf_mask: bool = False,
        dual_cue_tf_mask_hidden_channels: int = 8,
        dual_cue_tf_mask_residual_scale: float = 1.0,
        disable_content_stream: bool = False,
        use_cross_ear_interaction: bool = False,
        gru_hidden_size: int = 80,
        gru_num_layers: int = 1,
        temporal_encoder_type: str = "gru",
        mamba_num_layers: int = 2,
        mamba_state_dim: int = 16,
        mamba_expand_factor: int = 2,
        mamba_conv_kernel: int = 4,
        gru_dropout: float = 0.1,
        num_classes: int = 72,
        azimuth_range=(-180.0, 180.0),
        dropout: float = 0.2,
        use_attention_pooling: bool = True,
        attention_pooling_variant: str = "default",
        use_front_back_auxiliary: bool = True,
        use_regression: bool = False,
        use_pure_regression: bool = False,
        temporal_head_type: str = "default",
        temporal_mlp_hidden_dim: int = 128,
        temporal_mlp_num_layers: int = 2,
    ):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [24, 40, 64]
        if content_input_mode not in {"logmag", "complex_ri"}:
            raise ValueError(f"Unsupported content_input_mode: {content_input_mode}")
        if content_relation_mode not in {"mean_diff_absdiff", "mean_diff", "diff_only", "raw_concat", "learned_cross_attn"}:
            raise ValueError(f"Unsupported content_relation_mode: {content_relation_mode}")

        self.content_input_mode = content_input_mode
        self.content_relation_mode = content_relation_mode
        self.use_cross_ear_interaction = use_cross_ear_interaction
        self.dual_cue_fusion_mode = dual_cue_fusion_mode
        self.content_encoder_type = content_encoder_type
        self.disable_reliability_branch = disable_reliability_branch
        self.disable_content_stream = disable_content_stream
        self.cue_branch_mode = cue_branch_mode
        self.cue_band_mode = cue_band_mode

        content_in_channels = 1 if content_input_mode == "logmag" else 2
        if disable_content_stream:
            self.encoder = None
        elif content_encoder_type == "shared_2dcnn":
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
        elif content_encoder_type == "lite_v1":
            self.encoder = LightContentEncoderV1(
                in_channels=content_in_channels,
                channels=encoder_channels,
                out_dim=encoder_out_dim,
                dropout=dropout,
            )
        elif content_encoder_type == "bandwise_v2":
            if encoder_variant != "v2_balanced":
                raise ValueError(
                    "content_encoder_type=bandwise_v2 currently requires encoder_variant=v2_balanced"
                )
            self.encoder = BandwiseBinauralEncoderV2(
                in_channels=content_in_channels,
                channels=encoder_channels,
                out_dim=encoder_out_dim,
                dropout=dropout,
                num_bands=content_encoder_num_bands,
                band_out_dim=content_encoder_band_out_dim,
            )
        else:
            raise ValueError(f"Unsupported content_encoder_type: {content_encoder_type}")

        self.cue_encoder = DualBranchCueEncoder(
            cue_bands=lite_cue_bands,
            cue_freq_bins=freq_bins,
            cue_sample_rate=cue_sample_rate,
            cue_band_mode=cue_band_mode,
            temporal_hidden_dim=lite_cue_hidden_dim,
            value_out_dim=cue_value_out_dim,
            reliability_out_dim=cue_reliability_out_dim,
            cue_ild_bands=cue_ild_bands,
            cue_ipd_bands=cue_ipd_bands,
            cue_coherence_bands=cue_coherence_bands,
            cue_ild_out_dim=cue_ild_out_dim,
            cue_ipd_out_dim=cue_ipd_out_dim,
            kernel_size=lite_cue_kernel_size,
            dropout=dropout,
            encoder_type=lite_cue_encoder_type,
            value_encoder_type=cue_value_encoder_type,
            reliability_encoder_type=cue_reliability_encoder_type,
            fusion_mode=dual_cue_fusion_mode,
            reliability_weight_scale=dual_cue_reliability_weight_scale,
            branch_mode=cue_branch_mode,
            disable_reliability_branch=disable_reliability_branch,
            use_tf_mask=dual_cue_use_tf_mask,
            tf_mask_hidden_channels=dual_cue_tf_mask_hidden_channels,
            tf_mask_residual_scale=dual_cue_tf_mask_residual_scale,
        )
        if cue_branch_mode == "merged":
            cue_encoder_out_dim = cue_value_out_dim + cue_reliability_out_dim
        elif cue_branch_mode == "local_tf":
            cue_encoder_out_dim = cue_value_out_dim + cue_reliability_out_dim
        elif cue_branch_mode == "dual_local_tf":
            if disable_reliability_branch or dual_cue_fusion_mode == "gate":
                cue_encoder_out_dim = cue_value_out_dim
            else:
                cue_encoder_out_dim = cue_value_out_dim + cue_reliability_out_dim
            cue_encoder_out_dim += cue_value_out_dim + cue_reliability_out_dim
        elif cue_branch_mode == "dual_local_tf_gate":
            if disable_reliability_branch or dual_cue_fusion_mode == "gate":
                cue_encoder_out_dim = cue_value_out_dim
            else:
                cue_encoder_out_dim = cue_value_out_dim + cue_reliability_out_dim
        elif cue_branch_mode == "cue_specific_resolution":
            cue_encoder_out_dim = cue_ild_out_dim + cue_ipd_out_dim
            if not disable_reliability_branch:
                cue_encoder_out_dim += cue_reliability_out_dim
        elif disable_reliability_branch or dual_cue_fusion_mode == "gate":
            cue_encoder_out_dim = cue_value_out_dim
        else:
            cue_encoder_out_dim = cue_value_out_dim + cue_reliability_out_dim

        self.content_pair_attn = None
        self.content_pair_norm = None
        if disable_content_stream:
            self.content_fusion = None
            content_out_dim = 0
        else:
            if content_relation_mode == "learned_cross_attn":
                num_heads = 4 if encoder_out_dim % 4 == 0 else 1
                self.content_pair_attn = nn.MultiheadAttention(
                    embed_dim=encoder_out_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                self.content_pair_norm = nn.LayerNorm(encoder_out_dim)
                content_relation_dim = encoder_out_dim * 2
            elif content_relation_mode == "mean_diff_absdiff":
                content_relation_dim = encoder_out_dim * 3
            elif content_relation_mode == "mean_diff":
                content_relation_dim = encoder_out_dim * 2
            elif content_relation_mode == "raw_concat":
                content_relation_dim = encoder_out_dim * 2
            else:
                content_relation_dim = encoder_out_dim
            self.content_fusion = nn.Sequential(
                nn.Linear(content_relation_dim, content_fusion_dim),
                nn.LayerNorm(content_fusion_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            content_out_dim = content_fusion_dim

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

        temporal_input_dim = content_out_dim + cue_encoder_out_dim
        self.fusion_norm = nn.LayerNorm(temporal_input_dim)
        self.fusion_dropout = nn.Dropout(dropout)

        if temporal_head_type == "gru_mul_mlp":
            self.temporal_head = TemporalHeadMulMLP(
                input_dim=temporal_input_dim,
                gru_hidden_size=gru_hidden_size,
                gru_num_layers=gru_num_layers,
                num_classes=num_classes,
                gru_dropout=gru_dropout,
                dropout=dropout,
                mlp_hidden_dim=temporal_mlp_hidden_dim,
                mlp_num_layers=temporal_mlp_num_layers,
                use_front_back_auxiliary=use_front_back_auxiliary,
            )
        elif temporal_head_type == "default":
            self.temporal_head = TemporalHead(
                input_dim=temporal_input_dim,
                gru_hidden_size=gru_hidden_size,
                gru_num_layers=gru_num_layers,
                temporal_encoder_type=temporal_encoder_type,
                mamba_num_layers=mamba_num_layers,
                mamba_state_dim=mamba_state_dim,
                mamba_expand_factor=mamba_expand_factor,
                mamba_conv_kernel=mamba_conv_kernel,
                num_classes=num_classes,
                gru_dropout=gru_dropout,
                dropout=dropout,
                use_regression=use_regression,
                use_pure_regression=use_pure_regression,
                use_attention_pooling=use_attention_pooling,
                attention_pooling_variant=attention_pooling_variant,
                use_front_back_auxiliary=use_front_back_auxiliary,
                azimuth_range=tuple(azimuth_range),
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

        if self.disable_content_stream:
            t_enc = ild.shape[1]
            content_feat = None
        else:
            f_l = self.encoder(left_content)
            f_r = self.encoder(right_content)

            if self.use_cross_ear_interaction:
                cross_l = self.cross_norm_l(self.cross_rl(f_r))
                cross_r = self.cross_norm_r(self.cross_lr(f_l))
                f_l = f_l + cross_l
                f_r = f_r + cross_r

            t_enc = f_l.shape[1]
            mean_feat = 0.5 * (f_l + f_r)
            diff_feat = f_l - f_r
            if self.content_relation_mode == "learned_cross_attn":
                bsz, t_steps, feat_dim = f_l.shape
                ear_tokens = torch.stack([f_l, f_r], dim=2).reshape(bsz * t_steps, 2, feat_dim)
                attended_tokens, _ = self.content_pair_attn(
                    ear_tokens,
                    ear_tokens,
                    ear_tokens,
                    need_weights=False,
                )
                ear_tokens = self.content_pair_norm(ear_tokens + attended_tokens)
                content_feat = ear_tokens.reshape(bsz, t_steps, 2 * feat_dim)
            elif self.content_relation_mode == "mean_diff_absdiff":
                abs_diff_feat = diff_feat.abs()
                content_feat = torch.cat([mean_feat, diff_feat, abs_diff_feat], dim=-1)
            elif self.content_relation_mode == "mean_diff":
                content_feat = torch.cat([mean_feat, diff_feat], dim=-1)
            elif self.content_relation_mode == "raw_concat":
                content_feat = torch.cat([f_l, f_r], dim=-1)
            else:
                content_feat = diff_feat
            content_feat = self.content_fusion(content_feat)

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

        value_tensor = torch.stack([ild, ipd_sin, ipd_cos], dim=1)
        reliability_tensor = coherence.unsqueeze(1)
        cue_outputs = self.cue_encoder(value_tensor, reliability_tensor)
        cue_feat = cue_outputs["cue_feat"]

        fused = cue_feat if self.disable_content_stream else torch.cat([content_feat, cue_feat], dim=-1)
        fused = self.fusion_norm(fused)
        fused = self.fusion_dropout(fused)

        outputs = self.temporal_head(fused)
        outputs["cue_feat"] = cue_feat
        outputs["fused_feat"] = fused
        outputs["content_feat"] = content_feat
        outputs["cue_value_feat"] = cue_outputs["cue_value_feat"]
        outputs["cue_reliability_feat"] = cue_outputs["cue_reliability_feat"]
        outputs["cue_tf_mask"] = cue_outputs["cue_tf_mask"]
        if cue_outputs["cue_gate"] is not None:
            outputs["cue_gate"] = cue_outputs["cue_gate"]
        return outputs


class NativeLiteContentOnlyDOANet(nn.Module):
    """仅使用双耳内容流、不显式使用 ILD/IPD/coherence 的 baseline。"""

    def __init__(
        self,
        freq_bins: int = 257,
        encoder_channels=None,
        encoder_out_dim: int = 96,
        encoder_variant: str = "v2_balanced",
        content_input_mode: str = "logmag",
        content_relation_mode: str = "mean_diff_absdiff",
        content_fusion_dim: int = 80,
        use_cross_ear_interaction: bool = False,
        gru_hidden_size: int = 80,
        gru_num_layers: int = 1,
        temporal_encoder_type: str = "gru",
        mamba_num_layers: int = 2,
        mamba_state_dim: int = 16,
        mamba_expand_factor: int = 2,
        mamba_conv_kernel: int = 4,
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

        self.fusion_norm = nn.LayerNorm(content_fusion_dim)
        self.fusion_dropout = nn.Dropout(dropout)

        self.temporal_head = TemporalHead(
            input_dim=content_fusion_dim,
            gru_hidden_size=gru_hidden_size,
            gru_num_layers=gru_num_layers,
            temporal_encoder_type=temporal_encoder_type,
            mamba_num_layers=mamba_num_layers,
            mamba_state_dim=mamba_state_dim,
            mamba_expand_factor=mamba_expand_factor,
            mamba_conv_kernel=mamba_conv_kernel,
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
        fused = self.fusion_norm(content_feat)
        fused = self.fusion_dropout(fused)

        outputs = self.temporal_head(fused)
        outputs["content_feat"] = content_feat
        outputs["fused_feat"] = fused
        return outputs


class NativeLiteEarlyFusionDOANet(nn.Module):
    """单流早融合 baseline。

    输入特征:
      [mean log-magnitude, ILD, sin(IPD), cos(IPD), coherence]
    先统一送入一个共享 encoder，再经轻量 bottleneck + BiGRU 做分类。
    """

    def __init__(
        self,
        freq_bins: int = 257,
        encoder_channels=None,
        encoder_out_dim: int = 96,
        encoder_variant: str = "v2_balanced",
        early_fusion_dim: int = 80,
        gru_hidden_size: int = 80,
        gru_num_layers: int = 1,
        temporal_encoder_type: str = "gru",
        mamba_num_layers: int = 2,
        mamba_state_dim: int = 16,
        mamba_expand_factor: int = 2,
        mamba_conv_kernel: int = 4,
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

        if encoder_variant == "v1":
            encoder_cls = BinauralEncoder
        elif encoder_variant == "v2_balanced":
            encoder_cls = BinauralEncoderV2Balanced
        else:
            raise ValueError(f"Unsupported encoder_variant: {encoder_variant}")

        # early fusion stack: [mean_logmag, ild, sin(ipd), cos(ipd), coherence]
        self.encoder = encoder_cls(
            in_channels=5,
            channels=encoder_channels,
            out_dim=encoder_out_dim,
            dropout=dropout,
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(encoder_out_dim, early_fusion_dim),
            nn.LayerNorm(early_fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.temporal_head = TemporalHead(
            input_dim=early_fusion_dim,
            gru_hidden_size=gru_hidden_size,
            gru_num_layers=gru_num_layers,
            temporal_encoder_type=temporal_encoder_type,
            mamba_num_layers=mamba_num_layers,
            mamba_state_dim=mamba_state_dim,
            mamba_expand_factor=mamba_expand_factor,
            mamba_conv_kernel=mamba_conv_kernel,
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

        t_ref = min(log_mag_L.shape[1], log_mag_R.shape[1], ild.shape[1], ipd.shape[1])
        log_mag_L = log_mag_L[:, :t_ref, :]
        log_mag_R = log_mag_R[:, :t_ref, :]
        ild = ild[:, :t_ref, :]
        ipd = ipd[:, :t_ref, :]

        ipd_sin = batch.get("ipd_sin")
        ipd_cos = batch.get("ipd_cos")
        coherence = batch.get("coherence")

        if ipd_sin is None:
            ipd_sin = torch.sin(ipd)
        else:
            ipd_sin = ipd_sin[:, :t_ref, :]
        if ipd_cos is None:
            ipd_cos = torch.cos(ipd)
        else:
            ipd_cos = ipd_cos[:, :t_ref, :]
        if coherence is None:
            coherence = torch.ones_like(ild)
        else:
            coherence = coherence[:, :t_ref, :]

        mean_logmag = 0.5 * (log_mag_L + log_mag_R)
        x = torch.stack([mean_logmag, ild, ipd_sin, ipd_cos, coherence], dim=1)  # [B, 5, T, F]
        fused_feat = self.encoder(x)
        fused_feat = self.fusion_proj(fused_feat)
        outputs = self.temporal_head(fused_feat)
        outputs["fused_feat"] = fused_feat
        return outputs
