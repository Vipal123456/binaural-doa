"""SDEL-style CRNN baseline for horizontal-plane DOA regression.

This baseline adapts the core network style of Moving-Binaural-SDEL:
CNN -> BiGRU -> bidirectional multiplicative fusion -> MLP head.

Input features are constructed from the current project batch interface:
- MBMS proxy = 0.5 * (log_mag_L + log_mag_R)
- ILD
- cos(IPD)
- sin(IPD)
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Tuple[int, int] = (3, 3),
        pool_size: Tuple[int, int] = (1, 4),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        pad = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=pool_size),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SDELCRNNBaseline(nn.Module):
    """CRNN baseline with 2D angle-vector regression output."""

    def __init__(
        self,
        freq_bins: int,
        cnn_channels: Sequence[int] = (32, 64, 128),
        f_pool_size: Sequence[int] = (4, 4, 4),
        t_pool_size: Sequence[int] = (1, 1, 1),
        kernel_size: Tuple[int, int] = (3, 3),
        dropout: float = 0.2,
        gru_hidden_size: int = 128,
        gru_num_layers: int = 2,
        fnn_size: int = 128,
        num_fnn_layers: int = 2,
        num_classes: int = 72,
        azimuth_range: Tuple[float, float] = (-180.0, 180.0),
        use_front_back_auxiliary: bool = False,
        output_mode: str = "reg",
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.azimuth_range = tuple(float(v) for v in azimuth_range)
        self.use_front_back_auxiliary = use_front_back_auxiliary
        self.output_mode = output_mode

        cnn_channels = list(cnn_channels)
        f_pool_size = list(f_pool_size)
        t_pool_size = list(t_pool_size)
        assert len(cnn_channels) == len(f_pool_size) == len(t_pool_size)

        in_ch = 4
        blocks = []
        reduced_freq_bins = freq_bins
        for out_ch, f_pool, t_pool in zip(cnn_channels, f_pool_size, t_pool_size):
            blocks.append(
                ConvBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=kernel_size,
                    pool_size=(t_pool, f_pool),
                    dropout=dropout,
                )
            )
            reduced_freq_bins = max(1, reduced_freq_bins // f_pool)
            in_ch = out_ch
        self.cnn = nn.Sequential(*blocks)

        rnn_input_dim = cnn_channels[-1] * reduced_freq_bins
        self.gru = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_num_layers > 1 else 0.0,
        )

        fused_dim = gru_hidden_size
        mlp_layers: list[nn.Module] = []
        in_dim = fused_dim
        for _ in range(max(1, num_fnn_layers)):
            mlp_layers.append(nn.Linear(in_dim, fnn_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            in_dim = fnn_size
        self.mlp = nn.Sequential(*mlp_layers)
        self.angle_head = nn.Linear(in_dim, 2)
        self.classifier = nn.Linear(in_dim, self.num_classes)
        if self.use_front_back_auxiliary:
            self.front_back_classifier = nn.Linear(in_dim, 2)

        bin_centers = torch.linspace(
            self.azimuth_range[0] + (self.azimuth_range[1] - self.azimuth_range[0]) / self.num_classes / 2.0,
            self.azimuth_range[1] - (self.azimuth_range[1] - self.azimuth_range[0]) / self.num_classes / 2.0,
            self.num_classes,
        )
        self.register_buffer("bin_centers_deg", bin_centers)

    @staticmethod
    def _build_input(batch: dict) -> torch.Tensor:
        mbms_proxy = 0.5 * (batch["log_mag_L"] + batch["log_mag_R"])
        ild = batch["ild"]
        ipd_cos = batch["ipd_cos"]
        ipd_sin = batch["ipd_sin"]
        x = torch.stack([mbms_proxy, ild, ipd_cos, ipd_sin], dim=1)
        return x

    def _angle_vec_to_logits(self, angle_vec: torch.Tensor) -> torch.Tensor:
        pred_deg = torch.rad2deg(torch.atan2(angle_vec[:, 0], angle_vec[:, 1]))
        diff = pred_deg.unsqueeze(1) - self.bin_centers_deg.unsqueeze(0)
        diff = torch.remainder(diff + 180.0, 360.0) - 180.0
        logits = -(diff.abs() / 5.0)
        return logits

    def forward(self, batch: dict) -> dict:
        x = self._build_input(batch)            # [B, 4, T, F]
        x = self.cnn(x)                         # [B, C, T', F']
        x = x.permute(0, 2, 1, 3).contiguous() # [B, T', C, F']
        bsz, steps, ch, freq = x.shape
        x = x.view(bsz, steps, ch * freq)       # [B, T', C*F']

        x, _ = self.gru(x)                      # [B, T', 2H]
        h = x.shape[-1] // 2
        x = torch.tanh(x[:, :, :h]) * torch.tanh(x[:, :, h:])  # [B, T', H]
        pooled = x.mean(dim=1)                  # [B, H]
        feat = self.mlp(pooled)                 # [B, D]

        out = {}
        if self.output_mode == "reg":
            angle_vec = self.angle_head(feat)
            angle_vec = torch.nn.functional.normalize(angle_vec, p=2, dim=-1, eps=1e-6)
            logits = self._angle_vec_to_logits(angle_vec)
            out["angle_vec"] = angle_vec
            out["logits"] = logits
        else:
            out["logits"] = self.classifier(feat)

        if self.use_front_back_auxiliary:
            out["front_back_logits"] = self.front_back_classifier(feat)
        return out


class SDELCRNNSequenceBaseline(nn.Module):
    """Moving-speaker sequence variant of the SDEL-style CRNN baseline.

    The static SDEL baseline pools the full utterance into one clip-level
    prediction.  For moving DOA, we keep a fixed label-time axis by pooling the
    CNN-GRU features to ``label_steps`` and applying the same MLP/classifier at
    each step.
    """

    def __init__(
        self,
        freq_bins: int,
        cnn_channels: Sequence[int] = (32, 64, 128),
        f_pool_size: Sequence[int] = (4, 4, 4),
        t_pool_size: Sequence[int] = (1, 1, 1),
        kernel_size: Tuple[int, int] = (3, 3),
        dropout: float = 0.2,
        gru_hidden_size: int = 128,
        gru_num_layers: int = 2,
        fnn_size: int = 128,
        num_fnn_layers: int = 2,
        num_classes: int = 72,
        label_steps: int = 40,
        use_front_back_auxiliary: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.label_steps = int(label_steps)
        self.use_front_back_auxiliary = bool(use_front_back_auxiliary)

        cnn_channels = list(cnn_channels)
        f_pool_size = list(f_pool_size)
        t_pool_size = list(t_pool_size)
        assert len(cnn_channels) == len(f_pool_size) == len(t_pool_size)

        in_ch = 4
        blocks = []
        reduced_freq_bins = freq_bins
        for out_ch, f_pool, t_pool in zip(cnn_channels, f_pool_size, t_pool_size):
            blocks.append(
                ConvBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=kernel_size,
                    pool_size=(t_pool, f_pool),
                    dropout=dropout,
                )
            )
            reduced_freq_bins = max(1, reduced_freq_bins // f_pool)
            in_ch = out_ch
        self.cnn = nn.Sequential(*blocks)

        rnn_input_dim = cnn_channels[-1] * reduced_freq_bins
        self.gru = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_num_layers > 1 else 0.0,
        )
        self.temporal_pool = nn.AdaptiveAvgPool1d(self.label_steps)

        mlp_layers: list[nn.Module] = []
        in_dim = gru_hidden_size
        for _ in range(max(1, num_fnn_layers)):
            mlp_layers.append(nn.Linear(in_dim, fnn_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            in_dim = fnn_size
        self.mlp = nn.Sequential(*mlp_layers)
        self.classifier = nn.Linear(in_dim, self.num_classes)
        if self.use_front_back_auxiliary:
            self.front_back_classifier = nn.Linear(in_dim, 2)

    def forward(self, batch: dict) -> dict:
        x = SDELCRNNBaseline._build_input(batch)  # [B, 4, T, F]
        x = self.cnn(x)                           # [B, C, T', F']
        x = x.permute(0, 2, 1, 3).contiguous()    # [B, T', C, F']
        bsz, steps, ch, freq = x.shape
        x = x.view(bsz, steps, ch * freq)         # [B, T', C*F']

        x, _ = self.gru(x)                        # [B, T', 2H]
        h = x.shape[-1] // 2
        x = torch.tanh(x[:, :, :h]) * torch.tanh(x[:, :, h:])  # [B, T', H]
        x = self.temporal_pool(x.transpose(1, 2)).transpose(1, 2)  # [B, S, H]
        feat = self.mlp(x)                        # [B, S, D]
        logits = self.classifier(feat)            # [B, S, C]

        out = {"doa_logits": logits, "logits": logits, "sequence_feat": feat}
        if self.use_front_back_auxiliary:
            out["front_back_logits"] = self.front_back_classifier(feat)
        return out
