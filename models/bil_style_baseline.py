"""BiL GCC-PHAT CRN baselines adapted to the current project's protocol.

This module follows the official "Binaural Localization for Speech in Noise"
repository more closely than the earlier lightweight baseline:
1. Use GCC-PHAT as the only explicit binaural input feature.
2. Use 3 Conv2d blocks with shared channel width, then a GRU.
3. Apply tanh to the GRU output and optionally multiply forward/backward states.
4. Adapt only the output head to the current 72-class DOA protocol.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Tuple[int, int] = (3, 3),
        pool_size: Tuple[int, int] | None = (1, 2),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        pad = (kernel_size[0] // 2, kernel_size[1] // 2)
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
        ]
        if pool_size is not None:
            layers.append(nn.MaxPool2d(kernel_size=pool_size))
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BiLStyleGCCPHATCRNBaseline(nn.Module):
    def __init__(
        self,
        freq_bins: int = 257,
        gcc_bins: int = 64,
        cnn_channels: Sequence[int] = (128, 128, 128),
        f_pool_size: Sequence[int] = (2, 2, 2),
        t_pool_size: Sequence[int] = (1, 1, 1),
        kernel_size: Tuple[int, int] = (3, 3),
        dropout: float = 0.0,
        gru_hidden_size: int = 128,
        gru_num_layers: int = 2,
        bidirectional: bool = False,
        mlp_hidden_size: int = 128,
        num_classes: int = 72,
        use_front_back_auxiliary: bool = False,
    ) -> None:
        super().__init__()
        self.freq_bins = int(freq_bins)
        self.n_fft = (self.freq_bins - 1) * 2
        self.gcc_bins = int(gcc_bins)
        self.num_classes = int(num_classes)
        self.bidirectional = bool(bidirectional)
        self.use_front_back_auxiliary = bool(use_front_back_auxiliary)

        cnn_channels = list(cnn_channels)
        f_pool_size = list(f_pool_size)
        t_pool_size = list(t_pool_size)
        assert len(cnn_channels) == len(f_pool_size) == len(t_pool_size)

        blocks = []
        in_ch = 1
        reduced_gcc_bins = self.gcc_bins
        for out_ch, fp, tp in zip(cnn_channels, f_pool_size, t_pool_size):
            blocks.append(
                BiLConvBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=kernel_size,
                    pool_size=(tp, fp),
                    dropout=dropout,
                )
            )
            reduced_gcc_bins = max(1, reduced_gcc_bins // fp)
            in_ch = out_ch
        self.cnn = nn.Sequential(*blocks)

        rnn_input_dim = cnn_channels[-1] * reduced_gcc_bins
        self.gru = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if gru_num_layers > 1 else 0.0,
        )

        fused_dim = gru_hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, mlp_hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(mlp_hidden_size, num_classes)
        if self.use_front_back_auxiliary:
            self.front_back_classifier = nn.Linear(mlp_hidden_size, 2)

    def _build_gcc_phat(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute GCC-PHAT from cached complex spectra.

        Input spectra are [B, T, F]. Output is [B, 1, T, gcc_bins].
        """
        spec_l = torch.complex(batch["spec_real_L"], batch["spec_imag_L"])  # [B, T, F]
        spec_r = torch.complex(batch["spec_real_R"], batch["spec_imag_R"])  # [B, T, F]
        cross = spec_l * torch.conj(spec_r)
        cross = cross / cross.abs().clamp_min(1e-8)

        gcc = torch.fft.irfft(cross, n=self.n_fft, dim=-1)  # [B, T, n_fft]
        gcc = torch.fft.fftshift(gcc, dim=-1)

        center = gcc.shape[-1] // 2
        half = self.gcc_bins // 2
        start = max(0, center - half)
        end = start + self.gcc_bins
        gcc = gcc[..., start:end]
        if gcc.shape[-1] < self.gcc_bins:
            gcc = F.pad(gcc, (0, self.gcc_bins - gcc.shape[-1]))

        gcc = gcc.unsqueeze(1)  # [B, 1, T, gcc_bins]
        return gcc

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = self._build_gcc_phat(batch)           # [B, 1, T, G]
        x = self.cnn(x)                           # [B, C, T', G']
        x = x.permute(0, 2, 1, 3).contiguous()   # [B, T', C, G']
        bsz, steps, ch, gcc_bins = x.shape
        x = x.view(bsz, steps, ch * gcc_bins)     # [B, T', C*G']

        x, _ = self.gru(x)                        # [B, T', H*dir]
        if self.bidirectional:
            h = x.shape[-1] // 2
            x = torch.tanh(x[:, :, :h]) * torch.tanh(x[:, :, h:])
        else:
            x = torch.tanh(x)

        feat_seq = self.mlp(x)                    # [B, T', D]
        logits_seq = self.classifier(feat_seq)    # [B, T', C]
        logits = logits_seq.mean(dim=1)           # Official BiL uses output_mode="mean".
        out = {
            "logits": logits,
            "gcc_phat": x,
            "frame_logits": logits_seq,
            "sequence_feat": feat_seq,
        }
        if self.use_front_back_auxiliary:
            out["front_back_logits"] = self.front_back_classifier(feat_seq).mean(dim=1)
        return out


class BiLStyleGCCPHATCRNSequenceBaseline(nn.Module):
    """Moving sequence variant of the BiL-style GCC-PHAT CRN baseline."""

    def __init__(
        self,
        freq_bins: int = 257,
        gcc_bins: int = 64,
        cnn_channels: Sequence[int] = (128, 128, 128),
        f_pool_size: Sequence[int] = (2, 2, 2),
        t_pool_size: Sequence[int] = (1, 1, 1),
        kernel_size: Tuple[int, int] = (3, 3),
        dropout: float = 0.0,
        gru_hidden_size: int = 128,
        gru_num_layers: int = 2,
        bidirectional: bool = False,
        num_classes: int = 72,
        label_steps: int = 40,
        use_front_back_auxiliary: bool = False,
    ) -> None:
        super().__init__()
        self.freq_bins = int(freq_bins)
        self.n_fft = (self.freq_bins - 1) * 2
        self.gcc_bins = int(gcc_bins)
        self.num_classes = int(num_classes)
        self.bidirectional = bool(bidirectional)
        self.label_steps = int(label_steps)
        self.use_front_back_auxiliary = bool(use_front_back_auxiliary)

        cnn_channels = list(cnn_channels)
        f_pool_size = list(f_pool_size)
        t_pool_size = list(t_pool_size)
        assert len(cnn_channels) == len(f_pool_size) == len(t_pool_size)

        blocks = []
        in_ch = 1
        reduced_gcc_bins = self.gcc_bins
        for out_ch, fp, tp in zip(cnn_channels, f_pool_size, t_pool_size):
            blocks.append(
                BiLConvBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=kernel_size,
                    pool_size=(tp, fp),
                    dropout=dropout,
                )
            )
            reduced_gcc_bins = max(1, reduced_gcc_bins // fp)
            in_ch = out_ch
        self.cnn = nn.Sequential(*blocks)

        rnn_input_dim = cnn_channels[-1] * reduced_gcc_bins
        self.gru = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if gru_num_layers > 1 else 0.0,
        )

        seq_feat_dim = gru_hidden_size
        self.temporal_pool = nn.AdaptiveAvgPool1d(self.label_steps)
        self.mlp = nn.Sequential(
            nn.Linear(seq_feat_dim, gru_hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(gru_hidden_size, num_classes)
        if self.use_front_back_auxiliary:
            self.front_back_classifier = nn.Linear(gru_hidden_size, 2)

    def _build_gcc_phat(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        spec_l = torch.complex(batch["spec_real_L"], batch["spec_imag_L"])
        spec_r = torch.complex(batch["spec_real_R"], batch["spec_imag_R"])
        cross = spec_l * torch.conj(spec_r)
        cross = cross / cross.abs().clamp_min(1e-8)

        gcc = torch.fft.irfft(cross, n=self.n_fft, dim=-1)
        gcc = torch.fft.fftshift(gcc, dim=-1)

        center = gcc.shape[-1] // 2
        half = self.gcc_bins // 2
        start = max(0, center - half)
        end = start + self.gcc_bins
        gcc = gcc[..., start:end]
        if gcc.shape[-1] < self.gcc_bins:
            gcc = F.pad(gcc, (0, self.gcc_bins - gcc.shape[-1]))
        return gcc.unsqueeze(1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = self._build_gcc_phat(batch)
        x = self.cnn(x)
        x = x.permute(0, 2, 1, 3).contiguous()
        bsz, steps, ch, gcc_bins = x.shape
        x = x.view(bsz, steps, ch * gcc_bins)

        x, _ = self.gru(x)
        if self.bidirectional:
            h = x.shape[-1] // 2
            x = torch.tanh(x[:, :, :h]) * torch.tanh(x[:, :, h:])
        else:
            x = torch.tanh(x)

        x = self.temporal_pool(x.transpose(1, 2)).transpose(1, 2)  # [B, S, H]
        feat = self.mlp(x)
        logits = self.classifier(feat)
        out = {"doa_logits": logits, "logits": logits, "gcc_phat": x, "sequence_feat": feat}
        if self.use_front_back_auxiliary:
            out["front_back_logits"] = self.front_back_classifier(feat)
        return out
