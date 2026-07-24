"""FN-SSL style baseline adapted to the current 72-class KEMAR protocol.

Core idea borrowed from:
Full-Band and Narrow-Band Fusion for Sound Source Localization.

This adaptation keeps the model backbone style:
    4-channel input -> stacked FN blocks -> temporal pooling -> classifier

But it uses the current project batch interface and output protocol.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class FNBlock(nn.Module):
    """One full-band / narrow-band fusion block."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        dropout: float = 0.2,
        is_online: bool = True,
        is_first: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.full_hidden_size = hidden_size // 2
        self.narr_hidden_size = hidden_size if is_online else hidden_size // 2
        self.is_online = bool(is_online)
        self.is_first = bool(is_first)

        self.dropout_full = nn.Dropout(p=dropout)
        self.dropout_narr = nn.Dropout(p=dropout)

        self.full_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.full_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        narr_input_size = 2 * self.full_hidden_size + self.input_size if self.is_first else 2 * self.full_hidden_size
        self.narr_lstm = nn.LSTM(
            input_size=narr_input_size,
            hidden_size=self.narr_hidden_size,
            batch_first=True,
            bidirectional=not self.is_online,
        )

    def forward(
        self,
        x: torch.Tensor,
        nb_skip: torch.Tensor | None = None,
        fb_skip: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, T, F, C]
        bsz, steps, freq, _ = x.shape

        local_skip = x.permute(0, 2, 1, 3).reshape(bsz * freq, steps, -1)

        x_fb = x.reshape(bsz * steps, freq, -1)
        if fb_skip is not None:
            x_fb = x_fb + fb_skip
        x_fb, _ = self.full_lstm(x_fb)
        fb_skip = x_fb
        x_fb = self.dropout_full(x_fb)

        x_nb = x_fb.view(bsz, steps, freq, -1).permute(0, 2, 1, 3).reshape(bsz * freq, steps, -1)
        if self.is_first:
            x_nb = torch.cat((x_nb, local_skip), dim=-1)
        else:
            assert nb_skip is not None
            x_nb = x_nb + nb_skip

        x_nb, _ = self.narr_lstm(x_nb)
        nb_skip = x_nb
        x_nb = self.dropout_narr(x_nb)
        x_out = x_nb.view(bsz, freq, steps, -1).permute(0, 2, 1, 3)
        return x_out, fb_skip, nb_skip


class FNSSLBaseline(nn.Module):
    """FN-SSL style 72-class binaural DOA classifier."""

    def __init__(
        self,
        hidden_size: int = 256,
        dropout: float = 0.2,
        num_classes: int = 72,
        use_front_back_auxiliary: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.use_front_back_auxiliary = bool(use_front_back_auxiliary)

        input_size = 4
        self.block_1 = FNBlock(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            is_online=True,
            is_first=True,
        )
        self.block_2 = FNBlock(
            input_size=hidden_size,
            hidden_size=hidden_size,
            dropout=dropout,
            is_online=True,
            is_first=False,
        )
        self.block_3 = FNBlock(
            input_size=hidden_size,
            hidden_size=hidden_size,
            dropout=dropout,
            is_online=True,
            is_first=False,
        )

        self.time_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_size, self.num_classes)
        if self.use_front_back_auxiliary:
            self.front_back_classifier = nn.Linear(hidden_size, 2)

    @staticmethod
    def _build_input(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # [B, T, F] x 4 -> [B, 4, T, F]
        log_mag_l = batch["log_mag_L"]
        log_mag_r = batch["log_mag_R"]
        ipd_sin = batch["ipd_sin"]
        ipd_cos = batch["ipd_cos"]
        return torch.stack([log_mag_l, log_mag_r, ipd_sin, ipd_cos], dim=1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = self._build_input(batch)          # [B, 4, T, F]
        x = x.permute(0, 2, 3, 1)            # [B, T, F, 4]

        x, fb_skip, nb_skip = self.block_1(x)
        x, fb_skip, nb_skip = self.block_2(x, nb_skip=nb_skip, fb_skip=fb_skip)
        x, fb_skip, nb_skip = self.block_3(x, nb_skip=nb_skip, fb_skip=fb_skip)

        # x: [B, T, F, H]
        x = x.mean(dim=2)                    # [B, T, H]
        feat = self.time_pool(x.transpose(1, 2)).squeeze(-1)  # [B, H]
        feat = self.head(feat)
        logits = self.classifier(feat)

        out = {"logits": logits}
        if self.use_front_back_auxiliary:
            out["front_back_logits"] = self.front_back_classifier(feat)
        return out
