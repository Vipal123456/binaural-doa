"""FAViT-style binaural cue Transformer baseline.

核心思想：
1. 仅使用显式双耳 cue（ILD + IPD 或 ILD + sin/cos(IPD)）
2. 先压缩时间维，再沿频率方向切 vertical patches
3. 在频率优先 token 序列上做轻量 Transformer 建模
4. 输出整段级 DOA 分类结果
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FAViTStyleBaseline(nn.Module):
    """Frequency-oriented transformer baseline for binaural DOA."""

    def __init__(
        self,
        freq_bins: int = 257,
        cue_input_mode: str = "ild_ipd",
        time_bins: int = 16,
        num_patches: int = 16,
        embed_dim: int = 64,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_classes: int = 72,
        use_front_back_auxiliary: bool = False,
    ):
        super().__init__()
        if cue_input_mode not in {"ild_ipd", "ild_sincos"}:
            raise ValueError(f"Unsupported cue_input_mode: {cue_input_mode}")
        self.cue_input_mode = cue_input_mode
        self.time_bins = int(time_bins)
        self.num_patches = int(num_patches)
        self.embed_dim = int(embed_dim)
        self.use_front_back_auxiliary = use_front_back_auxiliary

        cue_channels = 2 if cue_input_mode == "ild_ipd" else 3
        padded_freq_bins = math.ceil(freq_bins / self.num_patches) * self.num_patches
        self.patch_freq_bins = padded_freq_bins // self.num_patches
        patch_dim = cue_channels * self.time_bins * self.patch_freq_bins

        self.patch_proj = nn.Linear(patch_dim, self.embed_dim)
        self.patch_norm = nn.LayerNorm(self.embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=int(self.embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.final_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        if self.use_front_back_auxiliary:
            self.front_back_classifier = nn.Linear(self.embed_dim, 2)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.patch_proj.weight)
        if self.patch_proj.bias is not None:
            nn.init.zeros_(self.patch_proj.bias)

    def _build_cue_input(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        ild = batch["ild"]
        if self.cue_input_mode == "ild_ipd":
            ipd = batch["ipd"]
            return torch.stack([ild, ipd], dim=1)  # [B, 2, T, F]
        ipd_sin = batch["ipd_sin"]
        ipd_cos = batch["ipd_cos"]
        return torch.stack([ild, ipd_sin, ipd_cos], dim=1)  # [B, 3, T, F]

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Build frequency-oriented tokens.

        x: [B, C, T, F]
        returns: [B, num_patches, patch_dim]
        """
        b, c, _, f = x.shape
        x = F.adaptive_avg_pool2d(x, (self.time_bins, f))  # [B, C, T', F]
        padded_freq = self.patch_freq_bins * self.num_patches
        if padded_freq > f:
            x = F.pad(x, (0, padded_freq - f))

        x = x.view(b, c, self.time_bins, self.num_patches, self.patch_freq_bins)
        x = x.permute(0, 3, 1, 2, 4).contiguous()  # [B, P, C, T', Fp]
        x = x.view(b, self.num_patches, -1)
        return x

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cue = self._build_cue_input(batch)           # [B, C, T, F]
        patch_tokens = self._patchify(cue)           # [B, P, patch_dim]
        tokens = self.patch_proj(patch_tokens)       # [B, P, D]
        tokens = self.patch_norm(tokens)

        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        x = torch.cat([cls, tokens], dim=1)          # [B, P+1, D]
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x)
        x = self.final_norm(x)
        pooled = self.dropout(x[:, 0])               # cls token

        out = {
            "logits": self.classifier(pooled),
            "cue_tokens": tokens,
            "patch_tokens": patch_tokens,
        }
        if self.use_front_back_auxiliary:
            out["front_back_logits"] = self.front_back_classifier(pooled)
        return out
