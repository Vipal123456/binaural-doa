"""FAViT-style binaural cue Transformer baselines.

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


class _AMViTFeatureEncoder(nn.Module):
    """Frequency-oriented transformer encoder used by AMViT branches."""

    def __init__(
        self,
        in_channels: int,
        freq_bins: int,
        time_bins: int,
        num_patches: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.time_bins = int(time_bins)
        self.num_patches = int(num_patches)
        self.embed_dim = int(embed_dim)

        padded_freq_bins = math.ceil(freq_bins / self.num_patches) * self.num_patches
        self.patch_freq_bins = padded_freq_bins // self.num_patches
        patch_dim = self.in_channels * self.time_bins * self.patch_freq_bins

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

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.patch_proj.weight)
        if self.patch_proj.bias is not None:
            nn.init.zeros_(self.patch_proj.bias)

    def _patchify(self, x: torch.Tensor, label_steps: int) -> torch.Tensor:
        b, c, _, f = x.shape
        x = F.adaptive_avg_pool2d(x, (label_steps * self.time_bins, f))
        padded_freq = self.patch_freq_bins * self.num_patches
        if padded_freq > f:
            x = F.pad(x, (0, padded_freq - f))

        x = x.view(b, c, label_steps, self.time_bins, self.num_patches, self.patch_freq_bins)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.view(b, label_steps, self.num_patches, -1)

    def forward(self, x: torch.Tensor, label_steps: int) -> torch.Tensor:
        patch_tokens = self._patchify(x, label_steps)         # [B, S, P, patch_dim]
        b, s, p, _ = patch_tokens.shape
        tokens = self.patch_proj(patch_tokens)
        tokens = self.patch_norm(tokens).view(b * s, p, self.embed_dim)

        cls = self.cls_token.expand(b * s, -1, -1)
        h = torch.cat([cls, tokens], dim=1)
        h = h + self.pos_embed[:, :h.size(1), :]
        h = self.transformer(h)
        h = self.final_norm(h)
        return self.dropout(h[:, 0]).view(b, s, self.embed_dim)


class FAViTStyleSequenceBaseline(nn.Module):
    """Minimal moving-speaker sequence variant of the FAViT baseline.

    The cue frontend and frequency-oriented patching stay the same as the
    static baseline. We only preserve a label-step axis and predict one DOA
    distribution per 100 ms step.
    """

    def __init__(
        self,
        freq_bins: int = 257,
        cue_input_mode: str = "ild_ipd",
        time_bins: int = 8,
        num_patches: int = 16,
        embed_dim: int = 64,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_classes: int = 72,
        label_steps: int = 40,
        use_front_back_auxiliary: bool = False,
    ):
        super().__init__()
        if cue_input_mode not in {"ild_ipd", "ild_sincos"}:
            raise ValueError(f"Unsupported cue_input_mode: {cue_input_mode}")
        self.cue_input_mode = cue_input_mode
        self.time_bins = int(time_bins)
        self.num_patches = int(num_patches)
        self.embed_dim = int(embed_dim)
        self.label_steps = int(label_steps)
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
        """Build one frequency-token set per label step.

        x: [B, C, T, F]
        returns: [B, S, P, patch_dim]
        """
        b, c, _, f = x.shape
        x = F.adaptive_avg_pool2d(x, (self.label_steps * self.time_bins, f))
        padded_freq = self.patch_freq_bins * self.num_patches
        if padded_freq > f:
            x = F.pad(x, (0, padded_freq - f))

        x = x.view(b, c, self.label_steps, self.time_bins, self.num_patches, self.patch_freq_bins)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # [B, S, P, C, T', Fp]
        return x.view(b, self.label_steps, self.num_patches, -1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cue = self._build_cue_input(batch)               # [B, C, T, F]
        patch_tokens = self._patchify(cue)               # [B, S, P, patch_dim]
        b, s, p, _ = patch_tokens.shape

        tokens = self.patch_proj(patch_tokens)           # [B, S, P, D]
        tokens = self.patch_norm(tokens)
        tokens = tokens.view(b * s, p, self.embed_dim)

        cls = self.cls_token.expand(b * s, -1, -1)
        x = torch.cat([cls, tokens], dim=1)              # [B*S, P+1, D]
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x)
        x = self.final_norm(x)
        pooled = self.dropout(x[:, 0]).view(b, s, self.embed_dim)

        out = {
            "doa_logits": self.classifier(pooled),
            "cue_tokens": tokens.view(b, s, p, self.embed_dim),
            "patch_tokens": patch_tokens,
        }
        if self.use_front_back_auxiliary:
            out["front_back_logits"] = self.front_back_classifier(pooled)
        return out


class AMViTStyleSequenceBaseline(nn.Module):
    """Moving-sequence adaptation of the AMViT paper.

    Branches:
    - top-down: SPL / SPR (left/right complex spectra)
    - bottom-up: ILD / IPD
    Modulation follows the paper default: element-wise multiplication.
    """

    def __init__(
        self,
        freq_bins: int = 257,
        time_bins: int = 8,
        num_patches: int = 16,
        embed_dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_classes: int = 72,
        label_steps: int = 40,
        modulation_type: str = "mul",
        modulation_hidden_dim: int = 128,
        classifier_hidden_dims = (512, 256, 100),
        use_front_back_auxiliary: bool = False,
    ):
        super().__init__()
        if modulation_type not in {"add", "sub", "mul", "dot", "mlp"}:
            raise ValueError(f"Unsupported modulation_type: {modulation_type}")
        self.label_steps = int(label_steps)
        self.embed_dim = int(embed_dim)
        self.modulation_type = modulation_type
        self.use_front_back_auxiliary = use_front_back_auxiliary

        common_kwargs = dict(
            freq_bins=freq_bins,
            time_bins=time_bins,
            num_patches=num_patches,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.spl_encoder = _AMViTFeatureEncoder(in_channels=2, **common_kwargs)
        self.spr_encoder = _AMViTFeatureEncoder(in_channels=2, **common_kwargs)
        self.ild_encoder = _AMViTFeatureEncoder(in_channels=1, **common_kwargs)
        self.ipd_encoder = _AMViTFeatureEncoder(in_channels=1, **common_kwargs)

        if modulation_type == "mlp":
            self.modulation_mlp = nn.Sequential(
                nn.Linear(embed_dim * 2, modulation_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(modulation_hidden_dim, embed_dim),
            )
        else:
            self.modulation_mlp = None

        classifier_layers = []
        in_dim = embed_dim
        for hidden_dim in classifier_hidden_dims:
            classifier_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        classifier_layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

        if self.use_front_back_auxiliary:
            self.front_back_classifier = nn.Linear(embed_dim, 2)

    def _modulate(self, rp_bu: torch.Tensor, rp_td: torch.Tensor) -> torch.Tensor:
        if self.modulation_type == "add":
            return rp_bu + rp_td
        if self.modulation_type == "sub":
            return rp_bu - rp_td
        if self.modulation_type == "mul":
            return rp_bu * rp_td
        if self.modulation_type == "dot":
            scale = (rp_bu * rp_td).sum(dim=-1, keepdim=True)
            return scale * rp_bu
        return self.modulation_mlp(torch.cat([rp_bu, rp_td], dim=-1))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        spl = torch.stack([batch["spec_real_L"], batch["spec_imag_L"]], dim=1)
        spr = torch.stack([batch["spec_real_R"], batch["spec_imag_R"]], dim=1)
        ild = batch["ild"].unsqueeze(1)
        ipd = batch["ipd"].unsqueeze(1)

        rp_spl = self.spl_encoder(spl, self.label_steps)
        rp_spr = self.spr_encoder(spr, self.label_steps)
        rp_ild = self.ild_encoder(ild, self.label_steps)
        rp_ipd = self.ipd_encoder(ipd, self.label_steps)

        rp_td = rp_spl * rp_spr
        rp_bu = rp_ild * rp_ipd
        mod = self._modulate(rp_bu, rp_td)
        logits = self.classifier(mod)

        out = {
            "doa_logits": logits,
            "rp_td": rp_td,
            "rp_bu": rp_bu,
            "mod_feat": mod,
        }
        if self.use_front_back_auxiliary:
            out["front_back_logits"] = self.front_back_classifier(mod)
        return out


class AMViTStyleBaseline(nn.Module):
    """Static AMViT-style baseline adapted from the ICASSP 2024 paper."""

    def __init__(
        self,
        freq_bins: int = 257,
        time_bins: int = 16,
        num_patches: int = 16,
        embed_dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_classes: int = 72,
        modulation_type: str = "mul",
        modulation_hidden_dim: int = 128,
        classifier_hidden_dims = (512, 256, 100),
        use_front_back_auxiliary: bool = False,
    ):
        super().__init__()
        if modulation_type not in {"add", "sub", "mul", "dot", "mlp"}:
            raise ValueError(f"Unsupported modulation_type: {modulation_type}")
        self.modulation_type = modulation_type
        self.use_front_back_auxiliary = use_front_back_auxiliary

        common_kwargs = dict(
            freq_bins=freq_bins,
            time_bins=time_bins,
            num_patches=num_patches,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.spl_encoder = _AMViTFeatureEncoder(in_channels=2, **common_kwargs)
        self.spr_encoder = _AMViTFeatureEncoder(in_channels=2, **common_kwargs)
        self.ild_encoder = _AMViTFeatureEncoder(in_channels=1, **common_kwargs)
        self.ipd_encoder = _AMViTFeatureEncoder(in_channels=1, **common_kwargs)

        if modulation_type == "mlp":
            self.modulation_mlp = nn.Sequential(
                nn.Linear(embed_dim * 2, modulation_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(modulation_hidden_dim, embed_dim),
            )
        else:
            self.modulation_mlp = None

        classifier_layers = []
        in_dim = embed_dim
        for hidden_dim in classifier_hidden_dims:
            classifier_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        classifier_layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

        if self.use_front_back_auxiliary:
            self.front_back_classifier = nn.Linear(embed_dim, 2)

    def _modulate(self, rp_bu: torch.Tensor, rp_td: torch.Tensor) -> torch.Tensor:
        if self.modulation_type == "add":
            return rp_bu + rp_td
        if self.modulation_type == "sub":
            return rp_bu - rp_td
        if self.modulation_type == "mul":
            return rp_bu * rp_td
        if self.modulation_type == "dot":
            scale = (rp_bu * rp_td).sum(dim=-1, keepdim=True)
            return scale * rp_bu
        return self.modulation_mlp(torch.cat([rp_bu, rp_td], dim=-1))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        spl = torch.stack([batch["spec_real_L"], batch["spec_imag_L"]], dim=1)
        spr = torch.stack([batch["spec_real_R"], batch["spec_imag_R"]], dim=1)
        ild = batch["ild"].unsqueeze(1)
        ipd = batch["ipd"].unsqueeze(1)

        rp_spl = self.spl_encoder(spl, label_steps=1).squeeze(1)
        rp_spr = self.spr_encoder(spr, label_steps=1).squeeze(1)
        rp_ild = self.ild_encoder(ild, label_steps=1).squeeze(1)
        rp_ipd = self.ipd_encoder(ipd, label_steps=1).squeeze(1)

        rp_td = rp_spl * rp_spr
        rp_bu = rp_ild * rp_ipd
        mod = self._modulate(rp_bu, rp_td)
        out = {
            "logits": self.classifier(mod),
            "rp_td": rp_td,
            "rp_bu": rp_bu,
            "mod_feat": mod,
        }
        if self.use_front_back_auxiliary:
            out["front_back_logits"] = self.front_back_classifier(mod)
        return out
