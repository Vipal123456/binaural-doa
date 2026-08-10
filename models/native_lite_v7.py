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

import math
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


def _adaptive_uniform_filterbank(num_bands: int, freq_bins: int) -> torch.Tensor:
    """Match adaptive average pooling with a fixed, normalized band matrix."""
    if not 1 <= num_bands <= freq_bins:
        raise ValueError(
            f"num_bands must be in [1, {freq_bins}], got {num_bands}"
        )
    filters = torch.zeros(num_bands, freq_bins)
    for band_idx in range(num_bands):
        start = (band_idx * freq_bins) // num_bands
        end = ((band_idx + 1) * freq_bins + num_bands - 1) // num_bands
        filters[band_idx, start:end] = 1.0 / float(end - start)
    return filters


class BinauralCueStatistics(nn.Module):
    """Compress complex spectra before forming nonlinear binaural cues.

    ``precue_stat`` forms band energy ILD, band cross-spectrum phase, and
    band coherence. ``phaseaware_stat`` replaces the band cross-spectrum phase
    with a coherence-weighted subband GCC delay estimate.
    """

    def __init__(
        self,
        mode: str,
        num_bands: int = 16,
        freq_bins: int = 257,
        sample_rate: int = 16000,
        delay_max_ms: float = 1.0,
        delay_bins: int = 33,
        delay_temperature: float = 20.0,
        eps: float = 1.0e-8,
    ):
        super().__init__()
        if mode not in {"precue_stat", "phaseaware_stat"}:
            raise ValueError(f"Unsupported cue statistics mode: {mode}")
        if delay_bins < 3 or delay_bins % 2 == 0:
            raise ValueError("delay_bins must be an odd integer >= 3")
        if delay_max_ms <= 0.0:
            raise ValueError("delay_max_ms must be positive")
        if delay_temperature <= 0.0:
            raise ValueError("delay_temperature must be positive")

        self.mode = mode
        self.num_bands = num_bands
        self.freq_bins = freq_bins
        self.sample_rate = sample_rate
        self.delay_temperature = delay_temperature
        self.eps = eps

        band_weights = _adaptive_uniform_filterbank(num_bands, freq_bins)
        frequencies = torch.linspace(0.0, sample_rate / 2.0, steps=freq_bins)
        band_centers = torch.sum(band_weights * frequencies.unsqueeze(0), dim=-1)
        delay_max_seconds = delay_max_ms / 1000.0
        delays = torch.linspace(-delay_max_seconds, delay_max_seconds, steps=delay_bins)
        gcc_phase_basis = torch.exp(
            -1j * 2.0 * torch.pi * frequencies.unsqueeze(1) * delays.unsqueeze(0)
        )

        self.register_buffer("band_weights", band_weights, persistent=False)
        self.register_buffer("band_centers_hz", band_centers, persistent=False)
        self.register_buffer("delays_seconds", delays, persistent=False)
        self.register_buffer("gcc_phase_basis", gcc_phase_basis, persistent=False)

        self.band_slices = []
        for band_idx in range(num_bands):
            nonzero = torch.nonzero(band_weights[band_idx] > 0, as_tuple=False).flatten()
            self.band_slices.append((int(nonzero[0]), int(nonzero[-1]) + 1))

    def _complex_spectra(
        self,
        batch: Dict[str, torch.Tensor],
        time_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        required = ("spec_real_L", "spec_imag_L", "spec_real_R", "spec_imag_R")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(
                f"cue statistics mode '{self.mode}' requires complex spectra; missing {missing}"
            )
        spec_l = torch.complex(
            batch["spec_real_L"][:, :time_steps, :],
            batch["spec_imag_L"][:, :time_steps, :],
        )
        spec_r = torch.complex(
            batch["spec_real_R"][:, :time_steps, :],
            batch["spec_imag_R"][:, :time_steps, :],
        )
        if spec_l.shape[-1] != self.freq_bins:
            raise ValueError(
                f"Expected {self.freq_bins} frequency bins, got {spec_l.shape[-1]}"
            )
        return spec_l, spec_r

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        time_steps: int,
    ) -> Dict[str, torch.Tensor]:
        spec_l, spec_r = self._complex_spectra(batch, time_steps)
        weights = self.band_weights.to(device=spec_l.device, dtype=spec_l.real.dtype)
        power_l = torch.einsum("btf,kf->btk", spec_l.abs().square(), weights)
        power_r = torch.einsum("btf,kf->btk", spec_r.abs().square(), weights)
        ild = 10.0 * torch.log10((power_l + self.eps) / (power_r + self.eps))
        cross = spec_l * spec_r.conj()

        if self.mode == "precue_stat":
            band_cross = torch.einsum(
                "btf,kf->btk", cross, weights.to(dtype=spec_l.dtype)
            )
            band_cross_mag = band_cross.abs()
            normalized_cross = band_cross / (band_cross_mag + self.eps)
            coherence = band_cross_mag / torch.sqrt(power_l * power_r + self.eps)
            value = torch.stack(
                [ild, normalized_cross.imag, normalized_cross.real], dim=1
            )
            return {
                "value_tensor": value,
                "reliability_tensor": coherence.clamp(0.0, 1.0).unsqueeze(1),
            }

        coherence_bins = batch.get("coherence")
        if coherence_bins is None:
            raise KeyError("phaseaware_stat requires the bin-level 'coherence' feature")
        coherence_bins = coherence_bins[:, :time_steps, :].clamp(0.0, 1.0)
        unit_cross = cross / (cross.abs() + self.eps)
        phase_basis = self.gcc_phase_basis.to(device=spec_l.device, dtype=spec_l.dtype)

        itd_bands = []
        reliability_bands = []
        for band_idx, (start, end) in enumerate(self.band_slices):
            band_weight = weights[band_idx, start:end]
            support = coherence_bins[..., start:end] * band_weight
            weighted_cross = unit_cross[..., start:end] * support
            gcc = torch.matmul(weighted_cross, phase_basis[start:end, :])
            gcc_score = gcc.abs() / support.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            delay_probability = torch.softmax(
                self.delay_temperature * gcc_score, dim=-1
            )
            itd = torch.sum(
                delay_probability * self.delays_seconds.to(gcc_score.dtype), dim=-1
            )

            peak = gcc_score.amax(dim=-1)
            mean = gcc_score.mean(dim=-1)
            sharpness = ((peak - mean) / (peak + self.eps)).clamp(0.0, 1.0)
            reliability = (peak.clamp(0.0, 1.0) * sharpness).clamp(0.0, 1.0)
            itd_bands.append(itd)
            reliability_bands.append(reliability)

        itd_seconds = torch.stack(itd_bands, dim=-1)
        reliability = torch.stack(reliability_bands, dim=-1)
        center_phase = (
            2.0
            * torch.pi
            * itd_seconds
            * self.band_centers_hz.to(device=itd_seconds.device, dtype=itd_seconds.dtype)
        )
        value = torch.stack([ild, torch.sin(center_phase), torch.cos(center_phase)], dim=1)
        return {
            "value_tensor": value,
            "reliability_tensor": reliability.unsqueeze(1),
            "itd_seconds": itd_seconds,
        }


class ReliabilityWeightedCPSD(nn.Module):
    """Estimate binaural cues from a learnable, locally weighted CPSD.

    A uniform window provides pilot power and cross-spectrum estimates. Three
    zero-initialized coefficients then score each frame by relative energy,
    phase agreement, and ILD agreement with that pilot. The same softmax
    weights estimate both auto-spectra and the complex cross-spectrum.
    """

    def __init__(
        self,
        time_frames: int = 5,
        score_logit_clip: float = 6.0,
        coefficient_mode: str = "global",
        frequency_anchors: int = 8,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if time_frames < 3 or time_frames % 2 == 0:
            raise ValueError("RW-CPSD time_frames must be an odd integer >= 3")
        if score_logit_clip <= 0.0:
            raise ValueError("RW-CPSD score_logit_clip must be positive")
        if coefficient_mode not in {"global", "frequency_anchors"}:
            raise ValueError(
                "RW-CPSD coefficient_mode must be 'global' or 'frequency_anchors'"
            )
        if coefficient_mode == "frequency_anchors" and frequency_anchors < 2:
            raise ValueError("frequency-aware RW-CPSD requires at least two anchors")
        self.time_frames = int(time_frames)
        self.score_logit_clip = float(score_logit_clip)
        self.coefficient_mode = coefficient_mode
        self.frequency_anchors = int(frequency_anchors)
        self.eps = float(eps)

        # [energy, phase agreement, ILD agreement]. At zero, softmax is
        # uniform and the module exactly matches fixed-window CPSD averaging.
        coefficient_shape = (
            (3, self.frequency_anchors)
            if coefficient_mode == "frequency_anchors"
            else (3,)
        )
        self.score_coefficients = nn.Parameter(torch.zeros(coefficient_shape))

    def frequency_score_coefficients(self, frequency_bins: int) -> torch.Tensor:
        """Return the three scoring coefficients at every STFT frequency bin."""
        if frequency_bins < 1:
            raise ValueError("frequency_bins must be positive")
        if self.coefficient_mode == "global":
            return self.score_coefficients[:, None].expand(-1, frequency_bins)
        return F.interpolate(
            self.score_coefficients.unsqueeze(0),
            size=frequency_bins,
            mode="linear",
            align_corners=True,
        ).squeeze(0)

    def _complex_spectra(
        self,
        batch: Dict[str, torch.Tensor],
        time_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        required = ("spec_real_L", "spec_imag_L", "spec_real_R", "spec_imag_R")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"RW-CPSD requires complex spectra; missing {missing}")
        spec_l = torch.complex(
            batch["spec_real_L"][:, :time_steps, :],
            batch["spec_imag_L"][:, :time_steps, :],
        )
        spec_r = torch.complex(
            batch["spec_real_R"][:, :time_steps, :],
            batch["spec_imag_R"][:, :time_steps, :],
        )
        return spec_l, spec_r

    @staticmethod
    def _complex_spectra_with_prefix(
        batch: Dict[str, torch.Tensor],
        time_steps: int,
        prefix: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        required = tuple(
            f"{prefix}{key}"
            for key in ("spec_real_L", "spec_imag_L", "spec_real_R", "spec_imag_R")
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"CPSD component spectra missing {missing}")
        spec_l = torch.complex(
            batch[f"{prefix}spec_real_L"][:, :time_steps, :],
            batch[f"{prefix}spec_imag_L"][:, :time_steps, :],
        )
        spec_r = torch.complex(
            batch[f"{prefix}spec_real_R"][:, :time_steps, :],
            batch[f"{prefix}spec_imag_R"][:, :time_steps, :],
        )
        return spec_l, spec_r

    def _score_bias(
        self,
        batch: Dict[str, torch.Tensor],
        time_steps: int,
    ) -> torch.Tensor | None:
        return None

    def _time_windows(self, x: torch.Tensor) -> torch.Tensor:
        """Return centered windows with shape [B, T, F, K]."""
        pad = self.time_frames // 2
        if x.shape[1] <= pad:
            raise ValueError(
                f"RW-CPSD needs more than {pad} time frames, got {x.shape[1]}"
            )
        x_bft = x.transpose(1, 2)
        if x_bft.is_complex():
            padded = torch.complex(
                F.pad(x_bft.real, (pad, pad), mode="reflect"),
                F.pad(x_bft.imag, (pad, pad), mode="reflect"),
            )
        else:
            padded = F.pad(x_bft, (pad, pad), mode="reflect")
        return padded.unfold(-1, self.time_frames, 1).permute(0, 2, 1, 3)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        time_steps: int,
    ) -> Dict[str, torch.Tensor]:
        spec_l, spec_r = self._complex_spectra(batch, time_steps)
        left = self._time_windows(spec_l)
        right = self._time_windows(spec_r)

        power_l_frames = left.abs().square()
        power_r_frames = right.abs().square()
        cross_frames = left * right.conj()

        pilot_power_l = power_l_frames.mean(dim=-1)
        pilot_power_r = power_r_frames.mean(dim=-1)
        pilot_cross = cross_frames.mean(dim=-1)

        log_energy = torch.log(power_l_frames + power_r_frames + self.eps)
        energy_score = log_energy - log_energy.mean(dim=-1, keepdim=True)
        energy_scale = energy_score.std(dim=-1, keepdim=True, unbiased=False)
        energy_score = energy_score / energy_scale.clamp_min(0.1)

        unit_cross = cross_frames / (cross_frames.abs() + self.eps)
        pilot_unit = pilot_cross / (pilot_cross.abs() + self.eps)
        phase_agreement = (unit_cross * pilot_unit.conj().unsqueeze(-1)).real

        instant_log_ratio = torch.log(power_l_frames + self.eps) - torch.log(
            power_r_frames + self.eps
        )
        pilot_log_ratio = torch.log(pilot_power_l + self.eps) - torch.log(
            pilot_power_r + self.eps
        )
        ild_agreement = -(
            instant_log_ratio - pilot_log_ratio.unsqueeze(-1)
        ).abs().clamp(max=4.0) / 4.0

        coeff = self.frequency_score_coefficients(energy_score.shape[2])
        coeff = coeff[:, None, :, None]
        score = (
            coeff[0] * energy_score
            + coeff[1] * phase_agreement
            + coeff[2] * ild_agreement
        )
        score_bias = self._score_bias(batch, time_steps)
        if score_bias is not None:
            score = score + score_bias
        score = score.clamp(-self.score_logit_clip, self.score_logit_clip)
        weights = torch.softmax(score, dim=-1)

        power_l = torch.sum(weights * power_l_frames, dim=-1)
        power_r = torch.sum(weights * power_r_frames, dim=-1)
        cross = torch.sum(weights * cross_frames, dim=-1)

        ild = 10.0 * torch.log10((power_l + self.eps) / (power_r + self.eps))
        cross_mag = cross.abs()
        normalized_cross = cross / (cross_mag + self.eps)
        coherence = cross_mag / torch.sqrt(power_l * power_r + self.eps)

        weighted_log_ratio = torch.sum(weights * instant_log_ratio, dim=-1)
        ild_log_ratio_var = torch.sum(
            weights
            * (instant_log_ratio - weighted_log_ratio.unsqueeze(-1)).square(),
            dim=-1,
        )
        ild_consistency = 1.0 / (1.0 + ild_log_ratio_var)
        ipd_consistency = torch.sum(weights * unit_cross, dim=-1).abs()
        value = torch.stack(
            [ild, normalized_cross.imag, normalized_cross.real], dim=1
        )
        return {
            "value_tensor": value,
            "reliability_tensor": coherence.clamp(0.0, 1.0).unsqueeze(1),
            "ild_consistency_tensor": ild_consistency.clamp(0.0, 1.0).unsqueeze(1),
            "ipd_consistency_tensor": ipd_consistency.clamp(0.0, 1.0).unsqueeze(1),
            "tf_weight": weights,
        }


class OracleTargetCPSD(ReliabilityWeightedCPSD):
    """Upper-bound estimator that computes RW-CPSD from target-only spectra."""

    def _complex_spectra(
        self,
        batch: Dict[str, torch.Tensor],
        time_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._complex_spectra_with_prefix(batch, time_steps, "target_")


class OracleTargetMaskedCPSD(ReliabilityWeightedCPSD):
    """Apply an ideal target-dominance prior to mixture CPSD weights."""

    def _score_bias(
        self,
        batch: Dict[str, torch.Tensor],
        time_steps: int,
    ) -> torch.Tensor:
        target_l, target_r = self._complex_spectra_with_prefix(
            batch, time_steps, "target_"
        )
        interferer_l, interferer_r = self._complex_spectra_with_prefix(
            batch, time_steps, "interferer_"
        )
        target_power = target_l.abs().square() + target_r.abs().square()
        interferer_power = (
            interferer_l.abs().square() + interferer_r.abs().square()
        )
        dominance = target_power / (
            target_power + interferer_power + self.eps
        )
        dominance_windows = self._time_windows(dominance)
        return torch.log(dominance_windows.clamp_min(1.0e-3))


class CueFactorizedCPSD(ReliabilityWeightedCPSD):
    """Estimate ILD and IPD with cue-specific local CPSD weights.

    ILD depends on a stable interaural power ratio, whereas IPD and coherence
    depend on a stable complex cross-spectrum.  This module therefore keeps a
    common five-frame pilot but learns separate weights for the two moments.
    Both branches are zero-initialized and exactly reduce to uniform CPSD
    averaging at initialization.
    """

    def __init__(
        self,
        time_frames: int = 5,
        score_logit_clip: float = 6.0,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__(
            time_frames=time_frames,
            score_logit_clip=score_logit_clip,
            coefficient_mode="global",
            eps=eps,
        )
        del self.score_coefficients
        # ILD: [relative energy, ILD agreement].
        self.ild_score_coefficients = nn.Parameter(torch.zeros(2))
        # IPD: [relative energy, phase agreement].
        self.ipd_score_coefficients = nn.Parameter(torch.zeros(2))

    def _target_score_bias(
        self,
        batch: Dict[str, torch.Tensor],
        spec_l: torch.Tensor,
        spec_r: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        return None, None, None

    def _score_residuals(
        self,
        energy_score: torch.Tensor,
        ild_agreement: torch.Tensor,
        phase_agreement: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return optional cue-specific residual logits for each local frame."""
        return None, None

    def _cue_scores(
        self,
        energy_score: torch.Tensor,
        ild_agreement: torch.Tensor,
        phase_agreement: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Form ILD/IPD frame logits before optional target-aware biases."""
        ild_score = (
            self.ild_score_coefficients[0] * energy_score
            + self.ild_score_coefficients[1] * ild_agreement
        )
        ipd_score = (
            self.ipd_score_coefficients[0] * energy_score
            + self.ipd_score_coefficients[1] * phase_agreement
        )
        ild_residual, ipd_residual = self._score_residuals(
            energy_score,
            ild_agreement,
            phase_agreement,
        )
        if ild_residual is not None:
            ild_score = ild_score + ild_residual
        if ipd_residual is not None:
            ipd_score = ipd_score + ipd_residual
        return ild_score, ipd_score

    def _auxiliary_outputs(
        self,
        batch: Dict[str, torch.Tensor],
        target_probability: torch.Tensor | None,
        ild_power_l: torch.Tensor,
        ild_power_r: torch.Tensor,
        ild_cross: torch.Tensor,
        ipd_power_l: torch.Tensor,
        ipd_power_r: torch.Tensor,
        ipd_cross: torch.Tensor,
    ) -> Dict[str, torch.Tensor | None]:
        return {}

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        time_steps: int,
    ) -> Dict[str, torch.Tensor]:
        spec_l, spec_r = self._complex_spectra(batch, time_steps)
        left = self._time_windows(spec_l)
        right = self._time_windows(spec_r)

        power_l_frames = left.abs().square()
        power_r_frames = right.abs().square()
        cross_frames = left * right.conj()

        pilot_power_l = power_l_frames.mean(dim=-1)
        pilot_power_r = power_r_frames.mean(dim=-1)
        pilot_cross = cross_frames.mean(dim=-1)

        log_energy = torch.log(power_l_frames + power_r_frames + self.eps)
        energy_score = log_energy - log_energy.mean(dim=-1, keepdim=True)
        energy_scale = energy_score.std(dim=-1, keepdim=True, unbiased=False)
        energy_score = energy_score / energy_scale.clamp_min(0.1)

        unit_cross = cross_frames / (cross_frames.abs() + self.eps)
        pilot_unit = pilot_cross / (pilot_cross.abs() + self.eps)
        phase_agreement = (unit_cross * pilot_unit.conj().unsqueeze(-1)).real

        instant_log_ratio = torch.log(power_l_frames + self.eps) - torch.log(
            power_r_frames + self.eps
        )
        pilot_log_ratio = torch.log(pilot_power_l + self.eps) - torch.log(
            pilot_power_r + self.eps
        )
        ild_agreement = -(
            instant_log_ratio - pilot_log_ratio.unsqueeze(-1)
        ).abs().clamp(max=4.0) / 4.0

        ild_score, ipd_score = self._cue_scores(
            energy_score,
            ild_agreement,
            phase_agreement,
        )
        ild_target_bias, ipd_target_bias, target_probability = self._target_score_bias(
            batch, spec_l, spec_r
        )
        if ild_target_bias is not None:
            ild_score = ild_score + ild_target_bias
        if ipd_target_bias is not None:
            ipd_score = ipd_score + ipd_target_bias
        ild_score = ild_score.clamp(
            -self.score_logit_clip, self.score_logit_clip
        )
        ipd_score = ipd_score.clamp(
            -self.score_logit_clip, self.score_logit_clip
        )
        ild_weights = torch.softmax(ild_score, dim=-1)
        ipd_weights = torch.softmax(ipd_score, dim=-1)

        ild_power_l = torch.sum(ild_weights * power_l_frames, dim=-1)
        ild_power_r = torch.sum(ild_weights * power_r_frames, dim=-1)
        ild_cross = torch.sum(ild_weights * cross_frames, dim=-1)
        ild = 10.0 * torch.log10(
            (ild_power_l + self.eps) / (ild_power_r + self.eps)
        )

        ipd_power_l = torch.sum(ipd_weights * power_l_frames, dim=-1)
        ipd_power_r = torch.sum(ipd_weights * power_r_frames, dim=-1)
        cross = torch.sum(ipd_weights * cross_frames, dim=-1)
        cross_mag = cross.abs()
        normalized_cross = cross / (cross_mag + self.eps)
        coherence = cross_mag / torch.sqrt(
            ipd_power_l * ipd_power_r + self.eps
        )

        weighted_log_ratio = torch.sum(
            ild_weights * instant_log_ratio, dim=-1
        )
        ild_log_ratio_var = torch.sum(
            ild_weights
            * (instant_log_ratio - weighted_log_ratio.unsqueeze(-1)).square(),
            dim=-1,
        )
        ild_consistency = 1.0 / (1.0 + ild_log_ratio_var)
        ipd_consistency = torch.sum(
            ipd_weights * unit_cross, dim=-1
        ).abs()

        value = torch.stack(
            [ild, normalized_cross.imag, normalized_cross.real], dim=1
        )
        output = {
            "value_tensor": value,
            "reliability_tensor": coherence.clamp(0.0, 1.0).unsqueeze(1),
            "ild_consistency_tensor": ild_consistency.clamp(0.0, 1.0).unsqueeze(1),
            "ipd_consistency_tensor": ipd_consistency.clamp(0.0, 1.0).unsqueeze(1),
            "tf_weight": 0.5 * (ild_weights + ipd_weights),
            "tf_weight_ild": ild_weights,
            "tf_weight_ipd": ipd_weights,
            "tf_score_ild": ild_score,
            "tf_score_ipd": ipd_score,
        }
        output.update(
            self._auxiliary_outputs(
                batch,
                target_probability,
                ild_power_l,
                ild_power_r,
                ild_cross,
                ipd_power_l,
                ipd_power_r,
                cross,
            )
        )
        return output


class NonlinearCueFactorizedCPSD(CueFactorizedCPSD):
    """B2 with compact frequency-aware nonlinear reliability residuals.

    The zero-initialized final layers make this estimator exactly equivalent
    to the original B2 estimator at initialization.  The residual heads can
    then learn frequency-dependent interactions that four global linear
    coefficients cannot express.
    """

    def __init__(
        self,
        time_frames: int = 5,
        score_logit_clip: float = 6.0,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__(time_frames, score_logit_clip, eps)
        self.ild_score_residual = self._make_residual_head()
        self.ipd_score_residual = self._make_residual_head()

    @staticmethod
    def _make_residual_head() -> nn.Sequential:
        head = nn.Sequential(
            nn.Linear(3, 8),
            nn.SiLU(inplace=True),
            nn.Linear(8, 1),
        )
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        return head

    def _score_residuals(
        self,
        energy_score: torch.Tensor,
        ild_agreement: torch.Tensor,
        phase_agreement: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        freq_bins = energy_score.shape[-2]
        frequency = torch.linspace(
            0.0,
            1.0,
            freq_bins,
            device=energy_score.device,
            dtype=energy_score.dtype,
        ).view(1, 1, freq_bins, 1)
        frequency = frequency.expand_as(energy_score)
        ild_input = torch.stack(
            [energy_score, ild_agreement, frequency], dim=-1
        )
        ipd_input = torch.stack(
            [energy_score, phase_agreement, frequency], dim=-1
        )
        return (
            self.ild_score_residual(ild_input).squeeze(-1),
            self.ipd_score_residual(ipd_input).squeeze(-1),
        )


class PrecisionWeightedCueFactorizedCPSD(CueFactorizedCPSD):
    """Estimate cue-specific observation precision before local CPSD pooling.

    The ILD head predicts log variance and is converted to inverse-variance
    logits.  The IPD head predicts log circular concentration.  Both heads
    start at zero, so the initial estimator is uniform five-frame CPSD.
    """

    def __init__(
        self,
        time_frames: int = 5,
        score_logit_clip: float = 6.0,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__(time_frames, score_logit_clip, eps)
        del self.ild_score_coefficients
        del self.ipd_score_coefficients
        self.ild_log_variance_head = self._make_uncertainty_head()
        self.ipd_log_concentration_head = self._make_uncertainty_head()

    @staticmethod
    def _make_uncertainty_head() -> nn.Sequential:
        head = nn.Sequential(
            nn.Linear(3, 8),
            nn.SiLU(inplace=True),
            nn.Linear(8, 1),
        )
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        return head

    def _cue_scores(
        self,
        energy_score: torch.Tensor,
        ild_agreement: torch.Tensor,
        phase_agreement: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        freq_bins = energy_score.shape[-2]
        frequency = torch.linspace(
            0.0,
            1.0,
            freq_bins,
            device=energy_score.device,
            dtype=energy_score.dtype,
        ).view(1, 1, freq_bins, 1)
        frequency = frequency.expand_as(energy_score)
        ild_input = torch.stack(
            [energy_score, -ild_agreement, frequency], dim=-1
        )
        ipd_input = torch.stack(
            [energy_score, phase_agreement, frequency], dim=-1
        )
        log_variance = self.ild_log_variance_head(ild_input).squeeze(-1)
        log_concentration = self.ipd_log_concentration_head(ipd_input).squeeze(-1)
        return -log_variance, log_concentration

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        time_steps: int,
    ) -> Dict[str, torch.Tensor | None]:
        output = super().forward(batch, time_steps)
        output["ild_log_variance"] = -output["tf_score_ild"]
        output["ipd_log_concentration"] = output["tf_score_ipd"]
        output["cue_uncertainty_loss"] = None
        return output


class CalibratedPrecisionCueFactorizedCPSD(
    PrecisionWeightedCueFactorizedCPSD
):
    """Precision-weighted CPSD with aggregate Gaussian/circular calibration."""

    def __init__(
        self,
        time_frames: int = 5,
        score_logit_clip: float = 6.0,
        ild_normalizer_db: float = 20.0,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__(time_frames, score_logit_clip, eps)
        if ild_normalizer_db <= 0.0:
            raise ValueError("ild_normalizer_db must be positive")
        self.ild_normalizer_db = float(ild_normalizer_db)

    def _weighted_mean(
        self,
        value: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sum(value * weight) / weight.sum().clamp_min(self.eps)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        time_steps: int,
    ) -> Dict[str, torch.Tensor | None]:
        output = super().forward(batch, time_steps)
        if "target_spec_real_L" not in batch:
            return output

        target_l, target_r = self._complex_spectra_with_prefix(
            batch, time_steps, "target_"
        )
        target_l_windows = self._time_windows(target_l)
        target_r_windows = self._time_windows(target_r)
        target_power_l = target_l_windows.abs().square().mean(dim=-1)
        target_power_r = target_r_windows.abs().square().mean(dim=-1)
        target_cross = (
            target_l_windows * target_r_windows.conj()
        ).mean(dim=-1)

        target_ild = 10.0 * torch.log10(
            (target_power_l + self.eps) / (target_power_r + self.eps)
        )
        predicted_ild = output["value_tensor"][:, 0]
        ild_error = (
            predicted_ild - target_ild.detach()
        ) / self.ild_normalizer_db

        ild_log_precision = output["tf_score_ild"]
        aggregate_log_variance = -torch.logsumexp(
            ild_log_precision, dim=-1
        )
        aggregate_log_variance = aggregate_log_variance.clamp(-8.0, 6.0)
        ild_nll = 0.5 * (
            ild_error.square() * torch.exp(-aggregate_log_variance)
            + aggregate_log_variance
            + math.log(2.0 * math.pi)
        )

        predicted_unit = torch.complex(
            output["value_tensor"][:, 2],
            output["value_tensor"][:, 1],
        )
        target_unit = target_cross / (target_cross.abs() + self.eps)
        phase_cosine = (
            predicted_unit * target_unit.detach().conj()
        ).real.clamp(-1.0, 1.0)
        log_mean_concentration = torch.logsumexp(
            output["tf_score_ipd"], dim=-1
        ) - math.log(float(self.time_frames))
        concentration = torch.exp(
            log_mean_concentration.clamp(-6.0, math.log(50.0))
        )
        log_i0 = torch.log(
            torch.special.i0e(concentration).clamp_min(self.eps)
        ) + concentration
        ipd_nll = (
            math.log(2.0 * math.pi)
            + log_i0
            - concentration * phase_cosine
        )

        target_energy = target_power_l + target_power_r
        ild_weight = (
            target_energy
            / target_energy.mean(dim=(1, 2), keepdim=True).clamp_min(self.eps)
        ).clamp(max=10.0).detach()
        target_cross_magnitude = target_cross.abs()
        ipd_weight = (
            target_cross_magnitude
            / target_cross_magnitude.mean(
                dim=(1, 2), keepdim=True
            ).clamp_min(self.eps)
        ).clamp(max=10.0).detach()

        ild_loss = self._weighted_mean(ild_nll, ild_weight)
        ipd_loss = self._weighted_mean(ipd_nll, ipd_weight)
        output["cue_uncertainty_loss"] = 0.5 * (ild_loss + ipd_loss)
        output["cue_uncertainty_ild_loss"] = ild_loss
        output["cue_uncertainty_ipd_loss"] = ipd_loss
        output["aggregate_ild_log_variance"] = aggregate_log_variance
        output["aggregate_ipd_concentration"] = concentration
        return output


class TargetDominanceHead(nn.Module):
    """Predict a target-speech dominance probability from common energy."""

    def __init__(self, hidden_channels: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, spec_l: torch.Tensor, spec_r: torch.Tensor) -> torch.Tensor:
        log_power = torch.log(
            spec_l.abs().square() + spec_r.abs().square() + 1.0e-8
        )
        normalized = log_power - log_power.mean(dim=(1, 2), keepdim=True)
        return torch.sigmoid(self.net(normalized.unsqueeze(1)).squeeze(1))


class TargetAwareRWCPSD(ReliabilityWeightedCPSD):
    """Current shared-weight RW-CPSD augmented by target dominance."""

    def __init__(
        self,
        time_frames: int = 5,
        score_logit_clip: float = 6.0,
        coefficient_mode: str = "global",
        frequency_anchors: int = 8,
        target_hidden_channels: int = 8,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__(
            time_frames=time_frames,
            score_logit_clip=score_logit_clip,
            coefficient_mode=coefficient_mode,
            frequency_anchors=frequency_anchors,
            eps=eps,
        )
        self.target_dominance_head = TargetDominanceHead(target_hidden_channels)
        self._target_probability: torch.Tensor | None = None

    def _score_bias(
        self,
        batch: Dict[str, torch.Tensor],
        time_steps: int,
    ) -> torch.Tensor:
        spec_l, spec_r = super()._complex_spectra(batch, time_steps)
        self._target_probability = self.target_dominance_head(spec_l, spec_r)
        return torch.log(
            self._time_windows(self._target_probability).clamp_min(1.0e-3)
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        time_steps: int,
    ) -> Dict[str, torch.Tensor | None]:
        self._target_probability = None
        output = super().forward(batch, time_steps)
        probability = self._target_probability
        output["target_probability"] = probability
        output["target_mask_loss"] = None
        output["target_covariance_loss"] = None
        if probability is not None and "target_spec_real_L" in batch:
            target_l, target_r = self._complex_spectra_with_prefix(
                batch, time_steps, "target_"
            )
            interferer_l, interferer_r = self._complex_spectra_with_prefix(
                batch, time_steps, "interferer_"
            )
            target_power = target_l.abs().square() + target_r.abs().square()
            interferer_power = (
                interferer_l.abs().square() + interferer_r.abs().square()
            )
            ideal_probability = target_power / (
                target_power + interferer_power + self.eps
            )
            output["target_mask_loss"] = F.binary_cross_entropy(
                probability.clamp(1.0e-4, 1.0 - 1.0e-4),
                ideal_probability.detach(),
            )
        return output


class TargetAwareCueFactorizedCPSD(CueFactorizedCPSD):
    """Cue-factorized CPSD with target dominance and covariance recovery loss."""

    def __init__(
        self,
        time_frames: int = 5,
        score_logit_clip: float = 6.0,
        target_hidden_channels: int = 8,
        target_bias_mode: str = "shared_unit",
        target_bias_max_strength: float = 2.0,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__(time_frames, score_logit_clip, eps)
        if target_bias_mode not in {
            "shared_unit",
            "cue_residual",
            "disabled",
            "oracle_shared",
        }:
            raise ValueError(f"Unsupported target_bias_mode: {target_bias_mode}")
        if target_bias_max_strength <= 0.0:
            raise ValueError("target_bias_max_strength must be positive")
        self.target_dominance_head = TargetDominanceHead(target_hidden_channels)
        self.target_bias_mode = target_bias_mode
        self.target_bias_max_strength = float(target_bias_max_strength)
        self.target_bias_coefficients = (
            nn.Parameter(torch.zeros(2))
            if target_bias_mode == "cue_residual"
            else None
        )

    def _target_score_bias(
        self,
        batch: Dict[str, torch.Tensor],
        spec_l: torch.Tensor,
        spec_r: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_probability = self.target_dominance_head(spec_l, spec_r)
        if self.target_bias_mode == "disabled":
            return None, None, target_probability
        bias_probability = target_probability
        if self.target_bias_mode == "oracle_shared":
            if "target_spec_real_L" not in batch:
                raise RuntimeError(
                    "oracle_shared target bias requires target/interferer sidecars"
                )
            time_steps = target_probability.shape[1]
            target_l, target_r = self._complex_spectra_with_prefix(
                batch, time_steps, "target_"
            )
            interferer_l, interferer_r = self._complex_spectra_with_prefix(
                batch, time_steps, "interferer_"
            )
            target_power = target_l.abs().square() + target_r.abs().square()
            interferer_power = (
                interferer_l.abs().square() + interferer_r.abs().square()
            )
            bias_probability = target_power / (
                target_power + interferer_power + self.eps
            )
        probability_windows = self._time_windows(bias_probability)
        log_probability = torch.log(probability_windows.clamp_min(1.0e-3))
        if self.target_bias_mode in {"shared_unit", "oracle_shared"}:
            return log_probability, log_probability, target_probability
        coefficients = self.target_bias_max_strength * torch.tanh(
            self.target_bias_coefficients
        )
        return (
            coefficients[0] * log_probability,
            coefficients[1] * log_probability,
            target_probability,
        )

    def _normalized_covariance_loss(
        self,
        power_l: torch.Tensor,
        power_r: torch.Tensor,
        cross: torch.Tensor,
        target_power_l: torch.Tensor,
        target_power_r: torch.Tensor,
        target_cross: torch.Tensor,
    ) -> torch.Tensor:
        trace = (power_l + power_r).clamp_min(self.eps)
        target_trace = (target_power_l + target_power_r).clamp_min(self.eps)
        return (
            F.l1_loss(power_l / trace, target_power_l / target_trace)
            + F.l1_loss(power_r / trace, target_power_r / target_trace)
            + F.l1_loss(cross.real / trace, target_cross.real / target_trace)
            + F.l1_loss(cross.imag / trace, target_cross.imag / target_trace)
        )

    def _auxiliary_outputs(
        self,
        batch: Dict[str, torch.Tensor],
        target_probability: torch.Tensor | None,
        ild_power_l: torch.Tensor,
        ild_power_r: torch.Tensor,
        ild_cross: torch.Tensor,
        ipd_power_l: torch.Tensor,
        ipd_power_r: torch.Tensor,
        ipd_cross: torch.Tensor,
    ) -> Dict[str, torch.Tensor | None]:
        output: Dict[str, torch.Tensor | None] = {
            "target_probability": target_probability,
            "target_mask_loss": None,
            "target_covariance_loss": None,
        }
        if target_probability is None or "target_spec_real_L" not in batch:
            return output

        time_steps = target_probability.shape[1]
        target_l, target_r = self._complex_spectra_with_prefix(
            batch, time_steps, "target_"
        )
        interferer_l, interferer_r = self._complex_spectra_with_prefix(
            batch, time_steps, "interferer_"
        )
        target_power = target_l.abs().square() + target_r.abs().square()
        interferer_power = (
            interferer_l.abs().square() + interferer_r.abs().square()
        )
        ideal_probability = target_power / (
            target_power + interferer_power + self.eps
        )
        output["target_mask_loss"] = F.binary_cross_entropy(
            target_probability.clamp(1.0e-4, 1.0 - 1.0e-4),
            ideal_probability.detach(),
        )

        target_l_windows = self._time_windows(target_l)
        target_r_windows = self._time_windows(target_r)
        target_power_l = target_l_windows.abs().square().mean(dim=-1)
        target_power_r = target_r_windows.abs().square().mean(dim=-1)
        target_cross = (
            target_l_windows * target_r_windows.conj()
        ).mean(dim=-1)
        output["target_covariance_loss"] = 0.5 * (
            self._normalized_covariance_loss(
                ild_power_l,
                ild_power_r,
                ild_cross,
                target_power_l,
                target_power_r,
                target_cross,
            )
            + self._normalized_covariance_loss(
                ipd_power_l,
                ipd_power_r,
                ipd_cross,
                target_power_l,
                target_power_r,
                target_cross,
            )
        )
        return output


class OracleSupervisedCueFactorizedCPSD(CueFactorizedCPSD):
    """B2 estimator with training-only supervision of ILD/IPD frame weights."""

    def __init__(
        self,
        time_frames: int = 5,
        score_logit_clip: float = 6.0,
        oracle_ild_scale_db: float = 6.0,
        oracle_ipd_scale_deg: float = 45.0,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__(time_frames, score_logit_clip, eps)
        if oracle_ild_scale_db <= 0.0 or oracle_ipd_scale_deg <= 0.0:
            raise ValueError("Oracle cue-error scales must be positive")
        self.oracle_ild_scale_db = float(oracle_ild_scale_db)
        self.oracle_ipd_scale_rad = math.radians(float(oracle_ipd_scale_deg))

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        time_steps: int,
    ) -> Dict[str, torch.Tensor | None]:
        output = super().forward(batch, time_steps)
        output["cue_reliability_loss"] = None
        if "target_spec_real_L" not in batch:
            return output

        mixture_l, mixture_r = self._complex_spectra(batch, time_steps)
        target_l, target_r = self._complex_spectra_with_prefix(
            batch, time_steps, "target_"
        )
        mixture_l_windows = self._time_windows(mixture_l)
        mixture_r_windows = self._time_windows(mixture_r)
        target_l_windows = self._time_windows(target_l)
        target_r_windows = self._time_windows(target_r)

        mixture_power_l = mixture_l_windows.abs().square()
        mixture_power_r = mixture_r_windows.abs().square()
        mixture_ild_db = 10.0 * torch.log10(
            (mixture_power_l + self.eps) / (mixture_power_r + self.eps)
        )
        target_power_l = target_l_windows.abs().square().mean(dim=-1)
        target_power_r = target_r_windows.abs().square().mean(dim=-1)
        target_ild_db = 10.0 * torch.log10(
            (target_power_l + self.eps) / (target_power_r + self.eps)
        )
        ild_error = (
            mixture_ild_db - target_ild_db.unsqueeze(-1)
        ).abs() / self.oracle_ild_scale_db

        mixture_cross = mixture_l_windows * mixture_r_windows.conj()
        mixture_unit = mixture_cross / (mixture_cross.abs() + self.eps)
        target_cross = (
            target_l_windows * target_r_windows.conj()
        ).mean(dim=-1)
        target_unit = target_cross / (target_cross.abs() + self.eps)
        phase_cosine = (
            mixture_unit * target_unit.conj().unsqueeze(-1)
        ).real.clamp(-1.0, 1.0)
        ipd_error = torch.acos(phase_cosine) / self.oracle_ipd_scale_rad

        oracle_ild_weights = torch.softmax(-ild_error, dim=-1).detach()
        oracle_ipd_weights = torch.softmax(-ipd_error, dim=-1).detach()
        predicted_ild_weights = output["tf_weight_ild"].clamp_min(1.0e-8)
        predicted_ipd_weights = output["tf_weight_ipd"].clamp_min(1.0e-8)
        ild_kl = torch.sum(
            oracle_ild_weights
            * (
                torch.log(oracle_ild_weights.clamp_min(1.0e-8))
                - torch.log(predicted_ild_weights)
            ),
            dim=-1,
        ).mean()
        ipd_kl = torch.sum(
            oracle_ipd_weights
            * (
                torch.log(oracle_ipd_weights.clamp_min(1.0e-8))
                - torch.log(predicted_ipd_weights)
            ),
            dim=-1,
        ).mean()
        output["cue_reliability_loss"] = 0.5 * (ild_kl + ipd_kl)
        output["oracle_tf_weight_ild"] = oracle_ild_weights
        output["oracle_tf_weight_ipd"] = oracle_ipd_weights
        return output


class NonlinearOracleSupervisedCueFactorizedCPSD(
    OracleSupervisedCueFactorizedCPSD,
    NonlinearCueFactorizedCPSD,
):
    """Nonlinear B2 estimator with training-only cue reliability targets."""

    pass


class DilatedDepthwiseTemporalBlock(nn.Module):
    """Residual depthwise-separable temporal block with a dilated receptive field."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


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
        elif encoder_type == "temporal_conv_ds_dilated":
            self.temporal_input = nn.Sequential(
                nn.Conv1d(flat_dim, temporal_hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(temporal_hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.temporal_blocks = nn.Sequential(
                *[
                    DilatedDepthwiseTemporalBlock(
                        temporal_hidden_dim,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        dropout=dropout,
                    )
                    for dilation in (1, 2, 4)
                ]
            )
            self.temporal_output = nn.Sequential(
                nn.Conv1d(temporal_hidden_dim, out_dim, kernel_size=1, bias=False),
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

    def _uniform_pool(
        self,
        cue_tensor: torch.Tensor,
        pool_weights: torch.Tensor | None = None,
        residual_alpha: torch.Tensor | float | None = None,
        eps: float = 1.0e-6,
    ) -> torch.Tensor:
        bsz, num_cues, time_steps, freq_bins = cue_tensor.shape
        flat = cue_tensor.reshape(bsz * num_cues * time_steps, 1, freq_bins)
        base = F.adaptive_avg_pool1d(flat, self.cue_bands).reshape(
            bsz, num_cues, time_steps, self.cue_bands
        )
        if pool_weights is None:
            return base
        if pool_weights.shape != (bsz, 1, time_steps, freq_bins):
            raise ValueError(
                "pool_weights must have shape [B, 1, T, F], got "
                f"{tuple(pool_weights.shape)}"
            )

        weighted_bands = []
        for band_idx in range(self.cue_bands):
            start = (band_idx * freq_bins) // self.cue_bands
            end = ((band_idx + 1) * freq_bins + self.cue_bands - 1) // self.cue_bands
            weight = pool_weights[..., start:end]
            numerator = (cue_tensor[..., start:end] * weight).sum(dim=-1)
            denominator = weight.sum(dim=-1).clamp_min(eps)
            weighted_bands.append(numerator / denominator)
        weighted = torch.stack(weighted_bands, dim=-1)
        if residual_alpha is None:
            return weighted
        return base + residual_alpha * (weighted - base)

    def forward(
        self,
        cue_tensor: torch.Tensor,
        pool_weights: torch.Tensor | None = None,
        residual_alpha: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        # cue_tensor: [B, C, T, F]
        bsz, num_cues, time_steps, freq_bins = cue_tensor.shape
        if self.band_mode == "uniform":
            x = self._uniform_pool(cue_tensor, pool_weights, residual_alpha)
        elif self.learnable_band_projection:
            if pool_weights is not None:
                raise ValueError("weighted pooling currently requires cue_band_mode='uniform'")
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
        if self.encoder_type == "temporal_conv_ds_dilated":
            x = x.transpose(1, 2)  # [B, C*bands, T]
            x = self.temporal_output(self.temporal_blocks(self.temporal_input(x)))
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


class SplitBandResidualProjection(nn.Module):
    """Learn a low/high-frequency projection while retaining uniform pooling."""

    def __init__(
        self,
        freq_bins: int,
        cue_bands: int,
        split_bin: int,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        if not 1 <= split_bin < freq_bins:
            raise ValueError(
                f"split_bin must be in [1, {freq_bins - 1}], got {split_bin}"
            )
        self.freq_bins = int(freq_bins)
        self.cue_bands = int(cue_bands)
        self.split_bin = int(split_bin)
        self.residual_scale = float(residual_scale)
        self.low_projection = nn.Linear(self.split_bin, self.cue_bands, bias=False)
        self.high_projection = nn.Linear(
            self.freq_bins - self.split_bin,
            self.cue_bands,
            bias=False,
        )
        self.residual_alpha_raw = nn.Parameter(torch.tensor(0.1))

        with torch.no_grad():
            self.low_projection.weight.copy_(
                _adaptive_uniform_filterbank(self.cue_bands, self.split_bin)
            )
            self.high_projection.weight.copy_(
                _adaptive_uniform_filterbank(
                    self.cue_bands,
                    self.freq_bins - self.split_bin,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.freq_bins:
            raise ValueError(
                f"Split-band projection expected {self.freq_bins} bins, got {x.shape[-1]}"
            )
        base = F.adaptive_avg_pool2d(
            x,
            output_size=(x.shape[2], self.cue_bands),
        )
        low = self.low_projection(x[..., : self.split_bin])
        high = self.high_projection(x[..., self.split_bin :])
        projected = 0.5 * (low + high)
        alpha = self.residual_scale * torch.tanh(self.residual_alpha_raw)
        return base + alpha * projected


class FineToCoarseSubbandRefinement(nn.Module):
    """Refine a coarse uniform filterbank with adjacent fine-band detail."""

    def __init__(
        self,
        channels: int,
        coarse_bands: int,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        if channels <= 0 or coarse_bands <= 0:
            raise ValueError("channels and coarse_bands must be positive")
        self.coarse_bands = int(coarse_bands)
        self.fine_bands = 2 * self.coarse_bands
        self.residual_scale = float(residual_scale)
        self.detail_gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Tanh(),
        )
        # Start exactly from the existing uniform coarse-band representation.
        self.residual_alpha_raw = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_steps = x.shape[2]
        coarse = F.adaptive_avg_pool2d(
            x,
            output_size=(time_steps, self.coarse_bands),
        )
        fine = F.adaptive_avg_pool2d(
            x,
            output_size=(time_steps, self.fine_bands),
        )
        detail = 0.5 * (fine[..., 0::2] - fine[..., 1::2])
        gate = self.detail_gate(torch.cat([coarse, detail], dim=1))
        alpha = self.residual_scale * torch.tanh(self.residual_alpha_raw)
        return coarse + alpha * gate * detail


class FrequencyBlurPool(nn.Module):
    """Low-pass each feature channel before stride-2 frequency downsampling."""

    def __init__(self):
        super().__init__()
        kernel = torch.tensor([1.0, 2.0, 1.0]).view(1, 1, 1, 3) / 4.0
        self.register_buffer("kernel", kernel, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (1, 1, 0, 0), mode="replicate")
        kernel = self.kernel.to(device=x.device, dtype=x.dtype).expand(
            channels, 1, 1, 3
        )
        return F.conv2d(x, kernel, stride=(1, 2), groups=channels)


class AnisotropicProgressiveCueBlock(nn.Module):
    """Separate local spectral and temporal modeling before frequency reduction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spectral_kernel_size: int,
        temporal_dilation: int,
        dropout: float,
    ):
        super().__init__()
        if spectral_kernel_size < 1 or spectral_kernel_size % 2 == 0:
            raise ValueError("spectral_kernel_size must be a positive odd integer")
        if temporal_dilation < 1:
            raise ValueError("temporal_dilation must be positive")

        self.spectral = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(1, spectral_kernel_size),
                padding=(0, spectral_kernel_size // 2),
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.temporal = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(3, 1),
                padding=(temporal_dilation, 0),
                dilation=(temporal_dilation, 1),
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout),
        )
        self.residual = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.SiLU(inplace=True)
        self.downsample = FrequencyBlurPool()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.temporal(self.spectral(x))
        return self.downsample(self.activation(x + residual))


class CueSpecificProgressiveTFEncoder(nn.Module):
    """Progressively encode ILD/IPD maps and aggregate ordered frequency evidence."""

    _AGGREGATION_MODES = {"mean", "attention", "coherence_attention"}

    def __init__(
        self,
        aggregation_mode: str,
        channels: Sequence[int] = (8, 12, 16),
        temporal_dilations: Sequence[int] = (1, 2, 4),
        ild_spectral_kernel_size: int = 7,
        ipd_spectral_kernel_size: int = 3,
        ild_out_dim: int = 8,
        ipd_out_dim: int = 16,
        out_dim: int = 32,
        ild_scale_db: float = 20.0,
        ild_clip_db: float = 40.0,
        coherence_beta_init: float = 0.5,
        dropout: float = 0.2,
        eps: float = 1.0e-4,
    ):
        super().__init__()
        if aggregation_mode not in self._AGGREGATION_MODES:
            raise ValueError(f"Unsupported progressive aggregation: {aggregation_mode}")
        channels = tuple(int(channel) for channel in channels)
        temporal_dilations = tuple(int(value) for value in temporal_dilations)
        if not channels or len(channels) != len(temporal_dilations):
            raise ValueError("channels and temporal_dilations must have equal non-zero length")
        if coherence_beta_init <= 0.0:
            raise ValueError("coherence_beta_init must be positive")

        self.aggregation_mode = aggregation_mode
        self.ild_scale_db = float(ild_scale_db)
        self.ild_clip_db = float(ild_clip_db)
        self.out_dim = int(out_dim)
        self.eps = float(eps)
        first_channels = channels[0]
        self.ild_stem = self._make_stem(1, first_channels)
        self.ipd_stem = self._make_stem(2, first_channels)
        self.ild_blocks = self._make_blocks(
            channels,
            temporal_dilations,
            ild_spectral_kernel_size,
            dropout,
        )
        self.ipd_blocks = self._make_blocks(
            channels,
            temporal_dilations,
            ipd_spectral_kernel_size,
            dropout,
        )
        self.reliability_downsampling = nn.Sequential(
            *(FrequencyBlurPool() for _ in channels)
        )
        final_channels = channels[-1]
        self.ild_attention = nn.Conv2d(final_channels, 1, kernel_size=1)
        self.ipd_attention = nn.Conv2d(final_channels, 1, kernel_size=1)
        beta_raw = torch.log(torch.expm1(torch.tensor(float(coherence_beta_init))))
        self.coherence_beta_raw = nn.Parameter(beta_raw)
        self.ild_projection = nn.Linear(final_channels, ild_out_dim)
        self.ipd_projection = nn.Linear(final_channels, ipd_out_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(ild_out_dim + ipd_out_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _make_stem(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def _make_blocks(
        channels: Sequence[int],
        temporal_dilations: Sequence[int],
        spectral_kernel_size: int,
        dropout: float,
    ) -> nn.ModuleList:
        blocks = []
        in_channels = channels[0]
        for out_channels, dilation in zip(channels, temporal_dilations):
            blocks.append(
                AnisotropicProgressiveCueBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    spectral_kernel_size=spectral_kernel_size,
                    temporal_dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        return nn.ModuleList(blocks)

    def _aggregate(
        self,
        x: torch.Tensor,
        scorer: nn.Module,
        coherence: torch.Tensor,
    ) -> torch.Tensor:
        if self.aggregation_mode == "mean":
            return x.mean(dim=-1).transpose(1, 2)

        score = scorer(x)
        if self.aggregation_mode == "coherence_attention":
            beta = F.softplus(self.coherence_beta_raw)
            score = score + beta * torch.log(coherence.clamp_min(self.eps))
        weights = torch.softmax(score, dim=-1)
        return torch.sum(weights * x, dim=-1).transpose(1, 2)

    def forward(
        self,
        value_tensor: torch.Tensor,
        reliability_tensor: torch.Tensor,
    ) -> torch.Tensor:
        ild = value_tensor[:, 0:1].clamp(-self.ild_clip_db, self.ild_clip_db)
        ild = ild / self.ild_scale_db
        ipd = value_tensor[:, 1:3]
        ild = self.ild_stem(ild)
        ipd = self.ipd_stem(ipd)
        for block in self.ild_blocks:
            ild = block(ild)
        for block in self.ipd_blocks:
            ipd = block(ipd)
        coherence = self.reliability_downsampling(
            reliability_tensor.clamp(0.0, 1.0)
        )
        if coherence.shape[-1] != ild.shape[-1]:
            raise RuntimeError("Progressive cue and coherence frequency sizes diverged")

        ild_feat = self.ild_projection(
            self._aggregate(ild, self.ild_attention, coherence)
        )
        ipd_feat = self.ipd_projection(
            self._aggregate(ipd, self.ipd_attention, coherence)
        )
        return self.output_projection(torch.cat([ild_feat, ipd_feat], dim=-1))


class AnisotropicOrderedCueBlock(nn.Module):
    """Residual local T-F block that keeps the frequency grid unchanged."""

    def __init__(
        self,
        channels: int,
        spectral_kernel_size: int,
        temporal_kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        if spectral_kernel_size < 1 or spectral_kernel_size % 2 == 0:
            raise ValueError("spectral_kernel_size must be a positive odd integer")
        if temporal_kernel_size < 1 or temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be a positive odd integer")
        self.spectral = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, spectral_kernel_size),
                padding=(0, spectral_kernel_size // 2),
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.temporal = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(temporal_kernel_size, 1),
                padding=(temporal_kernel_size // 2, 0),
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Dropout2d(dropout),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.temporal(self.spectral(x)))


class NarrowBandTemporalCueStabilizer(nn.Module):
    """Learn a per-frequency temporal residual without mixing frequency bins."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 8,
        kernel_size: int = 5,
    ):
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("temporal stabilizer kernel must be an odd integer >= 3")
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=(kernel_size, 1),
                padding=(kernel_size // 2, 0),
                groups=hidden_channels,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AxisResidualBiGRU(nn.Module):
    """Apply one shared BiGRU along time or ordered frequency bands."""

    def __init__(self, channels: int, axis: str):
        super().__init__()
        if axis not in {"time", "frequency"}:
            raise ValueError(f"Unsupported GRU axis: {axis}")
        if channels < 2:
            raise ValueError("AxisResidualBiGRU requires at least two channels")
        hidden_size = max(channels // 2, 1)
        self.axis = axis
        self.gru = nn.GRU(
            input_size=channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        gru_out_dim = 2 * hidden_size
        self.proj = nn.Identity() if gru_out_dim == channels else nn.Linear(gru_out_dim, channels)
        self.norm = nn.LayerNorm(channels)
        self.alpha_raw = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, time_steps, bands = x.shape
        if self.axis == "time":
            sequence = x.permute(0, 3, 2, 1).reshape(bsz * bands, time_steps, channels)
            residual, _ = self.gru(sequence)
            residual = self.norm(self.proj(residual))
            residual = residual.reshape(bsz, bands, time_steps, channels).permute(0, 3, 2, 1)
        else:
            sequence = x.permute(0, 2, 3, 1).reshape(bsz * time_steps, bands, channels)
            residual, _ = self.gru(sequence)
            residual = self.norm(self.proj(residual))
            residual = residual.reshape(bsz, time_steps, bands, channels).permute(0, 3, 1, 2)
        return x + torch.tanh(self.alpha_raw) * residual


class CueSpecificLocalTFValueEncoder(nn.Module):
    """Separate local T-F encoders for bounded ILD and circular IPD cues."""

    def __init__(
        self,
        cue_bands: int = 16,
        ild_out_dim: int = 8,
        ipd_out_dim: int = 16,
        hidden_channels: int = 8,
        temporal_kernel_size: int = 3,
        ild_scale_db: float = 20.0,
        ild_clip_db: float = 40.0,
        freq_bins: int = 257,
        sample_rate: int = 16000,
        use_band_projection: bool = False,
        band_split_hz: float = 1500.0,
        band_projection_residual_scale: float = 1.0,
        use_joint_correction: bool = False,
        joint_correction_residual_scale: float = 1.0,
        use_coherence_context: bool = True,
        use_cue_consistency_context: bool = False,
        use_fine_to_coarse_refinement: bool = False,
        fine_to_coarse_residual_scale: float = 1.0,
        local_block_type: str = "standard",
        ild_spectral_kernel_size: int = 7,
        ipd_spectral_kernel_size: int = 3,
        temporal_stabilizer_type: str = "none",
        temporal_stabilizer_hidden_channels: int = 8,
        temporal_stabilizer_kernel_size: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        if ild_scale_db <= 0 or ild_clip_db <= 0:
            raise ValueError("ILD scale and clipping values must be positive")
        self.cue_bands = int(cue_bands)
        self.ild_scale_db = float(ild_scale_db)
        self.ild_clip_db = float(ild_clip_db)
        self.freq_bins = int(freq_bins)
        self.sample_rate = int(sample_rate)
        self.use_band_projection = bool(use_band_projection)
        self.use_joint_correction = bool(use_joint_correction)
        self.use_coherence_context = bool(use_coherence_context)
        self.use_cue_consistency_context = bool(use_cue_consistency_context)
        if self.use_coherence_context and self.use_cue_consistency_context:
            raise ValueError(
                "coherence context and cue-specific consistency context are mutually exclusive"
            )
        self.use_fine_to_coarse_refinement = bool(use_fine_to_coarse_refinement)
        valid_stabilizers = {
            "none",
            "preconv",
            "postband_gru",
            "postband_gru_fullband",
        }
        if temporal_stabilizer_type not in valid_stabilizers:
            raise ValueError(
                f"Unsupported cue temporal stabilizer: {temporal_stabilizer_type}"
            )
        self.temporal_stabilizer_type = temporal_stabilizer_type
        if local_block_type not in {"standard", "anisotropic_residual"}:
            raise ValueError(f"Unsupported cue-specific local block: {local_block_type}")
        self.local_block_type = local_block_type
        if self.use_band_projection and self.use_fine_to_coarse_refinement:
            raise ValueError(
                "band projection and fine-to-coarse refinement are mutually exclusive"
            )
        split_bin = round(
            float(band_split_hz) * (self.freq_bins - 1) / (self.sample_rate / 2.0)
        )
        split_bin = max(1, min(int(split_bin), self.freq_bins - 1))
        has_local_context = (
            self.use_coherence_context or self.use_cue_consistency_context
        )
        self.ild_encoder = self._make_branch(
            in_channels=2 if has_local_context else 1,
            hidden_channels=hidden_channels,
            out_dim=ild_out_dim,
            temporal_kernel_size=temporal_kernel_size,
            spectral_kernel_size=ild_spectral_kernel_size,
            dropout=dropout,
        )
        self.ipd_encoder = self._make_branch(
            in_channels=3 if has_local_context else 2,
            hidden_channels=hidden_channels,
            out_dim=ipd_out_dim,
            temporal_kernel_size=temporal_kernel_size,
            spectral_kernel_size=ipd_spectral_kernel_size,
            dropout=dropout,
        )
        if self.temporal_stabilizer_type == "preconv":
            self.ild_pre_stabilizer = NarrowBandTemporalCueStabilizer(
                in_channels=2 if has_local_context else 1,
                out_channels=1,
                hidden_channels=temporal_stabilizer_hidden_channels,
                kernel_size=temporal_stabilizer_kernel_size,
            )
            self.ipd_pre_stabilizer = NarrowBandTemporalCueStabilizer(
                in_channels=3 if has_local_context else 2,
                out_channels=2,
                hidden_channels=temporal_stabilizer_hidden_channels,
                kernel_size=temporal_stabilizer_kernel_size,
            )
        else:
            self.ild_pre_stabilizer = None
            self.ipd_pre_stabilizer = None
        if self.use_band_projection:
            self.ild_encoder["band_projection"] = SplitBandResidualProjection(
                freq_bins=self.freq_bins,
                cue_bands=self.cue_bands,
                split_bin=split_bin,
                residual_scale=band_projection_residual_scale,
            )
            self.ipd_encoder["band_projection"] = SplitBandResidualProjection(
                freq_bins=self.freq_bins,
                cue_bands=self.cue_bands,
                split_bin=split_bin,
                residual_scale=band_projection_residual_scale,
            )
        if self.use_fine_to_coarse_refinement:
            self.ild_encoder["fine_to_coarse"] = FineToCoarseSubbandRefinement(
                channels=hidden_channels,
                coarse_bands=self.cue_bands,
                residual_scale=fine_to_coarse_residual_scale,
            )
            self.ipd_encoder["fine_to_coarse"] = FineToCoarseSubbandRefinement(
                channels=hidden_channels,
                coarse_bands=self.cue_bands,
                residual_scale=fine_to_coarse_residual_scale,
            )
        if self.use_joint_correction:
            self.ild_to_ipd = nn.Linear(ild_out_dim, ipd_out_dim, bias=False)
            self.joint_correction = nn.Sequential(
                nn.Linear(ipd_out_dim * 3, ipd_out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(ipd_out_dim, ipd_out_dim),
            )
            self.joint_correction_alpha_raw = nn.Parameter(torch.tensor(0.1))
            self.joint_correction_residual_scale = float(
                joint_correction_residual_scale
            )
        else:
            self.ild_to_ipd = None
            self.joint_correction = None
            self.register_parameter("joint_correction_alpha_raw", None)
            self.joint_correction_residual_scale = 0.0
    def _make_branch(
        self,
        in_channels: int,
        hidden_channels: int,
        out_dim: int,
        temporal_kernel_size: int,
        spectral_kernel_size: int,
        dropout: float,
    ) -> nn.ModuleDict:
        padding = temporal_kernel_size // 2
        if self.local_block_type == "anisotropic_residual":
            local = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU(inplace=True),
                AnisotropicOrderedCueBlock(
                    channels=hidden_channels,
                    spectral_kernel_size=spectral_kernel_size,
                    temporal_kernel_size=temporal_kernel_size,
                    dropout=dropout,
                ),
            )
        else:
            local = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=1,
                    groups=hidden_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU(inplace=True),
                nn.Dropout2d(dropout),
            )
        modules = {
            "local": local,
            "temporal": nn.Sequential(
                nn.Conv1d(
                    hidden_channels * self.cue_bands,
                    out_dim,
                    kernel_size=temporal_kernel_size,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
            ),
        }
        if self.temporal_stabilizer_type in {"postband_gru", "postband_gru_fullband"}:
            modules["narrowband_gru"] = AxisResidualBiGRU(hidden_channels, axis="time")
        if self.temporal_stabilizer_type == "postband_gru_fullband":
            modules["fullband_gru"] = AxisResidualBiGRU(hidden_channels, axis="frequency")
        return nn.ModuleDict(modules)

    def _encode(self, x: torch.Tensor, branch: nn.ModuleDict) -> torch.Tensor:
        x = branch["local"](x)
        if "band_projection" in branch:
            x = branch["band_projection"](x)
        elif "fine_to_coarse" in branch:
            x = branch["fine_to_coarse"](x)
        else:
            x = F.adaptive_avg_pool2d(x, output_size=(x.shape[2], self.cue_bands))
        if "narrowband_gru" in branch:
            x = branch["narrowband_gru"](x)
        if "fullband_gru" in branch:
            x = branch["fullband_gru"](x)
        bsz, channels, time_steps, bands = x.shape
        x = x.permute(0, 1, 3, 2).reshape(bsz, channels * bands, time_steps)
        return branch["temporal"](x).transpose(1, 2)

    def forward(
        self,
        value_tensor: torch.Tensor,
        reliability_tensor: torch.Tensor,
        ild_consistency_tensor: torch.Tensor | None = None,
        ipd_consistency_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ild = value_tensor[:, 0:1].clamp(-self.ild_clip_db, self.ild_clip_db)
        ild = ild / self.ild_scale_db
        ipd = value_tensor[:, 1:3]
        if self.use_coherence_context:
            ild_input = torch.cat([ild, reliability_tensor], dim=1)
            ipd_input = torch.cat([ipd, reliability_tensor], dim=1)
        elif self.use_cue_consistency_context:
            if ild_consistency_tensor is None or ipd_consistency_tensor is None:
                raise ValueError(
                    "cue-specific consistency context requires both ILD and IPD tensors"
                )
            if ild_consistency_tensor.shape != reliability_tensor.shape:
                raise ValueError(
                    "ILD consistency must match coherence shape, got "
                    f"{tuple(ild_consistency_tensor.shape)} and "
                    f"{tuple(reliability_tensor.shape)}"
                )
            if ipd_consistency_tensor.shape != reliability_tensor.shape:
                raise ValueError(
                    "IPD consistency must match coherence shape, got "
                    f"{tuple(ipd_consistency_tensor.shape)} and "
                    f"{tuple(reliability_tensor.shape)}"
                )
            ild_input = torch.cat([ild, ild_consistency_tensor], dim=1)
            ipd_input = torch.cat([ipd, ipd_consistency_tensor], dim=1)
        else:
            ild_input = ild
            ipd_input = ipd
        if self.ild_pre_stabilizer is not None:
            ild = ild + self.ild_pre_stabilizer(ild_input)
            ipd = ipd + self.ipd_pre_stabilizer(ipd_input)
            ipd = F.normalize(ipd, p=2, dim=1, eps=1e-6)
            if self.use_coherence_context:
                ild_input = torch.cat([ild, reliability_tensor], dim=1)
                ipd_input = torch.cat([ipd, reliability_tensor], dim=1)
            elif self.use_cue_consistency_context:
                ild_input = torch.cat([ild, ild_consistency_tensor], dim=1)
                ipd_input = torch.cat([ipd, ipd_consistency_tensor], dim=1)
            else:
                ild_input = ild
                ipd_input = ipd
        ild_feat = self._encode(ild_input, self.ild_encoder)
        ipd_feat = self._encode(ipd_input, self.ipd_encoder)
        if self.use_joint_correction:
            ild_context = self.ild_to_ipd(ild_feat)
            correction = self.joint_correction(
                torch.cat(
                    [ipd_feat, ild_context, ipd_feat * ild_context],
                    dim=-1,
                )
            )
            alpha = self.joint_correction_residual_scale * torch.tanh(
                self.joint_correction_alpha_raw
            )
            ipd_feat = ipd_feat + alpha * correction
        return torch.cat([ild_feat, ipd_feat], dim=-1)


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
        use_precompression_reliability_pooling: bool = False,
        precompression_pool_hidden_channels: int = 8,
        precompression_pool_residual_scale: float = 1.0,
        cue_specific_local_hidden_channels: int = 8,
        cue_specific_local_ild_scale_db: float = 20.0,
        cue_specific_local_ild_clip_db: float = 40.0,
        cue_specific_local_use_band_projection: bool = False,
        cue_specific_local_band_split_hz: float = 1500.0,
        cue_specific_local_band_projection_residual_scale: float = 1.0,
        cue_specific_local_use_joint_correction: bool = False,
        cue_specific_local_joint_correction_residual_scale: float = 1.0,
        cue_specific_local_use_coherence_context: bool = True,
        cue_specific_local_use_cue_consistency_context: bool = False,
        cue_specific_local_use_standalone_coherence: bool = True,
        cue_specific_local_use_fine_to_coarse_refinement: bool = False,
        cue_specific_local_fine_to_coarse_residual_scale: float = 1.0,
        cue_specific_local_block_type: str = "standard",
        cue_specific_local_ild_spectral_kernel_size: int = 7,
        cue_specific_local_ipd_spectral_kernel_size: int = 3,
        cue_specific_local_temporal_stabilizer_type: str = "none",
        cue_specific_local_temporal_stabilizer_hidden_channels: int = 8,
        cue_specific_local_temporal_stabilizer_kernel_size: int = 5,
        cue_progressive_aggregation: str = "mean",
        cue_progressive_channels: Sequence[int] = (8, 12, 16),
        cue_progressive_temporal_dilations: Sequence[int] = (1, 2, 4),
        cue_progressive_ild_kernel_size: int = 7,
        cue_progressive_ipd_kernel_size: int = 3,
        cue_progressive_out_dim: int = 32,
        cue_progressive_coherence_beta_init: float = 0.5,
    ):
        super().__init__()
        if fusion_mode not in {
            "concat",
            "gate",
            "reliability_weighted_concat",
            "rel_film_value",
            "residual_product_concat",
        }:
            raise ValueError(f"Unsupported DualBranchCueEncoder fusion_mode: {fusion_mode}")
        if branch_mode not in {
            "dual",
            "merged",
            "cue_specific_resolution",
            "local_tf",
            "dual_local_tf",
            "dual_local_tf_gate",
            "cue_specific_local_tf",
            "cue_specific_progressive_tf",
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
        if branch_mode in {"cue_specific_local_tf", "cue_specific_progressive_tf"}:
            if fusion_mode != "concat" or cue_band_mode != "uniform":
                raise ValueError(
                    f"{branch_mode} requires fusion_mode='concat' and cue_band_mode='uniform'"
                )
        if branch_mode == "local_tf" and fusion_mode != "concat":
            raise ValueError("branch_mode='local_tf' only supports fusion_mode='concat'")
        if fusion_mode == "reliability_weighted_concat" and disable_reliability_branch:
            raise ValueError("reliability_weighted_concat requires reliability branch")
        if fusion_mode == "rel_film_value" and disable_reliability_branch:
            raise ValueError("rel_film_value requires reliability branch")
        if fusion_mode == "residual_product_concat" and disable_reliability_branch:
            raise ValueError("residual_product_concat requires reliability branch")
        if fusion_mode == "residual_product_concat" and branch_mode != "dual":
            raise ValueError("residual_product_concat currently requires branch_mode='dual'")

        self.fusion_mode = fusion_mode
        self.reliability_weight_scale = reliability_weight_scale
        self.branch_mode = branch_mode
        self.disable_reliability_branch = disable_reliability_branch
        self.cue_band_mode = cue_band_mode
        self.use_tf_mask = use_tf_mask
        self.tf_mask_residual_scale = tf_mask_residual_scale
        self.use_precompression_reliability_pooling = use_precompression_reliability_pooling
        self.precompression_pool_residual_scale = precompression_pool_residual_scale
        self.cue_ild_bands = cue_ild_bands
        self.cue_ipd_bands = cue_ipd_bands
        self.cue_coherence_bands = cue_coherence_bands
        self.cue_ild_out_dim = cue_ild_out_dim
        self.cue_ipd_out_dim = cue_ipd_out_dim
        self.value_out_dim = value_out_dim
        self.reliability_out_dim = reliability_out_dim
        self.cue_specific_local_use_coherence_context = bool(
            cue_specific_local_use_coherence_context
        )
        self.cue_specific_local_use_cue_consistency_context = bool(
            cue_specific_local_use_cue_consistency_context
        )
        self.cue_specific_local_use_standalone_coherence = bool(
            cue_specific_local_use_standalone_coherence
        ) and not disable_reliability_branch
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
        if use_precompression_reliability_pooling:
            if branch_mode != "dual" or cue_band_mode != "uniform":
                raise ValueError(
                    "pre-compression reliability pooling requires "
                    "branch_mode='dual' and cue_band_mode='uniform'"
                )
            if disable_reliability_branch:
                raise ValueError("pre-compression reliability pooling requires coherence")
            self.precompression_weight_net = nn.Sequential(
                nn.Conv2d(2, 2, kernel_size=3, padding=1, groups=2, bias=False),
                nn.BatchNorm2d(2),
                nn.ReLU(inplace=True),
                nn.Conv2d(2, precompression_pool_hidden_channels, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(precompression_pool_hidden_channels, 1, kernel_size=1),
                nn.Sigmoid(),
            )
            self.precompression_pool_alpha_raw = nn.Parameter(torch.zeros(()))
        else:
            self.precompression_weight_net = None
            self.register_parameter("precompression_pool_alpha_raw", None)
        if branch_mode == "cue_specific_local_tf":
            self.cue_specific_local_value_encoder = CueSpecificLocalTFValueEncoder(
                cue_bands=cue_bands,
                ild_out_dim=cue_ild_out_dim,
                ipd_out_dim=cue_ipd_out_dim,
                hidden_channels=cue_specific_local_hidden_channels,
                temporal_kernel_size=kernel_size,
                ild_scale_db=cue_specific_local_ild_scale_db,
                ild_clip_db=cue_specific_local_ild_clip_db,
                freq_bins=cue_freq_bins,
                sample_rate=cue_sample_rate,
                use_band_projection=cue_specific_local_use_band_projection,
                band_split_hz=cue_specific_local_band_split_hz,
                band_projection_residual_scale=cue_specific_local_band_projection_residual_scale,
                use_joint_correction=cue_specific_local_use_joint_correction,
                joint_correction_residual_scale=cue_specific_local_joint_correction_residual_scale,
                use_coherence_context=self.cue_specific_local_use_coherence_context,
                use_cue_consistency_context=(
                    self.cue_specific_local_use_cue_consistency_context
                ),
                use_fine_to_coarse_refinement=cue_specific_local_use_fine_to_coarse_refinement,
                fine_to_coarse_residual_scale=cue_specific_local_fine_to_coarse_residual_scale,
                local_block_type=cue_specific_local_block_type,
                ild_spectral_kernel_size=cue_specific_local_ild_spectral_kernel_size,
                ipd_spectral_kernel_size=cue_specific_local_ipd_spectral_kernel_size,
                temporal_stabilizer_type=cue_specific_local_temporal_stabilizer_type,
                temporal_stabilizer_hidden_channels=(
                    cue_specific_local_temporal_stabilizer_hidden_channels
                ),
                temporal_stabilizer_kernel_size=(
                    cue_specific_local_temporal_stabilizer_kernel_size
                ),
                dropout=dropout,
            )
        else:
            self.cue_specific_local_value_encoder = None
        if branch_mode == "cue_specific_progressive_tf":
            self.cue_specific_progressive_encoder = CueSpecificProgressiveTFEncoder(
                aggregation_mode=cue_progressive_aggregation,
                channels=cue_progressive_channels,
                temporal_dilations=cue_progressive_temporal_dilations,
                ild_spectral_kernel_size=cue_progressive_ild_kernel_size,
                ipd_spectral_kernel_size=cue_progressive_ipd_kernel_size,
                ild_out_dim=cue_ild_out_dim,
                ipd_out_dim=cue_ipd_out_dim,
                out_dim=cue_progressive_out_dim,
                ild_scale_db=cue_specific_local_ild_scale_db,
                ild_clip_db=cue_specific_local_ild_clip_db,
                coherence_beta_init=cue_progressive_coherence_beta_init,
                dropout=dropout,
            )
        else:
            self.cue_specific_progressive_encoder = None
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
        elif branch_mode in {"cue_specific_local_tf", "cue_specific_progressive_tf"}:
            self.local_tf_encoder = None
            self.merged_encoder = None
            self.value_encoder = None
            self.ild_encoder = None
            self.ipd_encoder = None
            self.reliability_encoder = (
                LiteCueEncoder(
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
                if branch_mode == "cue_specific_local_tf"
                and self.cue_specific_local_use_standalone_coherence
                else None
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
        if fusion_mode == "residual_product_concat" and not disable_reliability_branch:
            self.rel_to_product = nn.Sequential(
                nn.Linear(reliability_out_dim, value_out_dim),
                nn.Sigmoid(),
            )
        else:
            self.rel_to_product = None

    @property
    def out_dim(self) -> int:
        if self.branch_mode == "cue_specific_progressive_tf":
            return self.cue_specific_progressive_encoder.out_dim
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
        if self.fusion_mode == "residual_product_concat":
            return 2 * self.value_out_dim + self.reliability_out_dim
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
        magnitude_context: torch.Tensor | None = None,
        ild_consistency_tensor: torch.Tensor | None = None,
        ipd_consistency_tensor: torch.Tensor | None = None,
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

        if self.use_precompression_reliability_pooling:
            if magnitude_context is None:
                raise ValueError("magnitude_context is required for pre-compression pooling")
            if magnitude_context.shape != reliability_tensor.shape:
                raise ValueError(
                    "magnitude_context must match reliability_tensor shape, got "
                    f"{tuple(magnitude_context.shape)} and {tuple(reliability_tensor.shape)}"
                )
            magnitude_mean = magnitude_context.mean(dim=-1, keepdim=True)
            magnitude_std = magnitude_context.std(dim=-1, keepdim=True, unbiased=False)
            magnitude_norm = (magnitude_context - magnitude_mean) / magnitude_std.clamp_min(1.0e-5)
            pool_weights = self.precompression_weight_net(
                torch.cat([magnitude_norm, reliability_tensor], dim=1)
            )
            pool_alpha = self.precompression_pool_residual_scale * torch.tanh(
                self.precompression_pool_alpha_raw
            )
        else:
            pool_weights = None
            pool_alpha = None

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
        if self.branch_mode == "cue_specific_local_tf":
            value_feat = self.cue_specific_local_value_encoder(
                value_tensor,
                reliability_tensor,
                ild_consistency_tensor=ild_consistency_tensor,
                ipd_consistency_tensor=ipd_consistency_tensor,
            )
            reliability_feat = (
                self.reliability_encoder(reliability_tensor)
                if self.reliability_encoder is not None
                else None
            )
            cue_feat = (
                torch.cat([value_feat, reliability_feat], dim=-1)
                if reliability_feat is not None
                else value_feat
            )
            return {
                "cue_feat": cue_feat,
                "cue_value_feat": value_feat,
                "cue_reliability_feat": reliability_feat,
                "cue_gate": None,
                "cue_tf_mask": tf_mask,
                "cue_tf_weight": None,
                "cue_pool_alpha": None,
            }
        if self.branch_mode == "cue_specific_progressive_tf":
            cue_feat = self.cue_specific_progressive_encoder(
                value_tensor,
                reliability_tensor,
            )
            return {
                "cue_feat": cue_feat,
                "cue_value_feat": cue_feat,
                "cue_reliability_feat": None,
                "cue_gate": None,
                "cue_tf_mask": tf_mask,
                "cue_tf_weight": None,
                "cue_pool_alpha": None,
            }
        value_feat = self.value_encoder(
            value_tensor,
            pool_weights=pool_weights,
            residual_alpha=pool_alpha,
        )
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
            elif self.fusion_mode == "residual_product_concat":
                gate = self.rel_to_product(reliability_feat)
                product_feat = value_feat * gate
                cue_feat = torch.cat([value_feat, reliability_feat, product_feat], dim=-1)
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
            "cue_tf_weight": pool_weights,
            "cue_pool_alpha": pool_alpha,
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
        class_angles_deg: Sequence[float] | None = None,
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
                class_angles_deg=class_angles_deg,
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
        ild = batch.get("ild")
        ipd = batch.get("ipd")

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
        ild = batch.get("ild")
        ipd = batch.get("ipd")

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


class ComplexConv2d(nn.Module):
    """Complex convolution implemented with coupled real-valued convolutions."""

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.real = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.imag = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real = self.real(x.real) - self.imag(x.imag)
        imag = self.real(x.imag) + self.imag(x.real)
        return torch.complex(real, imag)


class ComplexDepthwiseSeparableTFBlock(nn.Module):
    """Residual complex T-F block with depthwise and pointwise convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dilation: int,
    ):
        super().__init__()
        self.depthwise = ComplexConv2d(
            in_channels,
            in_channels,
            kernel_size=(3, 5),
            padding=(time_dilation, 2),
            dilation=(time_dilation, 1),
            groups=in_channels,
            bias=False,
        )
        self.pointwise = ComplexConv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.norm_real = nn.GroupNorm(1, out_channels)
        self.norm_imag = nn.GroupNorm(1, out_channels)
        self.residual = (
            ComplexConv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.pointwise(self.depthwise(x))
        x = torch.complex(self.norm_real(x.real), self.norm_imag(x.imag))
        x = torch.complex(F.silu(x.real), F.silu(x.imag))
        return x + residual


class RawComplexTFCueEncoder(nn.Module):
    """Jointly encode the two raw complex STFTs without handcrafted binaural cues."""

    def __init__(
        self,
        out_dim: int,
        channels: Sequence[int] = (8, 12, 16),
        pooled_freq_bins: int = 4,
        dropout: float = 0.2,
        eps: float = 1.0e-6,
    ):
        super().__init__()
        if len(channels) != 3:
            raise ValueError("RawComplexTFCueEncoder expects exactly three channel stages")
        if pooled_freq_bins < 1:
            raise ValueError("pooled_freq_bins must be positive")
        self.eps = eps
        blocks = []
        in_channels = 2
        for out_channels, dilation in zip(channels, (1, 2, 4)):
            blocks.append(
                ComplexDepthwiseSeparableTFBlock(
                    in_channels=in_channels,
                    out_channels=int(out_channels),
                    time_dilation=dilation,
                )
            )
            in_channels = int(out_channels)
        self.blocks = nn.ModuleList(blocks)
        self.pooled_freq_bins = int(pooled_freq_bins)
        projection_dim = 3 * in_channels * self.pooled_freq_bins
        self.projection = nn.Sequential(
            nn.Linear(projection_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.out_dim = out_dim

    @staticmethod
    def _frequency_pool(x: torch.Tensor) -> torch.Tensor:
        real = F.avg_pool2d(x.real, kernel_size=(1, 2), stride=(1, 2))
        imag = F.avg_pool2d(x.imag, kernel_size=(1, 2), stride=(1, 2))
        return torch.complex(real, imag)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        left = torch.complex(batch["spec_real_L"], batch["spec_imag_L"])
        right = torch.complex(batch["spec_real_R"], batch["spec_imag_R"])
        x = torch.stack([left, right], dim=1)  # [B, 2, T, F]

        # Remove common frame-level gain while retaining interaural relations.
        joint_rms = torch.sqrt(
            x.abs().square().mean(dim=(1, 3), keepdim=True).clamp_min(self.eps)
        )
        x = x / joint_rms

        for block in self.blocks:
            x = self._frequency_pool(block(x))

        target_size = (x.shape[-2], self.pooled_freq_bins)
        real = F.adaptive_avg_pool2d(x.real, target_size)
        imag = F.adaptive_avg_pool2d(x.imag, target_size)
        magnitude = torch.sqrt(real.square() + imag.square() + self.eps)
        features = torch.cat([real, imag, magnitude], dim=1)
        features = features.permute(0, 2, 1, 3).contiguous()
        features = features.flatten(start_dim=2)
        return self.projection(features)


class SharedComplexEarEncoder(nn.Module):
    """Encode both ears with exactly shared complex T-F blocks."""

    def __init__(
        self,
        channels: Sequence[int] = (8, 12, 16),
        eps: float = 1.0e-6,
    ):
        super().__init__()
        if len(channels) != 3:
            raise ValueError("SharedComplexEarEncoder expects exactly three channel stages")
        self.eps = float(eps)
        blocks = []
        in_channels = 1
        for out_channels, dilation in zip(channels, (1, 2, 4)):
            blocks.append(
                ComplexDepthwiseSeparableTFBlock(
                    in_channels=in_channels,
                    out_channels=int(out_channels),
                    time_dilation=dilation,
                )
            )
            in_channels = int(out_channels)
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = in_channels

    @staticmethod
    def _frequency_pool(x: torch.Tensor) -> torch.Tensor:
        real = F.avg_pool2d(x.real, kernel_size=(1, 2), stride=(1, 2))
        imag = F.avg_pool2d(x.imag, kernel_size=(1, 2), stride=(1, 2))
        return torch.complex(real, imag)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if left.shape != right.shape:
            raise ValueError("Left and right complex spectra must have identical shapes")
        if left.ndim != 3:
            raise ValueError("Complex spectra must have shape [B, T, F]")

        ears = torch.stack([left, right], dim=1)
        joint_frame_rms = torch.sqrt(
            ears.abs().square().mean(dim=(1, 3), keepdim=True).clamp_min(self.eps)
        )
        ears = ears / joint_frame_rms

        batch_size = ears.shape[0]
        x = torch.cat([ears[:, 0:1], ears[:, 1:2]], dim=0)
        for block in self.blocks:
            x = self._frequency_pool(block(x))
        return x[:batch_size], x[batch_size:]


class LatentContentEncoder(nn.Module):
    """Compress common latent magnitude while retaining coarse spectral order."""

    def __init__(
        self,
        in_channels: int,
        out_dim: int = 24,
        hidden_channels: int = 8,
        pooled_freq_bins: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.pooled_freq_bins = int(pooled_freq_bins)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels * self.pooled_freq_bins, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, common_log_magnitude: torch.Tensor) -> torch.Tensor:
        x = self.stem(common_log_magnitude)
        x = F.adaptive_avg_pool2d(x, (x.shape[-2], self.pooled_freq_bins))
        x = x.permute(0, 2, 1, 3).contiguous().flatten(start_dim=2)
        return self.projection(x)


class LatentCrossSpectrumEncoder(nn.Module):
    """Locally encode latent magnitude difference and normalized cross-spectrum."""

    def __init__(
        self,
        latent_channels: int,
        out_dim: int = 24,
        hidden_channels: int = 16,
        output_channels: int = 8,
        pooled_freq_bins: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.pooled_freq_bins = int(pooled_freq_bins)
        input_channels = 3 * latent_channels
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.local_block = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.output = nn.Sequential(
            nn.Conv2d(hidden_channels, output_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, output_channels),
            nn.SiLU(inplace=True),
        )
        self.projection = nn.Sequential(
            nn.Linear(output_channels * self.pooled_freq_bins, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, relation: torch.Tensor) -> torch.Tensor:
        x = self.stem(relation)
        x = x + self.local_block(x)
        x = self.output(x)
        x = F.adaptive_avg_pool2d(x, (x.shape[-2], self.pooled_freq_bins))
        x = x.permute(0, 2, 1, 3).contiguous().flatten(start_dim=2)
        return self.projection(x)


class NativeLiteLatentCrossSpectrumDOANet(nn.Module):
    """Shared complex ear encoder with compact latent content/spatial branches."""

    def __init__(
        self,
        complex_channels: Sequence[int] = (8, 12, 16),
        content_out_dim: int = 24,
        spatial_out_dim: int = 24,
        content_hidden_channels: int = 8,
        content_pooled_freq_bins: int = 8,
        spatial_hidden_channels: int = 16,
        spatial_output_channels: int = 8,
        spatial_pooled_freq_bins: int = 32,
        gru_hidden_size: int = 80,
        gru_num_layers: int = 1,
        gru_dropout: float = 0.1,
        num_classes: int = 25,
        azimuth_range=(-180.0, 180.0),
        class_angles_deg: Sequence[float] | None = None,
        dropout: float = 0.2,
        use_attention_pooling: bool = True,
        attention_pooling_variant: str = "default",
        use_front_back_auxiliary: bool = False,
        use_regression: bool = False,
        use_pure_regression: bool = False,
        eps: float = 1.0e-6,
    ):
        super().__init__()
        self.eps = float(eps)
        self.shared_complex_encoder = SharedComplexEarEncoder(
            channels=complex_channels,
            eps=eps,
        )
        latent_channels = self.shared_complex_encoder.out_channels
        self.content_encoder = LatentContentEncoder(
            in_channels=latent_channels,
            out_dim=content_out_dim,
            hidden_channels=content_hidden_channels,
            pooled_freq_bins=content_pooled_freq_bins,
            dropout=dropout,
        )
        self.spatial_encoder = LatentCrossSpectrumEncoder(
            latent_channels=latent_channels,
            out_dim=spatial_out_dim,
            hidden_channels=spatial_hidden_channels,
            output_channels=spatial_output_channels,
            pooled_freq_bins=spatial_pooled_freq_bins,
            dropout=dropout,
        )
        self.content_branch_norm = nn.LayerNorm(content_out_dim)
        self.spatial_branch_norm = nn.LayerNorm(spatial_out_dim)
        temporal_input_dim = content_out_dim + spatial_out_dim
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
            attention_pooling_variant=attention_pooling_variant,
            use_front_back_auxiliary=use_front_back_auxiliary,
            azimuth_range=tuple(azimuth_range),
            class_angles_deg=class_angles_deg,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        left = torch.complex(batch["spec_real_L"], batch["spec_imag_L"])
        right = torch.complex(batch["spec_real_R"], batch["spec_imag_R"])
        latent_left, latent_right = self.shared_complex_encoder(left, right)

        magnitude_left = latent_left.abs().clamp_min(self.eps)
        magnitude_right = latent_right.abs().clamp_min(self.eps)
        log_magnitude_left = torch.log(magnitude_left)
        log_magnitude_right = torch.log(magnitude_right)
        common = 0.5 * (log_magnitude_left + log_magnitude_right)
        magnitude_difference = log_magnitude_left - log_magnitude_right
        unit_cross = latent_left * latent_right.conj()
        unit_cross = unit_cross / (magnitude_left * magnitude_right).clamp_min(self.eps)

        spatial_relation = torch.cat(
            [magnitude_difference, unit_cross.real, unit_cross.imag],
            dim=1,
        )
        content_feat = self.content_branch_norm(self.content_encoder(common))
        spatial_feat = self.spatial_branch_norm(self.spatial_encoder(spatial_relation))
        fused = self.fusion_norm(torch.cat([content_feat, spatial_feat], dim=-1))
        fused = self.fusion_dropout(fused)

        outputs = self.temporal_head(fused)
        outputs["content_feat"] = content_feat
        outputs["cue_feat"] = spatial_feat
        outputs["spatial_feat"] = spatial_feat
        outputs["fused_feat"] = fused
        return outputs


class HighFrequencyFrontBackHead(nn.Module):
    """Small front/back head that preserves ordered high-frequency spectral shape."""

    def __init__(
        self,
        freq_bins: int = 257,
        start_ratio: float = 0.5,
        pooled_freq_bins: int = 16,
        hidden_channels: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        if not 0.0 <= start_ratio < 1.0:
            raise ValueError("start_ratio must be in [0, 1)")
        if pooled_freq_bins < 1 or hidden_channels < 2:
            raise ValueError("pooled_freq_bins must be positive and hidden_channels >= 2")

        self.start_bin = int(round((freq_bins - 1) * start_ratio))
        mid_channels = max(hidden_channels // 2, 4)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, mid_channels, kernel_size=(3, 7), padding=(1, 3), bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                hidden_channels,
                kernel_size=(3, 5),
                stride=(1, 2),
                padding=(1, 2),
                bias=False,
            ),
            nn.GroupNorm(1, hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, pooled_freq_bins))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * pooled_freq_bins, hidden_channels * 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, 2),
        )

    def forward(
        self,
        log_mag_l: torch.Tensor,
        log_mag_r: torch.Tensor,
    ) -> torch.Tensor:
        high_l = log_mag_l[..., self.start_bin:]
        high_r = log_mag_r[..., self.start_bin:]
        if high_l.shape[-1] < 2:
            raise ValueError(
                f"High-frequency slice has only {high_l.shape[-1]} bins; "
                f"input frequency bins={log_mag_l.shape[-1]}, start_bin={self.start_bin}"
            )

        mean_spectrum = 0.5 * (high_l + high_r)
        difference_spectrum = high_l - high_r
        mean_shape = mean_spectrum - mean_spectrum.mean(dim=-1, keepdim=True)
        difference_shape = difference_spectrum - difference_spectrum.mean(
            dim=-1, keepdim=True
        )
        spectral_shape = torch.stack([mean_shape, difference_shape], dim=1)
        return self.classifier(self.pool(self.encoder(spectral_shape)))


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
        content_ear_token_dim: int = 24,
        content_ear_token_heads: int = 4,
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
        dual_cue_use_precompression_reliability_pooling: bool = False,
        dual_cue_precompression_pool_hidden_channels: int = 8,
        dual_cue_precompression_pool_residual_scale: float = 1.0,
        cue_specific_local_hidden_channels: int = 8,
        cue_specific_local_ild_scale_db: float = 20.0,
        cue_specific_local_ild_clip_db: float = 40.0,
        cue_specific_local_use_band_projection: bool = False,
        cue_specific_local_band_split_hz: float = 1500.0,
        cue_specific_local_band_projection_residual_scale: float = 1.0,
        cue_specific_local_use_joint_correction: bool = False,
        cue_specific_local_joint_correction_residual_scale: float = 1.0,
        cue_specific_local_use_coherence_context: bool = True,
        cue_specific_local_use_cue_consistency_context: bool = False,
        cue_specific_local_use_standalone_coherence: bool = True,
        cue_specific_local_use_fine_to_coarse_refinement: bool = False,
        cue_specific_local_fine_to_coarse_residual_scale: float = 1.0,
        cue_specific_local_block_type: str = "standard",
        cue_specific_local_ild_spectral_kernel_size: int = 7,
        cue_specific_local_ipd_spectral_kernel_size: int = 3,
        cue_specific_local_temporal_stabilizer_type: str = "none",
        cue_specific_local_temporal_stabilizer_hidden_channels: int = 8,
        cue_specific_local_temporal_stabilizer_kernel_size: int = 5,
        cue_progressive_aggregation: str = "mean",
        cue_progressive_channels: Sequence[int] = (8, 12, 16),
        cue_progressive_temporal_dilations: Sequence[int] = (1, 2, 4),
        cue_progressive_ild_kernel_size: int = 7,
        cue_progressive_ipd_kernel_size: int = 3,
        cue_progressive_out_dim: int = 32,
        cue_progressive_coherence_beta_init: float = 0.5,
        cue_stat_mode: str = "postcue_uniform",
        cue_rw_cpsd_time_frames: int = 5,
        cue_rw_cpsd_logit_clip: float = 6.0,
        cue_rw_cpsd_coefficient_mode: str = "global",
        cue_rw_cpsd_frequency_anchors: int = 8,
        cue_target_bias_mode: str = "shared_unit",
        cue_target_bias_max_strength: float = 2.0,
        cue_oracle_ild_scale_db: float = 6.0,
        cue_oracle_ipd_scale_deg: float = 45.0,
        cue_delay_max_ms: float = 1.0,
        cue_delay_bins: int = 33,
        cue_delay_temperature: float = 20.0,
        cue_input_mode: str = "explicit",
        raw_complex_cue_out_dim: int = 112,
        raw_complex_channels: Sequence[int] = (8, 12, 16),
        raw_complex_pooled_bins: int = 4,
        disable_content_stream: bool = False,
        use_branchwise_fusion_norm: bool = False,
        use_cross_ear_interaction: bool = False,
        gru_hidden_size: int = 80,
        gru_num_layers: int = 1,
        temporal_encoder_type: str = "gru",
        temporal_aggregation_type: str = "attention",
        mamba_num_layers: int = 2,
        mamba_state_dim: int = 16,
        mamba_expand_factor: int = 2,
        mamba_conv_kernel: int = 4,
        gru_dropout: float = 0.1,
        num_classes: int = 72,
        azimuth_range=(-180.0, 180.0),
        class_angles_deg: Sequence[float] | None = None,
        dropout: float = 0.2,
        use_attention_pooling: bool = True,
        attention_pooling_variant: str = "default",
        use_front_back_auxiliary: bool = True,
        front_back_head_mode: str = "temporal",
        spectral_fb_start_ratio: float = 0.5,
        spectral_fb_pooled_bins: int = 16,
        spectral_fb_hidden_channels: int = 16,
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
        if content_relation_mode not in {
            "mean_only",
            "mean_diff_absdiff",
            "mean_diff",
            "diff_only",
            "raw_concat",
            "learned_cross_attn",
            "pre_common_energy",
            "ear_token_attention",
        }:
            raise ValueError(f"Unsupported content_relation_mode: {content_relation_mode}")
        if content_relation_mode == "pre_common_energy" and content_input_mode != "logmag":
            raise ValueError("pre_common_energy requires content_input_mode='logmag'")
        if content_relation_mode == "pre_common_energy" and use_cross_ear_interaction:
            raise ValueError("pre_common_energy is incompatible with cross-ear interaction")
        if content_relation_mode == "ear_token_attention":
            if content_ear_token_dim < 1 or content_ear_token_heads < 1:
                raise ValueError("ear-token dimensions and heads must be positive")
            if content_ear_token_dim % content_ear_token_heads != 0:
                raise ValueError("content_ear_token_dim must be divisible by its head count")
        if cue_stat_mode not in {
            "postcue_uniform",
            "precue_stat",
            "phaseaware_stat",
            "rw_cpsd",
            "cue_factorized_cpsd",
            "cue_factorized_cpsd_oracle_supervised",
            "cue_factorized_cpsd_nonlinear",
            "cue_factorized_cpsd_nonlinear_oracle_supervised",
            "cue_factorized_cpsd_precision",
            "cue_factorized_cpsd_precision_calibrated",
            "target_rw_cpsd",
            "target_cue_factorized_cpsd",
            "oracle_target_cpsd",
            "oracle_target_masked_cpsd",
        }:
            raise ValueError(f"Unsupported cue_stat_mode: {cue_stat_mode}")
        if cue_input_mode not in {"explicit", "raw_complex"}:
            raise ValueError(f"Unsupported cue_input_mode: {cue_input_mode}")
        if cue_input_mode == "raw_complex" and cue_stat_mode != "postcue_uniform":
            raise ValueError("raw_complex cue input does not support handcrafted cue statistics")
        if dual_cue_use_precompression_reliability_pooling and cue_stat_mode != "postcue_uniform":
            raise ValueError(
                "pre-compression reliability pooling requires cue_stat_mode='postcue_uniform'"
            )
        if cue_stat_mode in {"precue_stat", "phaseaware_stat"} and cue_band_mode != "uniform":
            raise ValueError("pre-cue statistics currently require cue_band_mode='uniform'")
        if cue_stat_mode in {"precue_stat", "phaseaware_stat"} and cue_branch_mode != "dual":
            raise ValueError("pre-cue statistics currently require cue_branch_mode='dual'")
        if front_back_head_mode not in {"temporal", "spectral", "fused"}:
            raise ValueError(f"Unsupported front_back_head_mode: {front_back_head_mode}")
        if front_back_head_mode != "temporal" and not use_front_back_auxiliary:
            raise ValueError(
                "front_back_head_mode requires use_front_back_auxiliary=True"
            )

        self.content_input_mode = content_input_mode
        self.content_relation_mode = content_relation_mode
        self.use_cross_ear_interaction = use_cross_ear_interaction
        self.dual_cue_fusion_mode = dual_cue_fusion_mode
        self.content_encoder_type = content_encoder_type
        self.disable_reliability_branch = disable_reliability_branch
        self.disable_content_stream = disable_content_stream
        self.use_branchwise_fusion_norm = bool(use_branchwise_fusion_norm)
        self.cue_branch_mode = cue_branch_mode
        self.cue_band_mode = cue_band_mode
        self.cue_stat_mode = cue_stat_mode
        self.cue_input_mode = cue_input_mode
        self.front_back_head_mode = front_back_head_mode

        if front_back_head_mode in {"spectral", "fused"}:
            self.spectral_front_back_head = HighFrequencyFrontBackHead(
                freq_bins=freq_bins,
                start_ratio=spectral_fb_start_ratio,
                pooled_freq_bins=spectral_fb_pooled_bins,
                hidden_channels=spectral_fb_hidden_channels,
                dropout=dropout,
            )
        else:
            self.spectral_front_back_head = None
        if front_back_head_mode == "fused":
            self.front_back_fusion = nn.Linear(4, 2)
        else:
            self.front_back_fusion = None

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

        self.cue_statistics = None
        self.raw_complex_cue_encoder = None
        if cue_input_mode == "explicit":
            if cue_stat_mode in {"precue_stat", "phaseaware_stat"}:
                self.cue_statistics = BinauralCueStatistics(
                    mode=cue_stat_mode,
                    num_bands=lite_cue_bands,
                    freq_bins=freq_bins,
                    sample_rate=cue_sample_rate,
                    delay_max_ms=cue_delay_max_ms,
                    delay_bins=cue_delay_bins,
                    delay_temperature=cue_delay_temperature,
                )
            elif cue_stat_mode == "rw_cpsd":
                self.cue_statistics = ReliabilityWeightedCPSD(
                    time_frames=cue_rw_cpsd_time_frames,
                    score_logit_clip=cue_rw_cpsd_logit_clip,
                    coefficient_mode=cue_rw_cpsd_coefficient_mode,
                    frequency_anchors=cue_rw_cpsd_frequency_anchors,
                )
            elif cue_stat_mode == "cue_factorized_cpsd":
                self.cue_statistics = CueFactorizedCPSD(
                    time_frames=cue_rw_cpsd_time_frames,
                    score_logit_clip=cue_rw_cpsd_logit_clip,
                )
            elif cue_stat_mode == "cue_factorized_cpsd_oracle_supervised":
                self.cue_statistics = OracleSupervisedCueFactorizedCPSD(
                    time_frames=cue_rw_cpsd_time_frames,
                    score_logit_clip=cue_rw_cpsd_logit_clip,
                    oracle_ild_scale_db=cue_oracle_ild_scale_db,
                    oracle_ipd_scale_deg=cue_oracle_ipd_scale_deg,
                )
            elif cue_stat_mode == "cue_factorized_cpsd_nonlinear":
                self.cue_statistics = NonlinearCueFactorizedCPSD(
                    time_frames=cue_rw_cpsd_time_frames,
                    score_logit_clip=cue_rw_cpsd_logit_clip,
                )
            elif cue_stat_mode == "cue_factorized_cpsd_nonlinear_oracle_supervised":
                self.cue_statistics = NonlinearOracleSupervisedCueFactorizedCPSD(
                    time_frames=cue_rw_cpsd_time_frames,
                    score_logit_clip=cue_rw_cpsd_logit_clip,
                    oracle_ild_scale_db=cue_oracle_ild_scale_db,
                    oracle_ipd_scale_deg=cue_oracle_ipd_scale_deg,
                )
            elif cue_stat_mode == "cue_factorized_cpsd_precision":
                self.cue_statistics = PrecisionWeightedCueFactorizedCPSD(
                    time_frames=cue_rw_cpsd_time_frames,
                    score_logit_clip=cue_rw_cpsd_logit_clip,
                )
            elif cue_stat_mode == "cue_factorized_cpsd_precision_calibrated":
                self.cue_statistics = CalibratedPrecisionCueFactorizedCPSD(
                    time_frames=cue_rw_cpsd_time_frames,
                    score_logit_clip=cue_rw_cpsd_logit_clip,
                    ild_normalizer_db=cue_specific_local_ild_scale_db,
                )
            elif cue_stat_mode == "target_cue_factorized_cpsd":
                self.cue_statistics = TargetAwareCueFactorizedCPSD(
                    time_frames=cue_rw_cpsd_time_frames,
                    score_logit_clip=cue_rw_cpsd_logit_clip,
                    target_bias_mode=cue_target_bias_mode,
                    target_bias_max_strength=cue_target_bias_max_strength,
                )
            elif cue_stat_mode == "target_rw_cpsd":
                self.cue_statistics = TargetAwareRWCPSD(
                    time_frames=cue_rw_cpsd_time_frames,
                    score_logit_clip=cue_rw_cpsd_logit_clip,
                    coefficient_mode=cue_rw_cpsd_coefficient_mode,
                    frequency_anchors=cue_rw_cpsd_frequency_anchors,
                )
            elif cue_stat_mode in {
                "oracle_target_cpsd",
                "oracle_target_masked_cpsd",
            }:
                oracle_cls = (
                    OracleTargetCPSD
                    if cue_stat_mode == "oracle_target_cpsd"
                    else OracleTargetMaskedCPSD
                )
                self.cue_statistics = oracle_cls(
                    time_frames=cue_rw_cpsd_time_frames,
                    score_logit_clip=cue_rw_cpsd_logit_clip,
                    coefficient_mode=cue_rw_cpsd_coefficient_mode,
                    frequency_anchors=cue_rw_cpsd_frequency_anchors,
                )

            cue_encoder_freq_bins = (
                lite_cue_bands
                if cue_stat_mode in {"precue_stat", "phaseaware_stat"}
                else freq_bins
            )
            self.cue_encoder = DualBranchCueEncoder(
                cue_bands=lite_cue_bands,
                cue_freq_bins=cue_encoder_freq_bins,
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
                use_precompression_reliability_pooling=dual_cue_use_precompression_reliability_pooling,
                precompression_pool_hidden_channels=dual_cue_precompression_pool_hidden_channels,
                precompression_pool_residual_scale=dual_cue_precompression_pool_residual_scale,
                cue_specific_local_hidden_channels=cue_specific_local_hidden_channels,
                cue_specific_local_ild_scale_db=cue_specific_local_ild_scale_db,
                cue_specific_local_ild_clip_db=cue_specific_local_ild_clip_db,
                cue_specific_local_use_band_projection=cue_specific_local_use_band_projection,
                cue_specific_local_band_split_hz=cue_specific_local_band_split_hz,
                cue_specific_local_band_projection_residual_scale=cue_specific_local_band_projection_residual_scale,
                cue_specific_local_use_joint_correction=cue_specific_local_use_joint_correction,
                cue_specific_local_joint_correction_residual_scale=cue_specific_local_joint_correction_residual_scale,
                cue_specific_local_use_coherence_context=cue_specific_local_use_coherence_context,
                cue_specific_local_use_cue_consistency_context=(
                    cue_specific_local_use_cue_consistency_context
                ),
                cue_specific_local_use_standalone_coherence=cue_specific_local_use_standalone_coherence,
                cue_specific_local_use_fine_to_coarse_refinement=cue_specific_local_use_fine_to_coarse_refinement,
                cue_specific_local_fine_to_coarse_residual_scale=cue_specific_local_fine_to_coarse_residual_scale,
                cue_specific_local_block_type=cue_specific_local_block_type,
                cue_specific_local_ild_spectral_kernel_size=cue_specific_local_ild_spectral_kernel_size,
                cue_specific_local_ipd_spectral_kernel_size=cue_specific_local_ipd_spectral_kernel_size,
                cue_specific_local_temporal_stabilizer_type=(
                    cue_specific_local_temporal_stabilizer_type
                ),
                cue_specific_local_temporal_stabilizer_hidden_channels=(
                    cue_specific_local_temporal_stabilizer_hidden_channels
                ),
                cue_specific_local_temporal_stabilizer_kernel_size=(
                    cue_specific_local_temporal_stabilizer_kernel_size
                ),
                cue_progressive_aggregation=cue_progressive_aggregation,
                cue_progressive_channels=cue_progressive_channels,
                cue_progressive_temporal_dilations=cue_progressive_temporal_dilations,
                cue_progressive_ild_kernel_size=cue_progressive_ild_kernel_size,
                cue_progressive_ipd_kernel_size=cue_progressive_ipd_kernel_size,
                cue_progressive_out_dim=cue_progressive_out_dim,
                cue_progressive_coherence_beta_init=cue_progressive_coherence_beta_init,
            )
        else:
            self.cue_encoder = None
            self.raw_complex_cue_encoder = RawComplexTFCueEncoder(
                out_dim=raw_complex_cue_out_dim,
                channels=raw_complex_channels,
                pooled_freq_bins=raw_complex_pooled_bins,
                dropout=dropout,
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
        elif cue_branch_mode == "cue_specific_progressive_tf":
            cue_encoder_out_dim = cue_progressive_out_dim
        elif cue_branch_mode in {"cue_specific_resolution", "cue_specific_local_tf"}:
            cue_encoder_out_dim = cue_ild_out_dim + cue_ipd_out_dim
            if (
                cue_branch_mode != "cue_specific_local_tf"
                and not disable_reliability_branch
            ) or (
                cue_branch_mode == "cue_specific_local_tf"
                and cue_specific_local_use_standalone_coherence
                and not disable_reliability_branch
            ):
                cue_encoder_out_dim += cue_reliability_out_dim
        elif disable_reliability_branch or dual_cue_fusion_mode == "gate":
            cue_encoder_out_dim = cue_value_out_dim
        elif dual_cue_fusion_mode == "residual_product_concat":
            cue_encoder_out_dim = 2 * cue_value_out_dim + cue_reliability_out_dim
        else:
            cue_encoder_out_dim = cue_value_out_dim + cue_reliability_out_dim
        if cue_input_mode == "raw_complex":
            cue_encoder_out_dim = raw_complex_cue_out_dim

        self.content_pair_attn = None
        self.content_pair_norm = None
        self.content_ear_projection = None
        self.content_ear_embedding = None
        self.content_pair_ffn = None
        self.content_pair_ffn_norm = None
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
            elif content_relation_mode == "ear_token_attention":
                self.content_ear_projection = nn.Linear(
                    encoder_out_dim,
                    content_ear_token_dim,
                )
                self.content_ear_embedding = nn.Parameter(
                    torch.empty(2, content_ear_token_dim)
                )
                nn.init.normal_(self.content_ear_embedding, std=0.02)
                self.content_pair_attn = nn.MultiheadAttention(
                    embed_dim=content_ear_token_dim,
                    num_heads=content_ear_token_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                self.content_pair_norm = nn.LayerNorm(content_ear_token_dim)
                self.content_pair_ffn = nn.Sequential(
                    nn.Linear(content_ear_token_dim, 2 * content_ear_token_dim),
                    nn.SiLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(2 * content_ear_token_dim, content_ear_token_dim),
                )
                self.content_pair_ffn_norm = nn.LayerNorm(content_ear_token_dim)
                content_relation_dim = 2 * content_ear_token_dim
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
        if self.use_branchwise_fusion_norm:
            if disable_content_stream:
                raise ValueError("branchwise fusion normalization requires the content stream")
            self.content_branch_norm = nn.LayerNorm(content_out_dim)
            self.cue_branch_norm = nn.LayerNorm(cue_encoder_out_dim)
        else:
            self.content_branch_norm = None
            self.cue_branch_norm = None
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
                use_front_back_auxiliary=(
                    use_front_back_auxiliary and front_back_head_mode in {"temporal", "fused"}
                ),
            )
        elif temporal_head_type == "default":
            self.temporal_head = TemporalHead(
                input_dim=temporal_input_dim,
                gru_hidden_size=gru_hidden_size,
                gru_num_layers=gru_num_layers,
                temporal_encoder_type=temporal_encoder_type,
                temporal_aggregation_type=temporal_aggregation_type,
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
                use_front_back_auxiliary=(
                    use_front_back_auxiliary and front_back_head_mode in {"temporal", "fused"}
                ),
                azimuth_range=tuple(azimuth_range),
                class_angles_deg=class_angles_deg,
            )
        else:
            raise ValueError(f"Unsupported temporal_head_type: {temporal_head_type}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        log_mag_L = batch["log_mag_L"]
        log_mag_R = batch["log_mag_R"]
        ild = batch.get("ild")
        ipd = batch.get("ipd")

        if self.content_input_mode == "logmag":
            left_content = log_mag_L.unsqueeze(1)
            right_content = log_mag_R.unsqueeze(1)
        else:
            left_content = torch.stack([batch["spec_real_L"], batch["spec_imag_L"]], dim=1)
            right_content = torch.stack([batch["spec_real_R"], batch["spec_imag_R"]], dim=1)

        if self.disable_content_stream:
            if self.cue_input_mode == "raw_complex":
                t_enc = batch["spec_real_L"].shape[1]
            else:
                t_enc = ild.shape[1]
            content_feat = None
        elif self.content_relation_mode == "pre_common_energy":
            common_log_energy = 0.5 * torch.logaddexp(
                2.0 * log_mag_L,
                2.0 * log_mag_R,
            ) - 0.5 * math.log(2.0)
            common_feat = self.encoder(common_log_energy.unsqueeze(1))
            t_enc = common_feat.shape[1]
            content_feat = self.content_fusion(common_feat)
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
            elif self.content_relation_mode == "ear_token_attention":
                bsz, t_steps, _ = f_l.shape
                left_token = self.content_ear_projection(f_l)
                right_token = self.content_ear_projection(f_r)
                ear_tokens = torch.stack([left_token, right_token], dim=2)
                ear_tokens = ear_tokens + self.content_ear_embedding.view(
                    1, 1, 2, -1
                )
                token_dim = ear_tokens.shape[-1]
                ear_tokens = ear_tokens.reshape(bsz * t_steps, 2, token_dim)
                attended_tokens, _ = self.content_pair_attn(
                    ear_tokens,
                    ear_tokens,
                    ear_tokens,
                    need_weights=False,
                )
                ear_tokens = self.content_pair_norm(ear_tokens + attended_tokens)
                refined_tokens = self.content_pair_ffn(ear_tokens)
                ear_tokens = self.content_pair_ffn_norm(
                    ear_tokens + refined_tokens
                )
                content_feat = ear_tokens.reshape(
                    bsz,
                    t_steps,
                    2 * token_dim,
                )
            elif self.content_relation_mode == "mean_diff_absdiff":
                abs_diff_feat = diff_feat.abs()
                content_feat = torch.cat([mean_feat, diff_feat, abs_diff_feat], dim=-1)
            elif self.content_relation_mode == "mean_only":
                content_feat = mean_feat
            elif self.content_relation_mode == "mean_diff":
                content_feat = torch.cat([mean_feat, diff_feat], dim=-1)
            elif self.content_relation_mode == "raw_concat":
                content_feat = torch.cat([f_l, f_r], dim=-1)
            else:
                content_feat = diff_feat
            content_feat = self.content_fusion(content_feat)

        if self.cue_input_mode == "raw_complex":
            cue_feat = self.raw_complex_cue_encoder(batch)[:, :t_enc]
            cue_outputs = {
                "cue_feat": cue_feat,
                "cue_value_feat": cue_feat,
                "cue_reliability_feat": None,
                "cue_tf_mask": None,
                "cue_gate": None,
            }
        elif self.cue_statistics is None:
            if ild is None or ipd is None:
                raise KeyError("explicit cue input requires ild and ipd tensors")
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
            magnitude_context = (0.5 * (log_mag_L[:, :t_enc, :] + log_mag_R[:, :t_enc, :])).unsqueeze(1)
            cue_outputs = self.cue_encoder(
                value_tensor,
                reliability_tensor,
                magnitude_context=magnitude_context,
            )
            cue_feat = cue_outputs["cue_feat"]
        else:
            cue_statistics = self.cue_statistics(batch, t_enc)
            value_tensor = cue_statistics["value_tensor"]
            reliability_tensor = cue_statistics["reliability_tensor"]
            cue_outputs = self.cue_encoder(
                value_tensor,
                reliability_tensor,
                ild_consistency_tensor=cue_statistics.get(
                    "ild_consistency_tensor"
                ),
                ipd_consistency_tensor=cue_statistics.get(
                    "ipd_consistency_tensor"
                ),
            )
            cue_feat = cue_outputs["cue_feat"]

        if self.use_branchwise_fusion_norm:
            content_feat = self.content_branch_norm(content_feat)
            cue_feat = self.cue_branch_norm(cue_feat)
        fused = cue_feat if self.disable_content_stream else torch.cat([content_feat, cue_feat], dim=-1)
        fused = self.fusion_norm(fused)
        fused = self.fusion_dropout(fused)

        outputs = self.temporal_head(fused)
        if self.spectral_front_back_head is not None:
            spectral_fb_logits = self.spectral_front_back_head(log_mag_L, log_mag_R)
            outputs["spectral_front_back_logits"] = spectral_fb_logits
            if self.front_back_head_mode == "spectral":
                outputs["front_back_logits"] = spectral_fb_logits
            else:
                temporal_fb_logits = outputs["front_back_logits"]
                outputs["temporal_front_back_logits"] = temporal_fb_logits
                outputs["front_back_logits"] = self.front_back_fusion(
                    torch.cat([temporal_fb_logits, spectral_fb_logits], dim=-1)
                )
        outputs["cue_feat"] = cue_feat
        outputs["fused_feat"] = fused
        outputs["content_feat"] = content_feat
        outputs["cue_value_feat"] = cue_outputs["cue_value_feat"]
        outputs["cue_reliability_feat"] = cue_outputs["cue_reliability_feat"]
        outputs["cue_tf_mask"] = cue_outputs["cue_tf_mask"]
        outputs["cue_tf_weight"] = cue_outputs.get("cue_tf_weight")
        outputs["cue_pool_alpha"] = cue_outputs.get("cue_pool_alpha")
        outputs["target_probability"] = cue_statistics.get(
            "target_probability"
        ) if self.cue_statistics is not None else None
        outputs["target_mask_loss"] = cue_statistics.get(
            "target_mask_loss"
        ) if self.cue_statistics is not None else None
        outputs["target_covariance_loss"] = cue_statistics.get(
            "target_covariance_loss"
        ) if self.cue_statistics is not None else None
        outputs["cue_reliability_loss"] = cue_statistics.get(
            "cue_reliability_loss"
        ) if self.cue_statistics is not None else None
        outputs["cue_uncertainty_loss"] = cue_statistics.get(
            "cue_uncertainty_loss"
        ) if self.cue_statistics is not None else None
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
