"""BiEAR-inspired active binaural waveform baseline for 72-class DOA.

This adapts the BiEAR active gammatone/ERB binaural front-end to this
project's single-source static DOA classification protocol. It keeps the
auditory-inspired adaptive Q front-end and ILD/IPD GRU encoders, while replacing
BiEAR's sector-presence, within-sector AoA and distance heads with a 72-class
azimuth classifier and optional front/back auxiliary classifier.
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def erb_hz(f_hz: np.ndarray) -> np.ndarray:
    return 24.7 * (4.37 * f_hz / 1000.0 + 1.0)


def erb_rate(f_hz: np.ndarray) -> np.ndarray:
    return 21.4 * np.log10(4.37 * f_hz / 1000.0 + 1.0)


def inv_erb_rate(erb: np.ndarray) -> np.ndarray:
    return (10.0 ** (erb / 21.4) - 1.0) * 1000.0 / 4.37


def erb_spaced_fc_and_q(
    n_bands: int = 64,
    fmin: float = 50.0,
    fmax: float = 7200.0,
    erb_factor: float = 1.019,
) -> tuple[np.ndarray, np.ndarray]:
    emin, emax = erb_rate(fmin), erb_rate(fmax)
    erb = np.linspace(emin, emax, n_bands)
    fc = inv_erb_rate(erb)
    bw = erb_factor * erb_hz(fc)
    q0 = fc / bw
    return fc, q0


def make_delta_q_profile(
    fc_hz: torch.Tensor,
    delta_q_base: float = 2.0,
    low_factor: float = 0.5,
    high_factor: float = 1.0,
) -> torch.Tensor:
    fc_np = fc_hz.detach().cpu().numpy()
    erb = erb_rate(fc_np)
    erb = (erb - erb.min()) / (erb.max() - erb.min() + 1e-12)
    mult = low_factor + (high_factor - low_factor) * erb
    profile = torch.tensor(mult, dtype=torch.float32, device=fc_hz.device)
    return torch.clamp(delta_q_base * profile, min=1e-3)


class FramewiseAdaptiveGammatoneFB(nn.Module):
    """Monaural framewise ERB/gammatone-like front-end with adaptive Q."""

    def __init__(
        self,
        sample_rate: int = 16000,
        segment_seconds: float = 2.0,
        timesteps: int = 40,
        n_fft: int = 1024,
        n_bands: int = 64,
        fmin: float = 50.0,
        fmax: Optional[float] = None,
        hop_ratio: float = 1.0,
        fixed_q: bool = False,
        delta_q_base: float = 0.5,
        delta_q_low_factor: float = 0.5,
        delta_q_high_factor: float = 1.0,
        delta_q_mode: str = "absolute",
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.segment_samples = int(round(float(segment_seconds) * self.sample_rate))
        self.timesteps = int(timesteps)
        self.n_fft = int(n_fft)
        self.n_bands = int(n_bands)
        self.fixed_q = bool(fixed_q)
        self.delta_q_mode = delta_q_mode.lower()

        if self.timesteps <= 0:
            raise ValueError("timesteps must be positive")

        win = max(1, int(round(self.segment_samples / self.timesteps)))
        hop = max(1, int(round(win * float(hop_ratio))))
        self.win = win
        self.hop = hop
        self.register_buffer("win_fn", torch.hann_window(self.win), persistent=False)
        self.register_buffer("f_fft", torch.linspace(0.0, self.sample_rate / 2.0, self.n_fft // 2 + 1))

        if fmax is None:
            fmax = self.sample_rate / 2.0 * 0.9
        fc_np, q0_np = erb_spaced_fc_and_q(self.n_bands, fmin, fmax)
        self.register_buffer("fc", torch.tensor(fc_np, dtype=torch.float32))
        self.register_buffer("q0", torch.tensor(q0_np, dtype=torch.float32))
        self.register_buffer(
            "delta_q_vec",
            make_delta_q_profile(
                torch.tensor(fc_np, dtype=torch.float32),
                delta_q_base=delta_q_base,
                low_factor=delta_q_low_factor,
                high_factor=delta_q_high_factor,
            ),
        )

        self.q_min = 0.5
        self.q_max = 20.0
        if not self.fixed_q:
            self.q_rnn = nn.GRU(input_size=2 * self.n_bands, hidden_size=128, batch_first=True)
            self.q_out = nn.Sequential(
                nn.Linear(128, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(128, self.n_bands),
            )
            nn.init.zeros_(self.q_out[-1].weight)
            nn.init.zeros_(self.q_out[-1].bias)
        else:
            self.q_rnn = None
            self.q_out = None

    def _frame(self, wav: torch.Tensor) -> torch.Tensor:
        bsz, nsamp = wav.shape
        if nsamp < self.segment_samples:
            wav = F.pad(wav, (0, self.segment_samples - nsamp))
        else:
            wav = wav[:, : self.segment_samples]

        if wav.shape[1] < self.win:
            wav = F.pad(wav, (0, self.win - wav.shape[1]))

        frames = wav.unfold(dimension=1, size=self.win, step=self.hop)
        if frames.shape[1] >= self.timesteps:
            frames = frames[:, : self.timesteps, :]
        else:
            frames = F.pad(frames, (0, 0, 0, self.timesteps - frames.shape[1]))
        return frames.contiguous().view(bsz, self.timesteps, self.win)

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if wav.dim() != 2:
            raise ValueError(f"Expected waveform shape [B, samples], got {tuple(wav.shape)}")

        device = wav.device
        frames = self._frame(wav.float())
        bsz = frames.shape[0]
        q_prev = torch.clamp(self.q0, self.q_min, self.q_max).to(device).unsqueeze(0).expand(bsz, -1)
        q_hidden = None
        y_mem = torch.zeros(bsz, self.n_bands, device=device)

        f = self.f_fft.to(device).view(1, 1, -1)
        fc = self.fc.to(device).view(1, self.n_bands, 1)
        win_fn = self.win_fn.to(device)
        delta_q_vec = self.delta_q_vec.to(device).unsqueeze(0)

        y_all = []
        q_all = []
        x_all = []
        beta = 0.8
        for t in range(self.timesteps):
            frame = frames[:, t, :] * win_fn
            spec = torch.fft.rfft(frame, n=self.n_fft)
            x_all.append(spec)

            bw = (self.fc.to(device).unsqueeze(0) / (q_prev + 1e-8)).unsqueeze(-1) + 1e-8
            weight = torch.exp(-0.5 * ((f - fc) / bw) ** 2)
            weight = weight / (weight.sum(dim=-1, keepdim=True) + 1e-8)
            weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)

            y = torch.einsum("bf,bnf->bn", spec.abs(), weight)
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            y_all.append(y)
            q_all.append(q_prev)

            if self.fixed_q:
                q_prev = self.q0.to(device).unsqueeze(0).expand(bsz, -1)
                q_hidden = None
                y_mem = torch.zeros_like(y_mem)
                continue

            y_ctrl = torch.clamp(torch.log1p(torch.clamp(y, min=0.0)), min=0.0, max=8.0)
            q_in = torch.cat([y_ctrl, y_mem], dim=-1).unsqueeze(1)
            q_h, q_hidden = self.q_rnn(q_in, q_hidden)
            q_h = torch.nan_to_num(q_h, nan=0.0, posinf=10.0, neginf=-10.0)
            if q_hidden is not None:
                q_hidden = torch.nan_to_num(q_hidden, nan=0.0, posinf=10.0, neginf=-10.0)
            delta = torch.tanh(self.q_out(q_h.squeeze(1)))
            if self.delta_q_mode == "relative":
                q_prev = self.q0.to(device).unsqueeze(0) * (1.0 + delta_q_vec * delta)
            else:
                q_prev = self.q0.to(device).unsqueeze(0) + delta_q_vec * delta
            q_prev = torch.clamp(q_prev, self.q_min, self.q_max)
            if not torch.isfinite(q_prev).all():
                q_prev = self.q0.to(device).unsqueeze(0).expand(bsz, -1)
                q_hidden = None
            y_mem = beta * y_mem + (1.0 - beta) * y_ctrl.detach()

        return torch.stack(y_all, dim=1), torch.stack(q_all, dim=1), torch.stack(x_all, dim=1)


class BinauralAdaptiveGammatoneFB(nn.Module):
    """Dual-ear adaptive front-end, with optional shared/single Q controller."""

    def __init__(
        self,
        sample_rate: int = 16000,
        segment_seconds: float = 2.0,
        timesteps: int = 40,
        n_fft: int = 1024,
        n_bands: int = 64,
        fixed_q: bool = False,
        controller_mode: str = "single",
        **kwargs,
    ):
        super().__init__()
        self.controller_mode = controller_mode
        self.n_bands = int(n_bands)
        self.front_l = FramewiseAdaptiveGammatoneFB(
            sample_rate=sample_rate,
            segment_seconds=segment_seconds,
            timesteps=timesteps,
            n_fft=n_fft,
            n_bands=n_bands,
            fixed_q=fixed_q,
            **kwargs,
        )
        if controller_mode == "shared":
            self.front_r = self.front_l
        else:
            self.front_r = FramewiseAdaptiveGammatoneFB(
                sample_rate=sample_rate,
                segment_seconds=segment_seconds,
                timesteps=timesteps,
                n_fft=n_fft,
                n_bands=n_bands,
                fixed_q=fixed_q,
                **kwargs,
            )

        self.register_buffer("f_fft", self.front_l.f_fft.detach().clone(), persistent=False)
        self.register_buffer("fc", self.front_l.fc.detach().clone(), persistent=False)

    def forward(self, wav_l: torch.Tensor, wav_r: torch.Tensor):
        y_l, q_l, x_l = self.front_l(wav_l)
        y_r, q_r, x_r = self.front_r(wav_r)
        return y_l, y_r, q_l, q_r, x_l, x_r


class ILDEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 200, latent_dim: int = 100):
        super().__init__()
        self.in_norm = nn.LayerNorm(input_dim)
        self.gru1 = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim, latent_dim, batch_first=True)

    def forward(self, log_l: torch.Tensor, log_r: torch.Tensor) -> torch.Tensor:
        ild = torch.clamp(torch.nan_to_num(log_l - log_r), -10.0, 10.0)
        h, _ = self.gru1(self.in_norm(ild))
        h, _ = self.gru2(h)
        return torch.nan_to_num(h.mean(dim=1))


class IPDEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 200, latent_dim: int = 100):
        super().__init__()
        self.in_norm = nn.LayerNorm(input_dim)
        self.gru1 = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim, latent_dim, batch_first=True)

    def forward(self, phase_l: torch.Tensor, phase_r: torch.Tensor) -> torch.Tensor:
        delta = phase_l - phase_r
        ipd = torch.atan2(torch.sin(delta), torch.cos(delta))
        h, _ = self.gru1(self.in_norm(torch.nan_to_num(ipd)))
        h, _ = self.gru2(h)
        return torch.nan_to_num(h.mean(dim=1))


class BiEARDoaClassifier(nn.Module):
    """BiEAR-inspired waveform classifier for this project's static DOA task."""

    def __init__(
        self,
        sample_rate: int = 16000,
        segment_seconds: float = 2.0,
        timesteps: int = 40,
        n_fft: int = 1024,
        n_bands: int = 64,
        latent_dim: int = 100,
        encoder_hidden_dim: int = 200,
        num_classes: int = 72,
        dropout: float = 0.2,
        use_front_back_auxiliary: bool = False,
        use_cc: bool = False,
        fixed_frontend_q: bool = False,
        controller_mode: str = "independent",
        delta_q_base: float = 0.5,
        delta_q_low_factor: float = 0.5,
        delta_q_high_factor: float = 1.0,
        delta_q_mode: str = "absolute",
    ):
        super().__init__()
        self.use_cc = bool(use_cc)
        self.use_front_back_auxiliary = bool(use_front_back_auxiliary)
        self.n_bands = int(n_bands)

        self.bifb = BinauralAdaptiveGammatoneFB(
            sample_rate=sample_rate,
            segment_seconds=segment_seconds,
            timesteps=timesteps,
            n_fft=n_fft,
            n_bands=n_bands,
            fixed_q=fixed_frontend_q,
            controller_mode=controller_mode,
            delta_q_base=delta_q_base,
            delta_q_low_factor=delta_q_low_factor,
            delta_q_high_factor=delta_q_high_factor,
            delta_q_mode=delta_q_mode,
        )
        self.encoder_ild = ILDEncoder(n_bands, hidden_dim=encoder_hidden_dim, latent_dim=latent_dim)
        self.encoder_ipd = IPDEncoder(n_bands, hidden_dim=encoder_hidden_dim, latent_dim=latent_dim)
        if self.use_cc:
            self.cc_proj = nn.Linear(n_bands, latent_dim)

        feat_dim = 2 * latent_dim + (latent_dim if self.use_cc else 0)
        self.body = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 400),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(200, num_classes)
        if self.use_front_back_auxiliary:
            self.front_back_classifier = nn.Linear(200, 2)
        else:
            self.front_back_classifier = None
        self.last_Q = None

    def _subband_phase_from_stft(
        self,
        stft: torch.Tensor,
        q_all: torch.Tensor,
        eps_mag: float = 1e-3,
    ) -> torch.Tensor:
        bsz, timesteps, _ = stft.shape
        device = stft.device
        f = self.bifb.f_fft.to(device).view(1, 1, -1)
        fc = self.bifb.fc.to(device).view(1, self.n_bands, 1)

        phases = []
        for t in range(timesteps):
            q = q_all[:, t, :]
            bw = (self.bifb.fc.to(device).unsqueeze(0) / (q + 1e-8)).unsqueeze(-1) + 1e-8
            weight = torch.exp(-0.5 * ((f - fc) / bw) ** 2)
            weight = weight / (weight.sum(dim=-1, keepdim=True) + 1e-8)
            weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)
            z = torch.einsum("bnf,bf->bn", torch.complex(weight, torch.zeros_like(weight)), stft[:, t, :])
            real = torch.nan_to_num(z.real, nan=0.0, posinf=0.0, neginf=0.0)
            imag = torch.nan_to_num(z.imag, nan=0.0, posinf=0.0, neginf=0.0)
            mag = torch.sqrt(real.square() + imag.square() + eps_mag * eps_mag)
            phases.append(torch.atan2(imag / mag, real / mag))
        return torch.stack(phases, dim=1).view(bsz, timesteps, self.n_bands)

    def _cc_feature(self, log_l: torch.Tensor, log_r: torch.Tensor) -> torch.Tensor:
        centered_l = log_l - log_l.mean(dim=1, keepdim=True)
        centered_r = log_r - log_r.mean(dim=1, keepdim=True)
        num = (centered_l * centered_r).mean(dim=1)
        den = torch.sqrt(centered_l.square().mean(dim=1) * centered_r.square().mean(dim=1) + 1e-8)
        return torch.nan_to_num(num / den)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "waveform" not in batch:
            raise KeyError("BiEARDoaClassifier requires batch['waveform']; set dataset.include_waveform=true")

        wav = batch["waveform"].float()
        if wav.dim() != 3 or wav.shape[1] < 2:
            raise ValueError(f"Expected waveform shape [B, 2, samples], got {tuple(wav.shape)}")
        wav_l = torch.clamp(wav[:, 0, :], -1.0, 1.0)
        wav_r = torch.clamp(wav[:, 1, :], -1.0, 1.0)

        y_l, y_r, q_l, q_r, stft_l, stft_r = self.bifb(wav_l, wav_r)
        self.last_Q = 0.5 * (q_l + q_r)

        y_l = torch.nan_to_num(y_l, nan=0.0, posinf=1e6, neginf=0.0)
        y_r = torch.nan_to_num(y_r, nan=0.0, posinf=1e6, neginf=0.0)
        log_l = torch.clamp(torch.log1p(y_l), 0.0, 12.0)
        log_r = torch.clamp(torch.log1p(y_r), 0.0, 12.0)
        phase_l = self._subband_phase_from_stft(stft_l, q_l)
        phase_r = self._subband_phase_from_stft(stft_r, q_r)

        z_ild = self.encoder_ild(log_l, log_r)
        z_ipd = self.encoder_ipd(phase_l, phase_r)
        feats = [z_ild, z_ipd]
        if self.use_cc:
            feats.append(self.cc_proj(self._cc_feature(log_l, log_r)))

        body = self.body(torch.cat(feats, dim=-1))
        out = {"logits": self.classifier(body)}
        if self.front_back_classifier is not None:
            out["front_back_logits"] = self.front_back_classifier(body)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
