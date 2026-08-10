#!/usr/bin/env python3
"""Compare candidate nonstationary excitations for the two-channel ANF."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from anf_generator import estimate_coherence, generate_signals
from anf_generator.CoherenceMatrix import Parameters as ANFParameters
from scipy.signal import fftconvolve, firwin2, istft, stft, welch

from tools.generate_cipic_roomsim25_anf_nonstationary_v3 import (
    ANF_NFFT,
    _load_noise_cached,
    _noise_start,
    load_subject_spacings,
    make_bundles,
)


SAMPLE_RATE = 16000
LENGTH = 32000
CONTEXT = ANF_NFFT
SOURCE_LENGTH = LENGTH + 2 * CONTEXT
DEMAND_ROOT = Path("/disk2/bywang/data/demand")


def frame_rms_cv(signal: np.ndarray, frame: int = 800, hop: int = 400) -> float:
    values = [
        math.sqrt(float(np.mean(np.square(signal[start : start + frame]))) + 1e-12)
        for start in range(0, len(signal) - frame + 1, hop)
    ]
    return float(np.std(values) / (np.mean(values) + 1e-12))


def normalize(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal - np.mean(signal), dtype=np.float64)
    return signal / math.sqrt(float(np.mean(np.square(signal))) + 1e-12)


def psd_shaped_gaussian(source_a: np.ndarray, source_b: np.ndarray, seed: int) -> np.ndarray:
    reference = np.concatenate([source_a, source_b])
    frequencies, psd = welch(reference, fs=SAMPLE_RATE, nperseg=2048, noverlap=1024)
    gain = np.sqrt(np.maximum(psd, 1e-12))
    gain /= math.sqrt(float(np.mean(np.square(gain))) + 1e-12)
    shaping_filter = firwin2(513, frequencies / (SAMPLE_RATE / 2.0), gain)
    white = np.random.default_rng(seed).standard_normal((2, SOURCE_LENGTH + 512))
    return np.stack([fftconvolve(value, shaping_filter, mode="valid")[:SOURCE_LENGTH] for value in white])


def envelope_modulated_gaussian(
    source_a: np.ndarray,
    source_b: np.ndarray,
    seed: int,
    envelope_samples: int,
) -> np.ndarray:
    inputs = psd_shaped_gaussian(source_a, source_b, seed)
    power = 0.5 * (np.square(source_a) + np.square(source_b))
    window = np.ones(envelope_samples, dtype=np.float64) / envelope_samples
    envelope = np.sqrt(np.maximum(fftconvolve(power, window, mode="same"), 1e-8))
    envelope /= math.sqrt(float(np.mean(np.square(envelope))) + 1e-12)
    return np.stack([normalize(value * envelope) for value in inputs])


def shared_spectrogram_random_phase(source_a: np.ndarray, source_b: np.ndarray, seed: int) -> np.ndarray:
    kwargs = dict(
        fs=1.0,
        window="hann",
        nperseg=ANF_NFFT,
        noverlap=ANF_NFFT // 4,
        nfft=ANF_NFFT,
        return_onesided=True,
        boundary="zeros",
        padded=True,
        scaling="spectrum",
    )
    _, _, spec_a = stft(source_a, **kwargs)
    _, _, spec_b = stft(source_b, **kwargs)
    magnitude = np.sqrt(0.5 * (np.square(np.abs(spec_a)) + np.square(np.abs(spec_b))))
    rng = np.random.default_rng(seed)
    outputs = []
    for _ in range(2):
        phase = rng.uniform(-np.pi, np.pi, size=magnitude.shape)
        phase[0] = 0.0
        phase[-1] = 0.0
        surrogate = magnitude * np.exp(1j * phase)
        _, signal = istft(
            surrogate,
            fs=1.0,
            window="hann",
            nperseg=ANF_NFFT,
            noverlap=ANF_NFFT // 4,
            nfft=ANF_NFFT,
            input_onesided=True,
            boundary=True,
            time_axis=-1,
            freq_axis=-2,
            scaling="spectrum",
        )
        outputs.append(normalize(signal[:SOURCE_LENGTH]))
    return np.stack(outputs)


def main() -> None:
    row = make_bundles("smoke", 42)["train"][0]["recipes"][0]
    path = DEMAND_ROOT / str(row["noise_scene"]) / f"ch{int(row['noise_channel_a']):02d}.wav"
    start_a = _noise_start(path, "train", float(row["noise_u_a"]), SAMPLE_RATE, SOURCE_LENGTH)
    start_b = _noise_start(path, "train", (float(row["noise_u_a"]) + 0.5) % 1.0, SAMPLE_RATE, SOURCE_LENGTH)
    source_a = normalize(_load_noise_cached(str(path), SAMPLE_RATE)[start_a : start_a + SOURCE_LENGTH])
    source_b = normalize(_load_noise_cached(str(path), SAMPLE_RATE)[start_b : start_b + SOURCE_LENGTH])
    spacing = load_subject_spacings(Path("/disk2/bywang/data/HRTF/anthropometry/anthro.mat"))[
        str(row["subject_id"])
    ]["spacing_m"]
    params = ANFParameters(
        mic_positions=np.asarray([[-spacing / 2, 0, 0], [spacing / 2, 0, 0]]),
        sc_type="spherical",
        sample_frequency=SAMPLE_RATE,
        nfft=ANF_NFFT,
    )
    methods = {
        "stationary_gaussian": lambda seed: psd_shaped_gaussian(source_a, source_b, seed),
        "envelope_50ms": lambda seed: envelope_modulated_gaussian(source_a, source_b, seed, 801),
        "envelope_100ms": lambda seed: envelope_modulated_gaussian(source_a, source_b, seed, 1601),
        "envelope_200ms": lambda seed: envelope_modulated_gaussian(source_a, source_b, seed, 3201),
        "shared_spectrogram": lambda seed: shared_spectrogram_random_phase(source_a, source_b, seed),
    }
    trials = 16
    for name, factory in methods.items():
        trial_nmse = []
        input_corr = []
        input_cv = []
        output_cv = []
        generated_trials = []
        target_lr = None
        for trial in range(trials):
            inputs = factory(1000 + trial)
            generated, target, _ = generate_signals(
                inputs, params, decomposition="evd", processing="balance+smooth"
            )
            noise = generated[:, CONTEXT : CONTEXT + LENGTH]
            estimated = estimate_coherence(noise, ANF_NFFT)
            target_lr = target.matrix[0, 1, 1:]
            error = estimated[0, 1, 1:] - target_lr
            trial_nmse.append(10.0 * np.log10(
                np.mean(np.abs(error) ** 2) / (np.mean(np.abs(target_lr) ** 2) + 1e-12)
            ))
            input_corr.append(float(np.mean(inputs[0] * inputs[1])))
            input_cv.append(frame_rms_cv(inputs[0]))
            output_cv.append(frame_rms_cv(noise[0]))
            generated_trials.append(noise)
        aggregate = estimate_coherence(np.concatenate(generated_trials, axis=1), ANF_NFFT)
        aggregate_error = aggregate[0, 1, 1:] - target_lr
        aggregate_nmse = 10.0 * np.log10(
            np.mean(np.abs(aggregate_error) ** 2) / (np.mean(np.abs(target_lr) ** 2) + 1e-12)
        )
        print(
            f"{name:22s} corr_abs={np.mean(np.abs(input_corr)):.4f} "
            f"coh_nmse_med={np.median(trial_nmse):+.3f} dB "
            f"coh_nmse_agg={aggregate_nmse:+.3f} dB "
            f"input_rms_cv={np.mean(input_cv):.3f} output_rms_cv={np.mean(output_cv):.3f}"
        )


if __name__ == "__main__":
    main()
