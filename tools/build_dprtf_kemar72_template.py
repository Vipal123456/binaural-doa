#!/usr/bin/env python3
"""Build a 72-class KEMAR direct-path RTF template from a SOFA file."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import scipy.io
import scipy.signal
from netCDF4 import Dataset


def _stft_np(signal: np.ndarray, n_fft: int, win_length: int, sample_rate: int, fre_used_ratio: float) -> np.ndarray:
    win_shift = win_length
    nfft_valid = int(n_fft / 2) + 1
    nf = int(nfft_valid * fre_used_ratio)
    nt = int((signal.shape[0] - win_length) / win_shift) + 1
    stft = np.zeros((nf, nt, signal.shape[1]), dtype=np.complex64)
    window = scipy.signal.get_window(window="hann", Nx=win_length)
    for ch_idx in range(signal.shape[1]):
        _, _, stft_temp = scipy.signal.stft(
            signal[:, ch_idx],
            fs=sample_rate,
            window=window,
            nperseg=win_length,
            noverlap=0,
            nfft=n_fft,
            boundary=None,
            padded=True,
        )
        stft[:, :, ch_idx] = stft_temp[0:nf, 0:nt]
    return stft


def _atf_to_rtf(atf: np.ndarray) -> np.ndarray:
    rtf_complex = atf[:, :, 1] / (atf[:, :, 0] + 1e-10)
    rtf_mag = np.abs(rtf_complex)
    rtf_phase = np.angle(rtf_complex)
    ild = np.log10(rtf_mag + 1e-10)
    ipd_sin = np.sin(rtf_phase)
    ipd_cos = np.cos(rtf_phase)
    return np.vstack((ild, ipd_sin, ipd_cos)).astype(np.float32)


def _resample_hrir(hrir: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return hrir
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return scipy.signal.resample_poly(hrir, up=up, down=down, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build KEMAR72 DP-RTF template")
    parser.add_argument(
        "--sofa_path",
        type=str,
        default="/disk2/bywang/data/sofamyroom/data/MIT_KEMAR_normal_pinna.sofa",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/disk2/bywang/DOA-net/data/dprtf_templates/rtf_dp_kemar72_16k.mat",
    )
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--win_length", type=int, default=512)
    parser.add_argument("--max_freq_hz", type=float, default=4000.0)
    args = parser.parse_args()

    sofa_path = Path(args.sofa_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = Dataset(str(sofa_path), "r")
    source_pos = np.asarray(ds.variables["SourcePosition"][:], dtype=np.float64)
    ir = np.asarray(ds.variables["Data.IR"][:], dtype=np.float32)  # [M, R, N]
    orig_sr = int(np.asarray(ds.variables["Data.SamplingRate"][:]).reshape(-1)[0])

    azimuths = list(range(0, 360, 5))
    elevation = 0.0
    fre_used_ratio = float(args.max_freq_hz) / (args.sample_rate / 2.0)

    rtf_columns = []
    matched_rows = []
    for az in azimuths:
        row_idx = np.where(
            np.isclose(np.mod(source_pos[:, 0], 360.0), az, atol=1e-6)
            & np.isclose(source_pos[:, 1], elevation, atol=1e-6)
        )[0]
        if len(row_idx) != 1:
            raise RuntimeError(f"Expected one SOFA row for az={az}, el={elevation}, got {len(row_idx)}")
        row = int(row_idx[0])
        matched_rows.append(row)

        hrir = ir[row].transpose(1, 0)  # [N, 2]
        hrir = _resample_hrir(hrir, orig_sr=orig_sr, target_sr=args.sample_rate)
        if hrir.shape[0] < args.n_fft:
            pad = np.zeros((args.n_fft - hrir.shape[0], hrir.shape[1]), dtype=np.float32)
            hrir = np.concatenate((hrir, pad), axis=0)
        elif hrir.shape[0] > args.n_fft:
            hrir = hrir[: args.n_fft, :]

        stft_rir = _stft_np(
            signal=hrir,
            n_fft=args.n_fft,
            win_length=args.win_length,
            sample_rate=args.sample_rate,
            fre_used_ratio=fre_used_ratio,
        )
        rtf_columns.append(_atf_to_rtf(stft_rir))

    rtf_dp = np.hstack(rtf_columns)
    scipy.io.savemat(
        str(output_path),
        {
            "rtf_dp": rtf_dp,
            "azimuth_deg": np.asarray(azimuths, dtype=np.float32)[np.newaxis, :],
            "sofa_rows": np.asarray(matched_rows, dtype=np.int32)[np.newaxis, :],
            "sample_rate": np.asarray([[args.sample_rate]], dtype=np.int32),
            "n_fft": np.asarray([[args.n_fft]], dtype=np.int32),
            "max_freq_hz": np.asarray([[args.max_freq_hz]], dtype=np.float32),
        },
    )
    print(f"Saved template: {output_path}")
    print(f"rtf_dp shape: {rtf_dp.shape}")


if __name__ == "__main__":
    main()
