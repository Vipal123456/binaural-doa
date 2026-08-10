#!/usr/bin/env python3
"""Build a 25-angle DP-RTF template from training-subject CIPIC HRIRs.

The template is the feature-space mean over training heads. Validation and
test heads are deliberately excluded so the resulting baseline does not use
unseen-subject calibration data.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import scipy.io
import scipy.signal
from netCDF4 import Dataset


CLASS_ANGLES_DEG = (
    -80, -65, -55, -45, -40, -35, -30, -25, -20, -15, -10, -5,
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 55, 65, 80,
)

TRAIN_SUBJECTS = (
    "058", "059", "060", "048", "050", "051", "010", "028", "124",
    "011", "012", "165", "147", "148", "152", "044", "127", "156",
    "015", "017", "018", "134", "135", "137", "158", "162", "163",
    "153", "154", "155",
)


def wrap_deg(angle: np.ndarray | float) -> np.ndarray:
    return (np.asarray(angle, dtype=np.float64) + 180.0) % 360.0 - 180.0


def resample_hrir(hrir: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return np.asarray(hrir, dtype=np.float32)
    divisor = math.gcd(int(source_sr), int(target_sr))
    return scipy.signal.resample_poly(
        hrir,
        target_sr // divisor,
        source_sr // divisor,
        axis=0,
    ).astype(np.float32, copy=False)


def hrir_to_dprtf(hrir: np.ndarray, n_fft: int, freq_bins_used: int) -> np.ndarray:
    if hrir.shape[0] < n_fft:
        hrir = np.pad(hrir, ((0, n_fft - hrir.shape[0]), (0, 0)))
    else:
        hrir = hrir[:n_fft]

    window = scipy.signal.get_window("hann", n_fft)
    spectra = []
    for channel in range(2):
        _, _, stft = scipy.signal.stft(
            hrir[:, channel],
            window=window,
            nperseg=n_fft,
            noverlap=0,
            nfft=n_fft,
            boundary=None,
            padded=True,
        )
        spectra.append(stft[:freq_bins_used, 0])

    rtf = spectra[1] / (spectra[0] + 1e-10)
    magnitude = np.abs(rtf)
    phase = np.angle(rtf)
    return np.concatenate(
        [np.log10(magnitude + 1e-10), np.sin(phase), np.cos(phase)]
    ).astype(np.float32)


def load_subject_features(
    sofa_path: Path,
    sample_rate: int,
    n_fft: int,
    freq_bins_used: int,
) -> np.ndarray:
    with Dataset(str(sofa_path), "r") as database:
        positions = np.asarray(database.variables["SourcePosition"][:], dtype=np.float64)
        impulse_responses = np.asarray(database.variables["Data.IR"][:], dtype=np.float32)
        source_sr = int(
            round(float(np.asarray(database.variables["Data.SamplingRate"][:]).reshape(-1)[0]))
        )

    sofa_azimuths = wrap_deg(positions[:, 0])
    elevations = positions[:, 1]
    columns = []
    for project_angle in CLASS_ANGLES_DEG:
        target_sofa_azimuth = float(wrap_deg(-project_angle))
        azimuth_error = np.abs(wrap_deg(sofa_azimuths - target_sofa_azimuth))
        matches = np.flatnonzero(
            np.isclose(elevations, 0.0, atol=1e-7)
            & np.isclose(azimuth_error, 0.0, atol=1e-7)
        )
        if len(matches) != 1:
            raise ValueError(
                f"Expected one horizontal HRIR for project angle {project_angle} in "
                f"{sofa_path}, found {len(matches)}"
            )
        hrir = impulse_responses[int(matches[0])].transpose(1, 0)
        hrir = resample_hrir(hrir, source_sr, sample_rate)
        columns.append(hrir_to_dprtf(hrir, n_fft, freq_bins_used))
    return np.stack(columns, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a train-head mean CIPIC-25 DP-RTF template"
    )
    parser.add_argument("--hrtf_root", type=Path, default=Path("/disk2/bywang/data/HRTF"))
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("data/dprtf_templates/rtf_dp_cipic25_train30_mean_16k.mat"),
    )
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--freq_bins_used", type=int, default=128)
    args = parser.parse_args()

    subject_features = []
    for subject in TRAIN_SUBJECTS:
        path = args.hrtf_root / f"subject_{subject}.sofa"
        if not path.is_file():
            raise FileNotFoundError(path)
        subject_features.append(
            load_subject_features(path, args.sample_rate, args.n_fft, args.freq_bins_used)
        )

    feature_set = np.stack(subject_features, axis=2)
    template = np.mean(feature_set, axis=2, dtype=np.float64).astype(np.float32)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(
        args.output_path,
        {
            "rtf_dp": template,
            "rtf_dp_subject_set": feature_set,
            "azimuth_deg": np.asarray(CLASS_ANGLES_DEG, dtype=np.float32)[None, :],
            "subject_ids": np.asarray(TRAIN_SUBJECTS, dtype="U3")[None, :],
            "sample_rate": np.asarray([[args.sample_rate]], dtype=np.int32),
            "n_fft": np.asarray([[args.n_fft]], dtype=np.int32),
            "freq_bins_used": np.asarray([[args.freq_bins_used]], dtype=np.int32),
            "template_policy": np.asarray(["training-subject feature mean"], dtype="U64"),
        },
    )
    print(f"Saved template: {args.output_path}")
    print(f"rtf_dp shape: {template.shape}")
    print(f"training subjects: {len(TRAIN_SUBJECTS)}")


if __name__ == "__main__":
    main()
