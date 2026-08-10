#!/usr/bin/env python3
"""Prepare a robust subject-disjoint binaural DOA dataset.

The dataset is generated from mono LibriSpeech utterances, CIPIC SOFA HRTFs,
pyroomacoustics room responses, and DEMAND noise.  The key invariant is:

    metadata azimuth == selected HRTF azimuth == room source azimuth

This avoids the label/room-direction mismatch that can corrupt DOA training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyroomacoustics as pra
import sofa
import soundfile as sf
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt


DEMAND_SCENES = ["OOFFICE", "PCAFETER", "TMETRO", "TBUS", "SPSQUARE", "NPARK"]


@dataclass(frozen=True)
class SubjectSplit:
    train: List[str]
    val: List[str]
    test: List[str]


def wrap_deg(angle: float) -> float:
    return ((angle + 180.0) % 360.0) - 180.0


def angular_error_deg(a: float, b: float) -> float:
    return abs(wrap_deg(a - b))


def bin_center(bin_idx: int, num_bins: int = 72) -> float:
    return -180.0 + (bin_idx + 0.5) * (360.0 / num_bins)


def angle_to_label(angle_deg: float, num_bins: int = 72) -> int:
    wrapped = wrap_deg(angle_deg)
    width = 360.0 / float(num_bins)
    idx = int(math.floor((wrapped + 180.0) / width))
    return max(0, min(num_bins - 1, idx))


def make_balanced_shuffled_bins(count: int, num_classes: int, np_rng: np.random.Generator) -> np.ndarray:
    """Return an approximately balanced class schedule in randomized order."""
    repeats = int(np.ceil(count / float(num_classes)))
    bins = np.tile(np.arange(num_classes, dtype=np.int64), repeats)[:count]
    np_rng.shuffle(bins)
    return bins


def spherical_to_cartesian(az_deg: float, el_deg: float, radius: float) -> Tuple[float, float, float]:
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    x = radius * math.cos(el) * math.sin(az)
    y = radius * math.cos(el) * math.cos(az)
    z = radius * math.sin(el)
    return x, y, z


def read_metadata_azimuth(path: Path) -> float:
    arr = np.loadtxt(path, delimiter=",")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    mid = len(arr) // 2
    return float(np.degrees(np.arctan2(arr[mid, 1], arr[mid, 2])))


def write_metadata_csv(path: Path, az_deg: float, el_deg: float, radius: float) -> None:
    x, y, z = spherical_to_cartesian(az_deg, el_deg, radius)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([0, x, y, z, 0, 0, 0, 0])


def list_subjects(hrtf_root: Path) -> List[str]:
    subjects = sorted(p.stem.replace("subject_", "") for p in hrtf_root.glob("subject_*.sofa"))
    if not subjects:
        raise FileNotFoundError(f"No subject_*.sofa found in {hrtf_root}")
    return subjects


def choose_subjects(
    all_subjects: Sequence[str],
    total_subjects: int,
    seed: int,
    force_include: Sequence[str],
) -> SubjectSplit:
    if len(all_subjects) < total_subjects:
        raise ValueError(f"Need {total_subjects} subjects, found {len(all_subjects)}")

    forced = [s for s in force_include if s in all_subjects]
    remaining = [s for s in all_subjects if s not in forced]
    rng = random.Random(seed)
    rng.shuffle(remaining)
    selected = forced + remaining[: total_subjects - len(forced)]

    # Keep the historical deterministic/readable order after selection.
    selected = sorted(selected)
    n_train = int(round(total_subjects * 0.8))
    n_val = max(1, int(round(total_subjects * 0.1)))
    n_test = total_subjects - n_train - n_val
    if n_test < 1:
        raise ValueError("Need at least one test subject")
    return SubjectSplit(
        train=selected[:n_train],
        val=selected[n_train : n_train + n_val],
        test=selected[n_train + n_val :],
    )


def list_librispeech_files(root: Path) -> List[Path]:
    files = sorted(root.rglob("*.flac"))
    if not files:
        raise FileNotFoundError(f"No .flac files found under {root}")
    return files


def list_noise_files(demand_root: Path, scenes: Sequence[str]) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for scene in scenes:
        scene_dir = demand_root / scene
        wavs = sorted(scene_dir.glob("ch*.wav"))
        if not wavs:
            raise FileNotFoundError(f"No ch*.wav noise files found in {scene_dir}")
        out[scene] = wavs
    return out


def resample_1d(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return x.astype(np.float32, copy=False)
    g = math.gcd(int(orig_sr), int(target_sr))
    return resample_poly(x, target_sr // g, orig_sr // g).astype(np.float32, copy=False)


def load_mono_resampled(path: Path, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(path, always_2d=False, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return resample_1d(np.asarray(audio, dtype=np.float32), sr, target_sr)


def fit_to_length(audio: np.ndarray, num_samples: int, rng: np.random.Generator) -> np.ndarray:
    if len(audio) == 0:
        return np.zeros(num_samples, dtype=np.float32)
    if len(audio) >= num_samples:
        max_start = len(audio) - num_samples
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        return audio[start : start + num_samples].astype(np.float32, copy=False)
    reps = int(np.ceil(num_samples / len(audio)))
    return np.tile(audio, reps)[:num_samples].astype(np.float32, copy=False)


def read_noise_segment(path: Path, length: int, target_sr: int, rng: random.Random) -> np.ndarray:
    info = sf.info(str(path))
    if info.frames >= length and info.samplerate == target_sr:
        start = rng.randint(0, max(0, info.frames - length))
        noise, sr = sf.read(str(path), start=start, frames=length, dtype="float32", always_2d=False)
    else:
        noise, sr = sf.read(str(path), dtype="float32", always_2d=False)

    noise = np.asarray(noise, dtype=np.float32)
    if noise.ndim > 1:
        noise = noise[:, 0]
    if sr != target_sr:
        noise = resample_1d(noise, sr, target_sr)
    if len(noise) < length:
        reps = int(np.ceil(length / max(1, len(noise))))
        noise = np.tile(noise, reps)
    if len(noise) > length:
        start = rng.randint(0, max(0, len(noise) - length))
        noise = noise[start : start + length]
    return noise.astype(np.float32, copy=False)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    signal = signal.astype(np.float32, copy=False)
    noise = noise.astype(np.float32, copy=False)
    noise = noise - float(np.mean(noise))

    sig_power = float(np.mean(signal.astype(np.float64) ** 2))
    noise_power = float(np.mean(noise.astype(np.float64) ** 2))
    if sig_power < 1e-12 or noise_power < 1e-12:
        return signal.copy()

    snr_linear = 10.0 ** (snr_db / 10.0)
    scale = math.sqrt(sig_power / (snr_linear * noise_power))
    return (signal + scale * noise).astype(np.float32, copy=False)


def peak_normalize(stereo: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_abs = float(np.max(np.abs(stereo)))
    if max_abs > peak and max_abs > 1e-8:
        stereo = stereo * (peak / max_abs)
    return stereo.astype(np.float32, copy=False)


def stereo_join_metric(stereo: np.ndarray, join_sample: int, sample_rate: int, window_ms: float = 10.0) -> float:
    if stereo.ndim != 2:
        return float("nan")
    n = max(1, int(round((window_ms / 1000.0) * sample_rate)))
    pre = stereo[max(0, join_sample - n):join_sample]
    post = stereo[join_sample:min(stereo.shape[0], join_sample + n)]
    pre_rms = rms(pre) if pre.size else 0.0
    post_rms = rms(post) if post.size else 0.0
    return float(abs(pre_rms - post_rms) / max(pre_rms, 1e-8))


def room_profile_params(profile: str, rng: random.Random) -> Tuple[Tuple[float, float, float], float]:
    if profile == "small":
        dims = (rng.uniform(3.5, 5.0), rng.uniform(3.0, 4.5), rng.uniform(2.5, 3.0))
        rt60 = rng.uniform(0.20, 0.45)
    elif profile == "medium":
        dims = (rng.uniform(5.0, 7.5), rng.uniform(4.0, 6.5), rng.uniform(2.7, 3.2))
        rt60 = rng.uniform(0.35, 0.65)
    elif profile == "large":
        dims = (rng.uniform(7.5, 10.0), rng.uniform(6.0, 8.5), rng.uniform(3.0, 3.8))
        rt60 = rng.uniform(0.50, 0.80)
    else:
        raise ValueError(f"Unknown room profile: {profile}")
    return dims, rt60


def choose_room_geometry(
    dims: Tuple[float, float, float],
    az_deg: float,
    min_distance: float,
    max_distance: float,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, float]:
    lx, ly, lz = dims
    height = min(1.5, lz - 0.7)
    margin = 0.55
    az = math.radians(az_deg)

    for _ in range(200):
        distance_hi = min(max_distance, 0.45 * min(lx, ly))
        distance_lo = min(min_distance, distance_hi)
        distance = rng.uniform(distance_lo, distance_hi)
        head = np.array([
            rng.uniform(1.0, lx - 1.0),
            rng.uniform(1.0, ly - 1.0),
            height,
        ], dtype=np.float64)
        source = head + np.array([
            distance * math.sin(az),
            distance * math.cos(az),
            0.0,
        ], dtype=np.float64)
        if (
            margin <= source[0] <= lx - margin
            and margin <= source[1] <= ly - margin
            and margin <= source[2] <= lz - margin
        ):
            return head, source, distance

    # Conservative fallback for very small rooms/awkward angles.
    head = np.array([lx / 2.0, ly / 2.0, height], dtype=np.float64)
    max_dx = (lx / 2.0 - margin) / max(abs(math.sin(az)), 1e-6)
    max_dy = (ly / 2.0 - margin) / max(abs(math.cos(az)), 1e-6)
    distance = max(0.6, min(max_distance, max_dx, max_dy))
    source = head + np.array([distance * math.sin(az), distance * math.cos(az), 0.0])
    source[0] = np.clip(source[0], margin, lx - margin)
    source[1] = np.clip(source[1], margin, ly - margin)
    return head, source, float(distance)


def synthesize_room_rir(
    sample_rate: int,
    dims: Tuple[float, float, float],
    rt60: float,
    head_center: np.ndarray,
    source_xyz: np.ndarray,
) -> np.ndarray:
    absorption, max_order = pra.inverse_sabine(rt60, dims)
    room = pra.ShoeBox(
        dims,
        fs=sample_rate,
        materials=pra.Material(absorption),
        max_order=max_order,
    )
    room.add_source(source_xyz)
    room.add_microphone_array(head_center.reshape(3, 1))
    room.compute_rir()
    rir = np.asarray(room.rir[0][0], dtype=np.float32)
    if len(rir) == 0:
        rir = np.array([1.0], dtype=np.float32)
    rir = rir / max(float(np.max(np.abs(rir))), 1e-8)
    return rir


def estimate_rt60_from_ir(ir: np.ndarray, sample_rate: int) -> float:
    """Rough Schroeder RT60 estimate for reporting/debugging."""
    if ir.size == 0 or np.max(np.abs(ir)) < 1e-8:
        return float("nan")
    energy = np.sum(np.square(ir.astype(np.float64)))
    if energy <= 1e-12:
        return float("nan")
    edc = np.cumsum(np.square(ir[::-1].astype(np.float64)))[::-1]
    edc_db = 10.0 * np.log10(np.maximum(edc / np.max(edc), 1e-12))
    times = np.arange(len(ir), dtype=np.float64) / float(sample_rate)
    mask = (edc_db <= -5.0) & (edc_db >= -35.0)
    if mask.sum() < 8:
        return float("nan")
    slope, _ = np.polyfit(times[mask], edc_db[mask], deg=1)
    if slope >= -1e-6:
        return float("nan")
    return float(-60.0 / slope)


def room_profile_rt60_range(profile: str) -> Tuple[float, float]:
    if profile == "small":
        return 0.20, 0.45
    if profile == "medium":
        return 0.35, 0.65
    if profile == "large":
        return 0.50, 0.80
    raise ValueError(f"Unknown room profile: {profile}")


def estimate_drr_from_brir(
    brir: np.ndarray,
    direct_delay_samples: int,
    sample_rate: int,
    direct_window_ms: float = 2.5,
) -> float:
    direct_window = max(1, int(round((direct_window_ms / 1000.0) * sample_rate)))
    direct_start = max(0, int(direct_delay_samples))
    direct_end = min(brir.shape[-1], direct_start + direct_window)
    direct_energy = float(np.sum(brir[:, direct_start:direct_end].astype(np.float64) ** 2))
    reverb_energy = float(np.sum(brir[:, direct_end:].astype(np.float64) ** 2))
    if direct_energy <= 1e-12 or reverb_energy <= 1e-12:
        return float("nan")
    return float(10.0 * np.log10(direct_energy / reverb_energy))


def decompose_brir_energies(
    brir: np.ndarray,
    direct_delay_samples: int,
    late_start_sample: int,
    sample_rate: int,
    direct_window_ms: float = 2.5,
) -> Dict[str, float]:
    direct_window = max(1, int(round((direct_window_ms / 1000.0) * sample_rate)))
    direct_start = max(0, int(direct_delay_samples))
    direct_end = min(brir.shape[-1], direct_start + direct_window)
    late_start = max(direct_end, int(late_start_sample))
    direct_energy = float(np.sum(brir[:, direct_start:direct_end].astype(np.float64) ** 2))
    early_energy = float(np.sum(brir[:, direct_end:late_start].astype(np.float64) ** 2))
    late_energy = float(np.sum(brir[:, late_start:].astype(np.float64) ** 2))
    reverberant_energy = early_energy + late_energy
    return {
        "direct_energy": direct_energy,
        "early_energy": early_energy,
        "late_energy": late_energy,
        "reverberant_energy": reverberant_energy,
        "direct_energy_db": float(10.0 * np.log10(max(direct_energy, 1e-12))),
        "early_energy_db": float(10.0 * np.log10(max(early_energy, 1e-12))),
        "late_energy_db": float(10.0 * np.log10(max(late_energy, 1e-12))),
    }


def estimate_early_late_ratio_db(brir: np.ndarray, split_sample: int) -> float:
    early_energy = float(np.sum(brir[:, :split_sample].astype(np.float64) ** 2))
    late_energy = float(np.sum(brir[:, split_sample:].astype(np.float64) ** 2))
    if early_energy <= 1e-12 or late_energy <= 1e-12:
        return float("nan")
    return float(10.0 * np.log10(early_energy / late_energy))


def band_limited_noise(x: np.ndarray, sample_rate: int, kind: str) -> np.ndarray:
    nyq = sample_rate * 0.5
    if kind == "low":
        sos = butter(4, 500.0 / nyq, btype="lowpass", output="sos")
    elif kind == "mid":
        sos = butter(4, [500.0 / nyq, 2000.0 / nyq], btype="bandpass", output="sos")
    elif kind == "high":
        sos = butter(4, 2000.0 / nyq, btype="highpass", output="sos")
    else:
        raise ValueError(f"Unsupported band kind: {kind}")
    return sosfilt(sos, x).astype(np.float32, copy=False)


def generate_binaural_diffuse_late_tail(
    length: int,
    sample_rate: int,
    target_rt60: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(length).astype(np.float32)
    indep_l = rng.standard_normal(length).astype(np.float32)
    indep_r = rng.standard_normal(length).astype(np.float32)
    shared_lf = rng.standard_normal(length).astype(np.float32)
    shared_hf = rng.standard_normal(length).astype(np.float32)

    low_common = band_limited_noise(common, sample_rate, "low")
    mid_common = band_limited_noise(common, sample_rate, "mid")
    high_common = band_limited_noise(common, sample_rate, "high")
    low_shared = band_limited_noise(shared_lf, sample_rate, "low")
    high_shared = band_limited_noise(shared_hf, sample_rate, "high")
    low_l = band_limited_noise(indep_l, sample_rate, "low")
    low_r = band_limited_noise(indep_r, sample_rate, "low")
    mid_l = band_limited_noise(indep_l, sample_rate, "mid")
    mid_r = band_limited_noise(indep_r, sample_rate, "mid")
    high_l = band_limited_noise(indep_l, sample_rate, "high")
    high_r = band_limited_noise(indep_r, sample_rate, "high")

    # Approximate diffuse-field binaural coherence: strongly correlated below
    # 500 Hz, moderately correlated in the mid band, and still weakly related
    # at high frequencies rather than fully independent.
    low_alpha, mid_alpha, high_alpha = 0.985, 0.72, 0.32
    low_shared_w, mid_shared_w, high_shared_w = 0.30, 0.10, 0.05
    tail_l = (
        1.00 * (
            low_alpha * low_common
            + low_shared_w * low_shared
            + math.sqrt(max(1e-6, 1.0 - low_alpha**2 - low_shared_w**2)) * low_l
        )
        + 0.85 * (
            mid_alpha * mid_common
            + mid_shared_w * low_shared
            + math.sqrt(max(1e-6, 1.0 - mid_alpha**2 - mid_shared_w**2)) * mid_l
        )
        + 0.45 * (
            high_alpha * high_common
            + high_shared_w * high_shared
            + math.sqrt(max(1e-6, 1.0 - high_alpha**2 - high_shared_w**2)) * high_l
        )
    )
    tail_r = (
        1.00 * (
            low_alpha * low_common
            + low_shared_w * low_shared
            + math.sqrt(max(1e-6, 1.0 - low_alpha**2 - low_shared_w**2)) * low_r
        )
        + 0.85 * (
            mid_alpha * mid_common
            + mid_shared_w * low_shared
            + math.sqrt(max(1e-6, 1.0 - mid_alpha**2 - mid_shared_w**2)) * mid_r
        )
        + 0.45 * (
            high_alpha * high_common
            + high_shared_w * high_shared
            + math.sqrt(max(1e-6, 1.0 - high_alpha**2 - high_shared_w**2)) * high_r
        )
    )

    t = np.arange(length, dtype=np.float32) / float(sample_rate)
    envelope = np.power(10.0, (-3.0 * t) / max(float(target_rt60), 1e-3)).astype(np.float32)
    tail = np.stack([tail_l, tail_r], axis=0).astype(np.float32)
    tail *= envelope[None, :]
    return tail.astype(np.float32, copy=False)


def corrcoef_stereo(x: np.ndarray) -> float:
    if x.ndim != 2 or x.shape[0] != 2 or x.shape[1] < 2:
        return float("nan")
    return float(np.corrcoef(x[0], x[1])[0, 1])


def bandwise_corr_summary(stereo: np.ndarray, sample_rate: int) -> Dict[str, float]:
    low = np.stack([
        band_limited_noise(stereo[0], sample_rate, "low"),
        band_limited_noise(stereo[1], sample_rate, "low"),
    ], axis=0)
    mid = np.stack([
        band_limited_noise(stereo[0], sample_rate, "mid"),
        band_limited_noise(stereo[1], sample_rate, "mid"),
    ], axis=0)
    high = np.stack([
        band_limited_noise(stereo[0], sample_rate, "high"),
        band_limited_noise(stereo[1], sample_rate, "high"),
    ], axis=0)
    return {
        "low_band_corr": corrcoef_stereo(low),
        "mid_band_corr": corrcoef_stereo(mid),
        "high_band_corr": corrcoef_stereo(high),
    }


def image_source_axis_positions(source_coord: float, room_len: float, max_order: int) -> List[Tuple[float, int]]:
    """Return 1D image-source coordinates and approximate reflection orders."""
    out = []
    for n in range(-max_order, max_order + 1):
        for q in (0, 1):
            coord = 2.0 * n * room_len + ((-1.0) ** q) * source_coord
            order = abs(2 * n + q)
            if order <= max_order:
                out.append((coord, order))
    return out


def synthesize_pathwise_hrtf_brir(
    subject: "HRTFSubject",
    sample_rate: int,
    dims: Tuple[float, float, float],
    rt60: float,
    head_center: np.ndarray,
    source_xyz: np.ndarray,
    max_order: int,
    brir_seconds: float,
    auto_order: bool = False,
    max_auto_order_cap: int = 8,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Approximate BRIR by spatializing image-source paths with CIPIC HRIRs.

    This is an image-source path-wise HRTF rendering approximation: each direct
    or reflected path contributes a delayed/scaled HRIR selected by arrival
    direction. It is more realistic than mono-RIR followed by one target HRIR,
    but remains a controlled simulated BRIR-like protocol.
    """
    c = 343.0
    absorption, sabine_order = pra.inverse_sabine(rt60, dims)
    requested_max_order = int(max_order)
    if auto_order:
        max_order = min(int(sabine_order), int(max_auto_order_cap))
    max_order = int(max_order)
    beta = math.sqrt(max(0.0, 1.0 - float(absorption)))
    brir_len = int(round(brir_seconds * sample_rate))
    brir_l = np.zeros(brir_len, dtype=np.float32)
    brir_r = np.zeros(brir_len, dtype=np.float32)

    xs = image_source_axis_positions(float(source_xyz[0]), dims[0], max_order)
    ys = image_source_axis_positions(float(source_xyz[1]), dims[1], max_order)
    zs = image_source_axis_positions(float(source_xyz[2]), dims[2], max_order)

    num_paths = 0
    direct_rendered_az = None
    direct_rendered_el = None
    path_debug: List[Dict[str, object]] = []
    for x, ox in xs:
        for y, oy in ys:
            for z, oz in zs:
                order = ox + oy + oz
                if order > max_order:
                    continue
                image = np.array([x, y, z], dtype=np.float64)
                vec = image - head_center
                dist = float(np.linalg.norm(vec))
                if dist < 1e-6:
                    continue
                delay = int(round((dist / c) * sample_rate))
                if delay >= brir_len:
                    continue
                az = math.degrees(math.atan2(vec[0], vec[1]))
                el = math.degrees(math.asin(np.clip(vec[2] / dist, -1.0, 1.0)))
                measurement_idx, rendered_az, rendered_el, _ = subject.pick_direction(az, el)
                h_l = subject.ir[measurement_idx, 0]
                h_r = subject.ir[measurement_idx, 1]
                gain = (beta ** order) / max(dist, 1e-6)
                end = min(brir_len, delay + len(h_l))
                n = end - delay
                if n <= 0:
                    continue
                brir_l[delay:end] += (gain * h_l[:n]).astype(np.float32)
                brir_r[delay:end] += (gain * h_r[:n]).astype(np.float32)
                path_debug.append({
                    "path_id": num_paths,
                    "order": int(order),
                    "image_source_x": float(image[0]),
                    "image_source_y": float(image[1]),
                    "image_source_z": float(image[2]),
                    "distance_m": float(dist),
                    "delay_samples": int(delay),
                    "gain": float(gain),
                    "arrival_azimuth_deg": float(az),
                    "arrival_elevation_deg": float(el),
                    "selected_hrir_index": int(measurement_idx),
                    "selected_hrir_azimuth_deg": float(rendered_az),
                    "selected_hrir_elevation_deg": float(rendered_el),
                })
                num_paths += 1
                if order == 0:
                    direct_rendered_az = rendered_az
                    direct_rendered_el = rendered_el

    peak = max(float(np.max(np.abs(brir_l))), float(np.max(np.abs(brir_r))), 1e-8)
    brir = np.stack([brir_l / peak, brir_r / peak], axis=0).astype(np.float32)
    mono_proxy = 0.5 * (brir[0] + brir[1])
    report = {
        "brir_method": "image_source_pathwise_hrtf_brir",
        "max_order": int(max_order),
        "requested_max_order": int(requested_max_order),
        "auto_order": bool(auto_order),
        "max_auto_order_cap": int(max_auto_order_cap),
        "sabine_max_order": int(sabine_order),
        "num_paths": int(num_paths),
        "brir_seconds": float(brir_seconds),
        "absorption": float(absorption),
        "reflection_beta": float(beta),
        "estimated_rt60": estimate_rt60_from_ir(mono_proxy, sample_rate),
        "direct_rendered_azimuth_deg": direct_rendered_az,
        "direct_rendered_elevation_deg": direct_rendered_el,
        "path_debug": path_debug,
    }
    return brir, report


def synthesize_hybrid_pathwise_hrtf_brir_v3(
    subject: "HRTFSubject",
    sample_rate: int,
    dims: Tuple[float, float, float],
    rt60: float,
    room_profile: str,
    head_center: np.ndarray,
    source_xyz: np.ndarray,
    max_order: int,
    brir_seconds: float,
    early_cut_ms: float = 80.0,
    late_start_ms: float = 80.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    c = 343.0
    beta = math.sqrt(max(0.0, 1.0 - float(pra.inverse_sabine(rt60, dims)[0])))
    direct_vec = source_xyz - head_center
    direct_dist = float(np.linalg.norm(direct_vec))
    direct_delay = int(round((direct_dist / c) * sample_rate))
    early_cut_samples = int(round((early_cut_ms / 1000.0) * sample_rate))
    late_start_offset = int(round((late_start_ms / 1000.0) * sample_rate))
    late_start_sample = direct_delay + late_start_offset
    late_duration_s = min(1.8, max(float(brir_seconds), max(1.2, 2.5 * float(rt60))))
    late_duration_samples = int(round(late_duration_s * sample_rate))
    brir_len = max(
        int(round(brir_seconds * sample_rate)),
        late_start_sample + late_duration_samples + 256,
    )
    early_l = np.zeros(brir_len, dtype=np.float32)
    early_r = np.zeros(brir_len, dtype=np.float32)

    xs = image_source_axis_positions(float(source_xyz[0]), dims[0], max_order)
    ys = image_source_axis_positions(float(source_xyz[1]), dims[1], max_order)
    zs = image_source_axis_positions(float(source_xyz[2]), dims[2], max_order)

    direct_rendered_az = None
    direct_rendered_el = None
    early_path_count = 0
    path_debug: List[Dict[str, object]] = []
    for x, ox in xs:
        for y, oy in ys:
            for z, oz in zs:
                order = ox + oy + oz
                if order > max_order:
                    continue
                image = np.array([x, y, z], dtype=np.float64)
                vec = image - head_center
                dist = float(np.linalg.norm(vec))
                if dist < 1e-6:
                    continue
                delay = int(round((dist / c) * sample_rate))
                if delay > direct_delay + early_cut_samples:
                    continue
                az = math.degrees(math.atan2(vec[0], vec[1]))
                el = math.degrees(math.asin(np.clip(vec[2] / dist, -1.0, 1.0)))
                measurement_idx, rendered_az, rendered_el, _ = subject.pick_direction(az, el)
                h_l = subject.ir[measurement_idx, 0]
                h_r = subject.ir[measurement_idx, 1]
                gain = (beta ** order) / max(dist, 1e-6)
                end = min(brir_len, delay + len(h_l))
                n = end - delay
                if n <= 0:
                    continue
                early_l[delay:end] += (gain * h_l[:n]).astype(np.float32)
                early_r[delay:end] += (gain * h_r[:n]).astype(np.float32)
                path_debug.append({
                    "path_id": early_path_count,
                    "order": int(order),
                    "image_source_x": float(image[0]),
                    "image_source_y": float(image[1]),
                    "image_source_z": float(image[2]),
                    "distance_m": float(dist),
                    "delay_samples": int(delay),
                    "gain": float(gain),
                    "arrival_azimuth_deg": float(az),
                    "arrival_elevation_deg": float(el),
                    "selected_hrir_index": int(measurement_idx),
                    "selected_hrir_azimuth_deg": float(rendered_az),
                    "selected_hrir_elevation_deg": float(rendered_el),
                })
                early_path_count += 1
                if order == 0:
                    direct_rendered_az = rendered_az
                    direct_rendered_el = rendered_el

    early = np.stack([early_l, early_r], axis=0)
    anchor_start = min(brir_len - 1, direct_delay + int(round(0.060 * sample_rate)))
    anchor_end = min(brir_len, direct_delay + int(round(0.080 * sample_rate)))
    anchor_window_ms = "60-80"
    if anchor_end <= anchor_start or rms(early[:, anchor_start:anchor_end]) < 1e-4:
        anchor_start = min(brir_len - 1, direct_delay + int(round(0.040 * sample_rate)))
        anchor_end = min(brir_len, direct_delay + int(round(0.080 * sample_rate)))
        anchor_window_ms = "40-80"
    anchor_mono = 0.5 * (early[0, anchor_start:anchor_end] + early[1, anchor_start:anchor_end])
    anchor_rms = rms(anchor_mono) if anchor_end > anchor_start else 0.0
    anchor_energy = float(np.mean(anchor_mono.astype(np.float64) ** 2)) if anchor_end > anchor_start else 0.0

    seed = int(abs(hash((subject.subject_id, room_profile, round(rt60, 4), tuple(np.round(source_xyz, 3))))) % (2**32))
    late = generate_binaural_diffuse_late_tail(late_duration_samples, sample_rate, rt60, seed)
    fade_len = min(int(round(0.020 * sample_rate)), late.shape[1])
    if fade_len > 1:
        late[:, :fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)[None, :]
    initial_rms = rms(0.5 * (late[0, : max(fade_len, 1)] + late[1, : max(fade_len, 1)]))
    if anchor_rms > 1e-8 and initial_rms > 1e-8:
        late *= (anchor_rms / initial_rms)

    full = early.copy()
    late_end = min(brir_len, late_start_sample + late.shape[1])
    late_n = late_end - late_start_sample
    if late_n > 0:
        full[:, late_start_sample:late_end] += late[:, :late_n]

    pre = full[:, max(0, late_start_sample - int(round(0.010 * sample_rate))):late_start_sample]
    post = full[:, late_start_sample: min(brir_len, late_start_sample + int(round(0.010 * sample_rate)))]
    pre_rms = rms(pre) if pre.size else 0.0
    post_rms = rms(post) if post.size else 0.0
    late_join_metric = abs(pre_rms - post_rms) / max(pre_rms, 1e-8)
    late_join_ok = bool(late_join_metric < 0.35)

    peak = max(float(np.max(np.abs(full))), 1e-8)
    brir = (full / peak).astype(np.float32, copy=False)
    mono_proxy = 0.5 * (brir[0] + brir[1])
    energy_stats = decompose_brir_energies(brir, direct_delay, late_start_sample, sample_rate)
    target_drr = (
        float(10.0 * np.log10(max(energy_stats["direct_energy"], 1e-12) / max(energy_stats["reverberant_energy"], 1e-12)))
        if energy_stats["reverberant_energy"] > 1e-12 else float("nan")
    )
    estimated_drr = target_drr
    band_corr = bandwise_corr_summary(brir[:, late_start_sample:], sample_rate)
    early_delays = [int(row["delay_samples"]) for row in path_debug]
    early_last_delay_ms = (max(early_delays) / float(sample_rate) * 1000.0) if early_delays else 0.0
    report = {
        "brir_method": "hybrid_pathwise_hrtf_brir_v3",
        "max_order": int(max_order),
        "sabine_max_order": int(pra.inverse_sabine(rt60, dims)[1]),
        "num_paths": int(early_path_count),
        "early_path_count": int(early_path_count),
        "early_cut_ms": float(early_cut_ms),
        "late_start_ms": float(late_start_ms),
        "late_tail_type": "binaural_diffuse_statistical",
        "brir_seconds": float(brir_len / sample_rate),
        "absorption": float(pra.inverse_sabine(rt60, dims)[0]),
        "reflection_beta": float(beta),
        "estimated_rt60": estimate_rt60_from_ir(mono_proxy, sample_rate),
        "target_rt60_range_lo": float(room_profile_rt60_range(room_profile)[0]),
        "target_rt60_range_hi": float(room_profile_rt60_range(room_profile)[1]),
        "direct_delay_samples": int(direct_delay),
        "late_start_sample": int(late_start_sample),
        "late_join_ok": late_join_ok,
        "late_join_metric": float(late_join_metric),
        "late_anchor_window_ms": anchor_window_ms,
        "late_anchor_energy": float(anchor_energy),
        "target_drr_db": target_drr,
        "estimated_drr_db": estimated_drr,
        "estimated_early_late_ratio_db": estimate_early_late_ratio_db(brir, late_start_sample),
        "early_last_delay_ms": float(early_last_delay_ms),
        **energy_stats,
        "direct_rendered_azimuth_deg": direct_rendered_az,
        "direct_rendered_elevation_deg": direct_rendered_el,
        "left_right_corrcoef": float(np.corrcoef(brir[0], brir[1])[0, 1]) if brir.shape[1] > 1 else float("nan"),
        **band_corr,
        "path_debug": path_debug,
        "clean_brir": brir,
    }
    return brir, report


class HRTFSubject:
    def __init__(self, sofa_path: Path, target_sr: int):
        self.sofa_path = sofa_path
        self.subject_id = sofa_path.stem.replace("subject_", "")
        db = sofa.Database.open(str(sofa_path))
        self.positions = np.asarray(db.Source.Position.get_values(), dtype=np.float64)
        ir = np.asarray(db.Data.IR.get_values(), dtype=np.float32)
        sofa_sr = int(round(float(db.Data.SamplingRate.get_values()[0])))
        if sofa_sr != target_sr:
            g = math.gcd(sofa_sr, target_sr)
            ir = resample_poly(ir, target_sr // g, sofa_sr // g, axis=-1).astype(np.float32)
        self.ir = ir
        self.sofa_azimuths = np.array([wrap_deg(float(a)) for a in self.positions[:, 0]], dtype=np.float64)
        # Project convention: 0 deg is front (+y), +90 deg is listener right (+x).
        # The CIPIC SOFA files used here have the opposite azimuth sign, so expose
        # world-convention azimuths for lookup/reporting while keeping HRIR indices.
        self.azimuths = np.array([wrap_deg(-float(a)) for a in self.sofa_azimuths], dtype=np.float64)
        self.elevations = np.asarray(self.positions[:, 1], dtype=np.float64)
        self.radii = np.asarray(self.positions[:, 2], dtype=np.float64)

    def pick_measurement(self, target_az: float) -> Tuple[int, float, float, float]:
        return self.pick_direction(target_az, 0.0)

    def pick_direction(self, target_az: float, target_el: float = 0.0) -> Tuple[int, float, float, float]:
        az_err = np.array([angular_error_deg(float(a), target_az) for a in self.azimuths])
        score = az_err + 2.0 * np.abs(self.elevations - float(target_el))
        idx = int(np.argmin(score))
        return idx, float(self.azimuths[idx]), float(self.elevations[idx]), float(self.radii[idx])

    def apply(self, mono: np.ndarray, measurement_idx: int, num_samples: int) -> np.ndarray:
        h_l = self.ir[measurement_idx, 0]
        h_r = self.ir[measurement_idx, 1]
        left = fftconvolve(mono, h_l, mode="full")[:num_samples]
        right = fftconvolve(mono, h_r, mode="full")[:num_samples]
        stereo = np.stack([left, right], axis=1)
        return peak_normalize(stereo)


def render_sample(
    speech: np.ndarray,
    subject: HRTFSubject,
    target_bin: int,
    noise_files: Dict[str, List[Path]],
    scenes: Sequence[str],
    sample_rate: int,
    num_samples: int,
    source_distance_min: float,
    source_distance_max: float,
    rendering_mode: str,
    brir_max_order: int,
    brir_seconds: float,
    rng: random.Random,
    np_rng: np.random.Generator,
    no_noise: bool = False,
    fixed_snr_db: Optional[float] = None,
    brir_auto_order: bool = False,
    brir_max_auto_order_cap: int = 8,
) -> Tuple[np.ndarray, Dict[str, object]]:
    target_az = bin_center(target_bin)
    measurement_idx, az_deg, el_deg, radius = subject.pick_measurement(target_az)

    profile = rng.choice(["small", "medium", "large"])
    dims, rt60 = room_profile_params(profile, rng)
    head_center, source_xyz, source_distance = choose_room_geometry(
        dims,
        az_deg,
        source_distance_min,
        source_distance_max,
        rng,
    )
    room_rir = synthesize_room_rir(sample_rate, dims, rt60, head_center, source_xyz)

    brir_report: Dict[str, object] = {
        "brir_method": "mono_room_rir_then_hrtf",
        "max_order": None,
        "sabine_max_order": None,
        "num_paths": None,
        "brir_seconds": None,
        "absorption": None,
        "reflection_beta": None,
        "estimated_rt60": estimate_rt60_from_ir(room_rir, sample_rate),
        "direct_rendered_azimuth_deg": az_deg,
        "direct_rendered_elevation_deg": el_deg,
    }
    if rendering_mode == "mono_room_rir_then_hrtf":
        roomed_mono = fftconvolve(speech, room_rir, mode="full")[:num_samples].astype(np.float32)
        in_rms = rms(speech)
        out_rms = rms(roomed_mono)
        if out_rms > 1e-9:
            roomed_mono *= in_rms / out_rms
        stereo = subject.apply(roomed_mono, measurement_idx, num_samples)
    elif rendering_mode == "image_source_pathwise_hrtf_brir":
        brir, brir_report = synthesize_pathwise_hrtf_brir(
            subject=subject,
            sample_rate=sample_rate,
            dims=dims,
            rt60=rt60,
            head_center=head_center,
            source_xyz=source_xyz,
            max_order=brir_max_order,
            brir_seconds=brir_seconds,
            auto_order=brir_auto_order,
            max_auto_order_cap=brir_max_auto_order_cap,
        )
        left = fftconvolve(speech, brir[0], mode="full")[:num_samples].astype(np.float32)
        right = fftconvolve(speech, brir[1], mode="full")[:num_samples].astype(np.float32)
        stereo = np.stack([left, right], axis=1)
        in_rms = rms(speech)
        out_rms = rms(stereo)
        if out_rms > 1e-9:
            stereo *= in_rms / out_rms
        stereo = peak_normalize(stereo)
    elif rendering_mode == "hybrid_pathwise_hrtf_brir_v3":
        brir, brir_report = synthesize_hybrid_pathwise_hrtf_brir_v3(
            subject=subject,
            sample_rate=sample_rate,
            dims=dims,
            rt60=rt60,
            room_profile=profile,
            head_center=head_center,
            source_xyz=source_xyz,
            max_order=brir_max_order,
            brir_seconds=brir_seconds,
        )
        left = fftconvolve(speech, brir[0], mode="full")[:num_samples].astype(np.float32)
        right = fftconvolve(speech, brir[1], mode="full")[:num_samples].astype(np.float32)
        stereo = np.stack([left, right], axis=1)
        in_rms = rms(speech)
        out_rms = rms(stereo)
        if out_rms > 1e-9:
            stereo *= in_rms / out_rms
        stereo = peak_normalize(stereo)
    else:
        raise ValueError(f"Unsupported rendering_mode: {rendering_mode}")

    clean_reverb = stereo.copy()
    waveform_late_join_metric = None
    waveform_late_join_ok = None
    if rendering_mode == "hybrid_pathwise_hrtf_brir_v3":
        late_start_sample = brir_report.get("late_start_sample")
        if late_start_sample is not None:
            waveform_late_join_metric = stereo_join_metric(clean_reverb, int(late_start_sample), sample_rate, window_ms=10.0)
            waveform_late_join_ok = bool(waveform_late_join_metric < 0.35)

    if no_noise:
        scene = "none"
        left_noise_path = Path("none")
        right_noise_path = Path("none")
        snr_db = 999.0
        mixed = peak_normalize(stereo)
    else:
        scene = rng.choice(list(scenes))
        left_noise_path = rng.choice(noise_files[scene])
        right_noise_path = rng.choice(noise_files[scene])
        snr_db = float(fixed_snr_db) if fixed_snr_db is not None else rng.uniform(-10.0, 10.0)
        noise_l = read_noise_segment(left_noise_path, num_samples, sample_rate, rng)
        noise_r = read_noise_segment(right_noise_path, num_samples, sample_rate, rng)
        mixed_l = mix_at_snr(stereo[:, 0], noise_l, snr_db)
        mixed_r = mix_at_snr(stereo[:, 1], noise_r, snr_db)
        mixed = peak_normalize(np.stack([mixed_l, mixed_r], axis=1))

    rendered_az = brir_report.get("direct_rendered_azimuth_deg")
    if rendered_az is None:
        rendered_az = az_deg
    rendered_el = brir_report.get("direct_rendered_elevation_deg")
    if rendered_el is None:
        rendered_el = el_deg
    rendered_label = angle_to_label(float(rendered_az), 72)

    report = {
        "rendering_mode": rendering_mode,
        "azimuth_deg": az_deg,
        "target_azimuth_deg": target_az,
        "rendered_azimuth_deg": rendered_az,
        "doa_class": target_bin,
        "target_label": int(target_bin),
        "rendered_label": int(rendered_label),
        "cipic_selected_azimuth": float(rendered_az),
        "azimuth_bin": target_bin,
        "elevation_deg": el_deg,
        "target_elevation_deg": 0.0,
        "rendered_elevation_deg": rendered_el,
        "radius": radius,
        "room_profile": profile,
        "room_dims_m": f"{dims[0]:.2f}x{dims[1]:.2f}x{dims[2]:.2f}",
        "rt60_s": rt60,
        "target_rt60": rt60,
        "estimated_rt60": brir_report.get("estimated_rt60"),
        "head_center_xyz": f"{head_center[0]:.3f},{head_center[1]:.3f},{head_center[2]:.3f}",
        "source_xyz": f"{source_xyz[0]:.3f},{source_xyz[1]:.3f},{source_xyz[2]:.3f}",
        "source_distance_m": source_distance,
        "room_source_azimuth_deg": az_deg,
        "source_azimuth_true": float(target_az),
        "source_azimuth_rendered": float(rendered_az),
        "demand_scene": scene,
        "noise_ch_left": left_noise_path.name,
        "noise_ch_right": right_noise_path.name,
        "noise_id": f"{scene}:{left_noise_path.name}|{right_noise_path.name}",
        "snr_db": snr_db,
        "clean_brir": brir_report.get("clean_brir"),
        "clean_reverb_waveform": clean_reverb,
        "waveform_late_join_metric": waveform_late_join_metric,
        "waveform_late_join_ok": waveform_late_join_ok,
        **brir_report,
    }
    return mixed, report


def ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_output_dir(path: Path, overwrite: bool, resume: bool) -> None:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if resume:
        path.mkdir(parents=True, exist_ok=True)
        return
    ensure_empty_dir(path, overwrite)


def rt60_close_to_target_ok(report: Dict[str, object]) -> bool:
    target_rt60 = float(report["target_rt60"])
    estimated_rt60 = float(report["estimated_rt60"])
    return abs(estimated_rt60 - target_rt60) <= max(0.08, 0.20 * target_rt60)


def rt60_within_profile_ok(report: Dict[str, object]) -> bool:
    estimated_rt60 = float(report["estimated_rt60"])
    lo, hi = room_profile_rt60_range(str(report["room_profile"]))
    return lo <= estimated_rt60 <= hi


def hybrid_gate2_ok(report: Dict[str, object]) -> bool:
    if report.get("rendering_mode") != "hybrid_pathwise_hrtf_brir_v3":
        return True
    waveform_join_ok = bool(report.get("waveform_late_join_ok", report.get("late_join_ok", False)))
    return bool(
        waveform_join_ok
        and rt60_close_to_target_ok(report)
        and rt60_within_profile_ok(report)
    )


def write_manifest(output_root: Path, args: argparse.Namespace, split: SubjectSplit) -> None:
    manifest = {
        "dataset": "librispeech_cipic_multisubject_robust50h_v1",
        "speech_root": str(args.librispeech_root),
        "hrtf_root": str(args.hrtf_root),
        "demand_root": str(args.demand_root),
        "sample_rate": args.sample_rate,
        "duration_sec": args.duration_sec,
        "num_classes": args.num_classes,
        "snr_db": [-10.0, 10.0],
        "rendering_mode": args.rendering_mode,
        "brir_max_order": args.brir_max_order,
        "brir_auto_order": args.brir_auto_order,
        "brir_max_auto_order_cap": args.brir_max_auto_order_cap,
        "brir_seconds": args.brir_seconds,
        "max_attempts_per_sample": args.max_attempts_per_sample,
        "total_subjects": args.total_subjects,
        "recordings_per_subject": args.recordings_per_subject,
        "split": {
            "train_subjects": split.train,
            "val_subjects": split.val,
            "test_subjects_unseen": split.test,
        },
        "counts": {
            "train_recordings": len(split.train) * args.recordings_per_subject,
            "val_recordings": len(split.val) * args.recordings_per_subject,
            "test_recordings": len(split.test) * args.recordings_per_subject,
        },
        "invariant": "metadata_azimuth == HRTF_azimuth == room_source_azimuth",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def generate_split(
    split_name: str,
    split_dir_name: str,
    subject_ids: Sequence[str],
    args: argparse.Namespace,
    speech_files: Sequence[Path],
    noise_files: Dict[str, List[Path]],
    hrtf_cache: Dict[str, HRTFSubject],
    py_rng: random.Random,
    np_rng: np.random.Generator,
) -> None:
    split_root = args.output_root / split_dir_name
    wav_dir = split_root / "binaural_dev"
    meta_dir = split_root / "metadata_dev"
    wav_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    report_path = split_root / "mixing_report.csv"
    fieldnames = [
        "file_id",
        "split",
        "rendering_mode",
        "subject_id",
        "speech_path",
        "sofa_path",
        "target_azimuth_deg",
        "rendered_azimuth_deg",
        "doa_class",
        "target_label",
        "rendered_label",
        "cipic_selected_azimuth",
        "azimuth_deg",
        "azimuth_bin",
        "elevation_deg",
        "target_elevation_deg",
        "rendered_elevation_deg",
        "room_profile",
        "room_dims_m",
        "rt60_s",
        "target_rt60",
        "estimated_rt60",
        "head_center_xyz",
        "source_xyz",
        "source_distance_m",
        "room_source_azimuth_deg",
        "brir_method",
        "max_order",
        "sabine_max_order",
        "num_paths",
        "early_path_count",
        "brir_seconds",
        "absorption",
        "reflection_beta",
        "direct_delay_samples",
        "late_start_sample",
        "early_cut_ms",
        "late_start_ms",
        "late_tail_type",
        "late_join_ok",
        "late_join_metric",
        "waveform_late_join_ok",
        "waveform_late_join_metric",
        "late_anchor_window_ms",
        "late_anchor_energy",
        "rt60_close_to_target_ok",
        "rt60_within_profile_ok",
        "quality_gate_ok",
        "target_drr_db",
        "estimated_drr_db",
        "estimated_early_late_ratio_db",
        "left_right_corrcoef",
        "direct_energy_db",
        "early_energy_db",
        "late_energy_db",
        "early_last_delay_ms",
        "low_band_corr",
        "mid_band_corr",
        "high_band_corr",
        "demand_scene",
        "noise_ch_left",
        "noise_ch_right",
        "noise_id",
        "snr_db",
        "sample_rate",
        "duration_sec",
        "num_samples",
    ]

    existing_report_ids = set()
    if args.resume and report_path.is_file():
        with report_path.open(newline="", encoding="utf-8") as f:
            existing_report_ids = {row["file_id"] for row in csv.DictReader(f)}

    report_mode = "a" if args.resume and report_path.is_file() else "w"
    num_samples = int(round(args.duration_sec * args.sample_rate))
    with report_path.open(report_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if report_mode == "w":
            writer.writeheader()

        for subject_id in subject_ids:
            subject = hrtf_cache[subject_id]
            bin_schedule = make_balanced_shuffled_bins(args.recordings_per_subject, args.num_classes, np_rng)
            for local_idx in range(1, args.recordings_per_subject + 1):
                global_idx = local_idx
                file_id = f"{subject_id}_{global_idx:06d}"
                wav_path = wav_dir / f"binaural{file_id}.wav"
                meta_path = meta_dir / f"metadata{file_id}.csv"
                if args.resume and wav_path.is_file() and meta_path.is_file() and file_id in existing_report_ids:
                    continue

                target_bin = int(bin_schedule[local_idx - 1])
                speech_path = speech_files[int(np_rng.integers(0, len(speech_files)))]
                speech = load_mono_resampled(speech_path, args.sample_rate)
                speech = fit_to_length(speech, num_samples, np_rng)

                mixed = None
                report = None
                for _attempt in range(args.max_attempts_per_sample):
                    mixed, report = render_sample(
                        speech=speech,
                        subject=subject,
                        target_bin=target_bin,
                        noise_files=noise_files,
                        scenes=args.scenes,
                        sample_rate=args.sample_rate,
                        num_samples=num_samples,
                        source_distance_min=args.source_distance_min,
                        source_distance_max=args.source_distance_max,
                        rendering_mode=args.rendering_mode,
                        brir_max_order=args.brir_max_order,
                        brir_seconds=args.brir_seconds,
                        rng=py_rng,
                        np_rng=np_rng,
                        no_noise=args.no_noise,
                        fixed_snr_db=args.fixed_snr_db,
                        brir_auto_order=args.brir_auto_order,
                        brir_max_auto_order_cap=args.brir_max_auto_order_cap,
                    )
                    if hybrid_gate2_ok(report):
                        break
                if mixed is None or report is None:
                    raise RuntimeError(f"Failed to render sample {file_id}")

                sf.write(str(wav_path), mixed, args.sample_rate, subtype="PCM_16")
                write_metadata_csv(
                    meta_path,
                    az_deg=float(report["azimuth_deg"]),
                    el_deg=float(report["elevation_deg"]),
                    radius=float(report["radius"]),
                )

                metadata_az = read_metadata_azimuth(meta_path)
                if angular_error_deg(metadata_az, float(report["azimuth_deg"])) > 1e-4:
                    raise RuntimeError(f"Metadata azimuth mismatch for {file_id}")

                row = {
                    "file_id": file_id,
                    "split": split_name,
                    "rendering_mode": report["rendering_mode"],
                    "subject_id": subject_id,
                    "speech_path": str(speech_path),
                    "sofa_path": str(subject.sofa_path),
                    "target_azimuth_deg": f"{float(report['target_azimuth_deg']):.6f}",
                    "rendered_azimuth_deg": f"{float(report['rendered_azimuth_deg']):.6f}",
                    "doa_class": int(report["doa_class"]),
                    "target_label": int(report["target_label"]),
                    "rendered_label": int(report["rendered_label"]),
                    "cipic_selected_azimuth": f"{float(report['cipic_selected_azimuth']):.6f}",
                    "azimuth_deg": f"{float(report['azimuth_deg']):.6f}",
                    "azimuth_bin": int(report["azimuth_bin"]),
                    "elevation_deg": f"{float(report['elevation_deg']):.6f}",
                    "target_elevation_deg": f"{float(report['target_elevation_deg']):.6f}",
                    "rendered_elevation_deg": f"{float(report['rendered_elevation_deg']):.6f}",
                    "room_profile": report["room_profile"],
                    "room_dims_m": report["room_dims_m"],
                    "rt60_s": f"{float(report['rt60_s']):.6f}",
                    "target_rt60": f"{float(report['target_rt60']):.6f}",
                    "estimated_rt60": "" if report.get("estimated_rt60") is None else f"{float(report['estimated_rt60']):.6f}",
                    "head_center_xyz": report["head_center_xyz"],
                    "source_xyz": report["source_xyz"],
                    "source_distance_m": f"{float(report['source_distance_m']):.6f}",
                    "room_source_azimuth_deg": f"{float(report['room_source_azimuth_deg']):.6f}",
                    "brir_method": report["brir_method"],
                    "max_order": "" if report.get("max_order") is None else int(report["max_order"]),
                    "sabine_max_order": "" if report.get("sabine_max_order") is None else int(report["sabine_max_order"]),
                    "num_paths": "" if report.get("num_paths") is None else int(report["num_paths"]),
                    "early_path_count": "" if report.get("early_path_count") is None else int(report["early_path_count"]),
                    "brir_seconds": "" if report.get("brir_seconds") is None else f"{float(report['brir_seconds']):.3f}",
                    "absorption": "" if report.get("absorption") is None else f"{float(report['absorption']):.6f}",
                    "reflection_beta": "" if report.get("reflection_beta") is None else f"{float(report['reflection_beta']):.6f}",
                    "direct_delay_samples": "" if report.get("direct_delay_samples") is None else int(report["direct_delay_samples"]),
                    "late_start_sample": "" if report.get("late_start_sample") is None else int(report["late_start_sample"]),
                    "early_cut_ms": "" if report.get("early_cut_ms") is None else f"{float(report['early_cut_ms']):.3f}",
                    "late_start_ms": "" if report.get("late_start_ms") is None else f"{float(report['late_start_ms']):.3f}",
                    "late_tail_type": report.get("late_tail_type", ""),
                    "late_join_ok": "" if report.get("late_join_ok") is None else int(bool(report["late_join_ok"])),
                    "late_join_metric": "" if report.get("late_join_metric") is None else f"{float(report['late_join_metric']):.6f}",
                    "waveform_late_join_ok": "" if report.get("waveform_late_join_ok") is None else int(bool(report["waveform_late_join_ok"])),
                    "waveform_late_join_metric": "" if report.get("waveform_late_join_metric") is None else f"{float(report['waveform_late_join_metric']):.6f}",
                    "late_anchor_window_ms": report.get("late_anchor_window_ms", ""),
                    "late_anchor_energy": "" if report.get("late_anchor_energy") is None else f"{float(report['late_anchor_energy']):.6f}",
                    "rt60_close_to_target_ok": int(rt60_close_to_target_ok(report)) if report["rendering_mode"] == "hybrid_pathwise_hrtf_brir_v3" else "",
                    "rt60_within_profile_ok": int(rt60_within_profile_ok(report)) if report["rendering_mode"] == "hybrid_pathwise_hrtf_brir_v3" else "",
                    "quality_gate_ok": int(hybrid_gate2_ok(report)),
                    "target_drr_db": "" if report.get("target_drr_db") is None or math.isnan(float(report["target_drr_db"])) else f"{float(report['target_drr_db']):.6f}",
                    "estimated_drr_db": "" if report.get("estimated_drr_db") is None or math.isnan(float(report["estimated_drr_db"])) else f"{float(report['estimated_drr_db']):.6f}",
                    "estimated_early_late_ratio_db": "" if report.get("estimated_early_late_ratio_db") is None or math.isnan(float(report["estimated_early_late_ratio_db"])) else f"{float(report['estimated_early_late_ratio_db']):.6f}",
                    "left_right_corrcoef": "" if report.get("left_right_corrcoef") is None or math.isnan(float(report["left_right_corrcoef"])) else f"{float(report['left_right_corrcoef']):.6f}",
                    "direct_energy_db": "" if report.get("direct_energy_db") is None else f"{float(report['direct_energy_db']):.6f}",
                    "early_energy_db": "" if report.get("early_energy_db") is None else f"{float(report['early_energy_db']):.6f}",
                    "late_energy_db": "" if report.get("late_energy_db") is None else f"{float(report['late_energy_db']):.6f}",
                    "early_last_delay_ms": "" if report.get("early_last_delay_ms") is None else f"{float(report['early_last_delay_ms']):.6f}",
                    "low_band_corr": "" if report.get("low_band_corr") is None else f"{float(report['low_band_corr']):.6f}",
                    "mid_band_corr": "" if report.get("mid_band_corr") is None else f"{float(report['mid_band_corr']):.6f}",
                    "high_band_corr": "" if report.get("high_band_corr") is None else f"{float(report['high_band_corr']):.6f}",
                    "demand_scene": report["demand_scene"],
                    "noise_ch_left": report["noise_ch_left"],
                    "noise_ch_right": report["noise_ch_right"],
                    "noise_id": report["noise_id"],
                    "snr_db": f"{float(report['snr_db']):.6f}",
                    "sample_rate": args.sample_rate,
                    "duration_sec": f"{args.duration_sec:.3f}",
                    "num_samples": num_samples,
                }
                writer.writerow(row)

                json_meta = {
                    "file_id": file_id,
                    "rendering_mode": report["rendering_mode"],
                    "target_azimuth": float(report["target_azimuth_deg"]),
                    "rendered_azimuth": float(report["rendered_azimuth_deg"]),
                    "cipic_selected_azimuth": float(report["cipic_selected_azimuth"]),
                    "doa_class": int(report["doa_class"]),
                    "target_label": int(report["target_label"]),
                    "rendered_label": int(report["rendered_label"]),
                    "subject_id": subject_id,
                    "source_distance": float(report["source_distance_m"]),
                    "source_position": report["source_xyz"],
                    "listener_position": report["head_center_xyz"],
                    "room_profile": report["room_profile"],
                    "room_dimensions": report["room_dims_m"],
                    "target_rt60": float(report["target_rt60"]),
                    "estimated_rt60": report.get("estimated_rt60"),
                    "max_order": report.get("max_order"),
                    "num_paths": report.get("num_paths"),
                    "early_path_count": report.get("early_path_count"),
                    "direct_delay_samples": report.get("direct_delay_samples"),
                    "late_start_sample": report.get("late_start_sample"),
                    "early_cut_ms": report.get("early_cut_ms"),
                    "late_start_ms": report.get("late_start_ms"),
                    "target_drr_db": report.get("target_drr_db"),
                    "estimated_drr_db": report.get("estimated_drr_db"),
                    "estimated_early_late_ratio_db": report.get("estimated_early_late_ratio_db"),
                    "left_right_corrcoef": report.get("left_right_corrcoef"),
                    "late_tail_type": report.get("late_tail_type"),
                    "late_join_ok": report.get("late_join_ok"),
                    "late_join_metric": report.get("late_join_metric"),
                    "waveform_late_join_ok": report.get("waveform_late_join_ok"),
                    "waveform_late_join_metric": report.get("waveform_late_join_metric"),
                    "late_anchor_window_ms": report.get("late_anchor_window_ms"),
                    "late_anchor_energy": report.get("late_anchor_energy"),
                    "rt60_close_to_target_ok": rt60_close_to_target_ok(report) if report["rendering_mode"] == "hybrid_pathwise_hrtf_brir_v3" else None,
                    "rt60_within_profile_ok": rt60_within_profile_ok(report) if report["rendering_mode"] == "hybrid_pathwise_hrtf_brir_v3" else None,
                    "quality_gate_ok": hybrid_gate2_ok(report),
                    "early_last_delay_ms": report.get("early_last_delay_ms"),
                    "direct_energy_db": report.get("direct_energy_db"),
                    "early_energy_db": report.get("early_energy_db"),
                    "late_energy_db": report.get("late_energy_db"),
                    "low_band_corr": report.get("low_band_corr"),
                    "mid_band_corr": report.get("mid_band_corr"),
                    "high_band_corr": report.get("high_band_corr"),
                    "noise_id": report["noise_id"],
                    "snr_db": float(report["snr_db"]),
                }
                (meta_dir / f"metadata{file_id}.json").write_text(
                    json.dumps(json_meta, indent=2),
                    encoding="utf-8",
                )

                if args.path_debug_csv is not None and report.get("path_debug"):
                    with args.path_debug_csv.open("a", newline="", encoding="utf-8") as path_f:
                        path_writer = csv.DictWriter(path_f, fieldnames=[
                            "file_id",
                            "split",
                            "subject_id",
                            "target_azimuth_deg",
                            "rendered_azimuth_deg",
                            "path_id",
                            "order",
                            "image_source_x",
                            "image_source_y",
                            "image_source_z",
                            "distance_m",
                            "delay_samples",
                            "gain",
                            "arrival_azimuth_deg",
                            "arrival_elevation_deg",
                            "selected_hrir_index",
                            "selected_hrir_azimuth_deg",
                            "selected_hrir_elevation_deg",
                        ])
                        for path_row in report["path_debug"]:
                            path_writer.writerow({
                                "file_id": file_id,
                                "split": split_name,
                                "subject_id": subject_id,
                                "target_azimuth_deg": f"{float(report['target_azimuth_deg']):.6f}",
                                "rendered_azimuth_deg": f"{float(report['rendered_azimuth_deg']):.6f}",
                                **path_row,
                            })

                done = (subject_ids.index(subject_id) * args.recordings_per_subject) + local_idx
                if done % args.log_interval == 0:
                    total = len(subject_ids) * args.recordings_per_subject
                    print(f"[{split_name}] {done}/{total} generated", flush=True)


def quality_check_root(root: Path) -> Dict[str, object]:
    report_path = root / "mixing_report.csv"
    wav_dir = root / "binaural_dev"
    meta_dir = root / "metadata_dev"
    rows = list(csv.DictReader(report_path.open(newline="", encoding="utf-8")))
    az_diffs = []
    snrs = []
    rt60s = []
    estimated_rt60s = []
    num_paths = []
    subjects = {}
    for row in rows:
        file_id = row["file_id"]
        meta_az = read_metadata_azimuth(meta_dir / f"metadata{file_id}.csv")
        az = float(row["azimuth_deg"])
        room_az = float(row["room_source_azimuth_deg"])
        az_diffs.append(max(angular_error_deg(meta_az, az), angular_error_deg(room_az, az)))
        snrs.append(float(row["snr_db"]))
        rt60s.append(float(row["rt60_s"]))
        if row.get("estimated_rt60"):
            estimated_rt60s.append(float(row["estimated_rt60"]))
        if row.get("num_paths"):
            num_paths.append(int(row["num_paths"]))
        subjects[row["subject_id"]] = subjects.get(row["subject_id"], 0) + 1

    wav_count = len(list(wav_dir.glob("binaural*.wav")))
    meta_count = len(list(meta_dir.glob("metadata*.csv")))
    return {
        "root": str(root),
        "rows": len(rows),
        "wav_count": wav_count,
        "metadata_count": meta_count,
        "subjects": subjects,
        "max_azimuth_mismatch_deg": max(az_diffs) if az_diffs else None,
        "snr_min": min(snrs) if snrs else None,
        "snr_max": max(snrs) if snrs else None,
        "rt60_min": min(rt60s) if rt60s else None,
        "rt60_max": max(rt60s) if rt60s else None,
        "estimated_rt60_min": min(estimated_rt60s) if estimated_rt60s else None,
        "estimated_rt60_max": max(estimated_rt60s) if estimated_rt60s else None,
        "num_paths_min": min(num_paths) if num_paths else None,
        "num_paths_max": max(num_paths) if num_paths else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare robust multisubject binaural DOA dataset")
    parser.add_argument("--librispeech_root", type=Path, default=Path("/disk2/bywang/data/LibriSpeech/train-clean-100"))
    parser.add_argument("--hrtf_root", type=Path, default=Path("/disk2/bywang/data/HRTF"))
    parser.add_argument("--demand_root", type=Path, default=Path("/disk2/bywang/data/demand"))
    parser.add_argument("--output_root", type=Path, default=Path("/disk2/bywang/DOA-net/data/librispeech_cipic_multisubject_robust50h_v1"))
    parser.add_argument("--total_subjects", type=int, default=30)
    parser.add_argument("--recordings_per_subject", type=int, default=600)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration_sec", type=float, default=10.0)
    parser.add_argument("--source_distance_min", type=float, default=1.0)
    parser.add_argument("--source_distance_max", type=float, default=1.5)
    parser.add_argument(
        "--rendering_mode",
        choices=["mono_room_rir_then_hrtf", "image_source_pathwise_hrtf_brir", "hybrid_pathwise_hrtf_brir_v3"],
        default="mono_room_rir_then_hrtf",
    )
    parser.add_argument("--brir_max_order", type=int, default=6)
    parser.add_argument("--brir_auto_order", action="store_true", help="Use Sabine-derived max_order capped by --brir_max_auto_order_cap")
    parser.add_argument("--brir_max_auto_order_cap", type=int, default=8, help="Upper bound for Sabine-derived BRIR image-source order")
    parser.add_argument("--brir_seconds", type=float, default=1.0)
    parser.add_argument("--max_attempts_per_sample", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=72)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_include", nargs="*", default=["003"])
    parser.add_argument("--scenes", nargs="+", default=DEMAND_SCENES)
    parser.add_argument("--no_noise", action="store_true", help="Disable DEMAND noise for BRIR sanity/debug generation")
    parser.add_argument("--fixed_snr_db", type=float, default=None, help="Use a fixed SNR instead of uniform [-10, 10] dB")
    parser.add_argument("--path_debug_csv", type=Path, default=None, help="Optional CSV path for path-wise BRIR diagnostics")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted generation run")
    parser.add_argument("--smoke", action="store_true", help="Generate a tiny dataset for validation")
    parser.add_argument("--log_interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.output_root = args.output_root.with_name(args.output_root.name + "_smoke")
        args.total_subjects = min(args.total_subjects, 10)
        args.recordings_per_subject = min(args.recordings_per_subject, 12)
        args.log_interval = 10

    ensure_output_dir(args.output_root, args.overwrite, args.resume)
    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    all_subjects = list_subjects(args.hrtf_root)
    split = choose_subjects(all_subjects, args.total_subjects, args.seed, args.force_include)
    if not args.resume or not (args.output_root / "manifest.json").is_file():
        write_manifest(args.output_root, args, split)

    print("Selected subjects:", json.dumps({
        "train": split.train,
        "val": split.val,
        "test": split.test,
    }, indent=2), flush=True)

    speech_files = list_librispeech_files(args.librispeech_root)
    noise_files = list_noise_files(args.demand_root, args.scenes)
    needed_subjects = sorted(set(split.train + split.val + split.test))
    hrtf_cache = {
        sid: HRTFSubject(args.hrtf_root / f"subject_{sid}.sofa", args.sample_rate)
        for sid in needed_subjects
    }

    if args.path_debug_csv is not None:
        args.path_debug_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.path_debug_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "file_id",
                "split",
                "subject_id",
                "target_azimuth_deg",
                "rendered_azimuth_deg",
                "path_id",
                "order",
                "image_source_x",
                "image_source_y",
                "image_source_z",
                "distance_m",
                "delay_samples",
                "gain",
                "arrival_azimuth_deg",
                "arrival_elevation_deg",
                "selected_hrir_index",
                "selected_hrir_azimuth_deg",
                "selected_hrir_elevation_deg",
            ])
            writer.writeheader()

    generate_split("train", "train_subjects", split.train, args, speech_files, noise_files, hrtf_cache, py_rng, np_rng)
    generate_split("val", "val_subjects", split.val, args, speech_files, noise_files, hrtf_cache, py_rng, np_rng)
    generate_split("test", "test_subjects_unseen", split.test, args, speech_files, noise_files, hrtf_cache, py_rng, np_rng)

    qc = {
        "train": quality_check_root(args.output_root / "train_subjects"),
        "val": quality_check_root(args.output_root / "val_subjects"),
        "test": quality_check_root(args.output_root / "test_subjects_unseen"),
    }
    (args.output_root / "quality_report.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print("Quality report:", json.dumps(qc, indent=2), flush=True)
    print(f"Done: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
