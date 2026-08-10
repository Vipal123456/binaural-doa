#!/usr/bin/env python3
"""Prepare moving single-speaker binaural DOA sequence datasets.

The first supported condition is clean_moving: dynamic CIPIC HRTF rendering
with 40 chunks of 100 ms for each 4 s sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyroomacoustics as pra
import sofa
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

from prepare_robust_multisubject_dataset import (
    bandwise_corr_summary,
    decompose_brir_energies,
    estimate_early_late_ratio_db,
    generate_binaural_diffuse_late_tail,
    room_profile_rt60_range,
)


DEMAND_SCENES = ["OOFFICE", "PCAFETER", "TMETRO", "TBUS", "SPSQUARE", "NPARK"]


def wrap_deg(angle: float) -> float:
    return ((float(angle) + 180.0) % 360.0) - 180.0


def angular_error(a: float, b: float) -> float:
    return abs(wrap_deg(a - b))


def angle_to_label(angle: float, num_classes: int = 72) -> int:
    wrapped = wrap_deg(angle)
    return int(np.clip(math.floor((wrapped + 180.0) / (360.0 / num_classes)), 0, num_classes - 1))


def list_subjects(hrtf_root: Path) -> List[str]:
    subjects = sorted(p.stem.replace("subject_", "") for p in hrtf_root.glob("subject_*.sofa"))
    if not subjects:
        raise FileNotFoundError(f"No subject_*.sofa files found in {hrtf_root}")
    return subjects


def split_subjects(subjects: Sequence[str], total: int, seed: int) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    selected = list(subjects)
    rng.shuffle(selected)
    selected = sorted(selected[:total])
    n_train = max(1, int(round(total * 0.8)))
    n_val = max(1, int(round(total * 0.1)))
    return {
        "train_subjects": selected[:n_train],
        "val_subjects": selected[n_train:n_train + n_val],
        "test_subjects_unseen": selected[n_train + n_val:],
    }


def parse_subject_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_mono(path: Path, sample_rate: int) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        g = math.gcd(int(sr), int(sample_rate))
        audio = resample_poly(audio, sample_rate // g, sr // g).astype(np.float32)
    return np.asarray(audio, dtype=np.float32)


def fit_length(audio: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    if len(audio) >= length:
        start = int(rng.integers(0, len(audio) - length + 1))
        return audio[start:start + length].astype(np.float32, copy=False)
    reps = int(np.ceil(length / max(1, len(audio))))
    return np.tile(audio, reps)[:length].astype(np.float32, copy=False)


def peak_normalize(stereo: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(stereo)))
    if peak > 0.95 and peak > 1e-8:
        stereo = stereo * (0.95 / peak)
    return stereo.astype(np.float32, copy=False)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def list_noise_files(demand_root: Path, scenes: Sequence[str]) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for scene in scenes:
        scene_dir = demand_root / scene
        wavs = sorted(scene_dir.glob("ch*.wav"))
        if not wavs:
            raise FileNotFoundError(f"No ch*.wav noise files found in {scene_dir}")
        out[scene] = wavs
    return out


def read_noise_segment(
    path: Path,
    length: int,
    target_sr: int,
    rng: random.Random,
    start: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    info = sf.info(str(path))
    read_start = 0
    if info.frames >= length and info.samplerate == target_sr:
        max_start = max(0, info.frames - length)
        read_start = int(start) if start is not None else rng.randint(0, max_start)
        read_start = max(0, min(read_start, max_start))
        noise, sr = sf.read(str(path), start=read_start, frames=length, dtype="float32", always_2d=False)
    else:
        noise, sr = sf.read(str(path), dtype="float32", always_2d=False)

    noise = np.asarray(noise, dtype=np.float32)
    if noise.ndim > 1:
        noise = noise[:, 0]
    if sr != target_sr:
        g = math.gcd(int(sr), int(target_sr))
        noise = resample_poly(noise, target_sr // g, sr // g).astype(np.float32)
    if len(noise) < length:
        reps = int(np.ceil(length / max(1, len(noise))))
        noise = np.tile(noise, reps)
    if len(noise) > length:
        if start is None or info.samplerate != target_sr:
            read_start = rng.randint(0, max(0, len(noise) - length))
        noise = noise[read_start:read_start + length]
    return noise.astype(np.float32, copy=False), int(read_start)


def decorrelate_noise(noise: np.ndarray, sample_rate: int, rng: random.Random) -> np.ndarray:
    shift = rng.randint(max(1, sample_rate // 1000), max(2, sample_rate // 200))
    if rng.random() < 0.5:
        shift = -shift
    shifted = np.roll(noise, shift).astype(np.float32, copy=False)
    # A tiny FIR coloration avoids identical left/right noise while preserving the same scene/segment.
    return (0.92 * shifted + 0.08 * np.roll(shifted, 1)).astype(np.float32, copy=False)


def mix_stereo_at_snr(stereo: np.ndarray, noise_l: np.ndarray, noise_r: np.ndarray, snr_db: float) -> np.ndarray:
    signal = stereo.astype(np.float32, copy=False)
    noise = np.stack([noise_l, noise_r], axis=1).astype(np.float32, copy=False)
    noise = noise - np.mean(noise, axis=0, keepdims=True)
    sig_power = float(np.mean(signal.astype(np.float64) ** 2))
    noise_power = float(np.mean(noise.astype(np.float64) ** 2))
    if sig_power < 1e-12 or noise_power < 1e-12:
        return signal.copy()
    scale = math.sqrt(sig_power / ((10.0 ** (snr_db / 10.0)) * noise_power))
    return (signal + scale * noise).astype(np.float32, copy=False)


def choose_binaural_noise(
    noise_files: Dict[str, List[Path]],
    length: int,
    sample_rate: int,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    scene = rng.choice(sorted(noise_files))
    files = noise_files[scene]
    left_path = rng.choice(files)
    left, start = read_noise_segment(left_path, length, sample_rate, rng)
    if len(files) > 1:
        candidates = [p for p in files if p != left_path]
        right_path = rng.choice(candidates)
        right, right_start = read_noise_segment(right_path, length, sample_rate, rng, start=start)
        mode = "same_scene_different_channel"
    else:
        right_path = left_path
        right = decorrelate_noise(left, sample_rate, rng)
        right_start = start
        mode = "same_scene_same_segment_decorrelated"
    return left, right, {
        "noise_scene": scene,
        "noise_ch_left": left_path.name,
        "noise_ch_right": right_path.name,
        "noise_start_left": int(start),
        "noise_start_right": int(right_start),
        "noise_mode": mode,
        "noise_id": f"{scene}:{left_path.name}@{start}|{right_path.name}@{right_start}",
    }


def room_profile_params(
    profile: str,
    rng: random.Random,
    rt60_min: Optional[float] = None,
    rt60_max: Optional[float] = None,
) -> Tuple[Tuple[float, float, float], float]:
    if profile == "small":
        dims = (rng.uniform(3.5, 5.0), rng.uniform(3.0, 4.5), rng.uniform(2.5, 3.0))
        default_rt60 = (0.20, 0.45)
    elif profile == "medium":
        dims = (rng.uniform(5.0, 7.5), rng.uniform(4.0, 6.5), rng.uniform(2.7, 3.2))
        default_rt60 = (0.35, 0.65)
    elif profile == "large":
        dims = (rng.uniform(7.5, 10.0), rng.uniform(6.0, 8.5), rng.uniform(3.0, 3.8))
        default_rt60 = (0.50, 0.80)
    else:
        raise ValueError(f"Unknown room profile: {profile}")
    lo = default_rt60[0] if rt60_min is None else float(rt60_min)
    hi = default_rt60[1] if rt60_max is None else float(rt60_max)
    return dims, rng.uniform(lo, hi)


def choose_room_geometry(
    dims: Tuple[float, float, float],
    az_deg: float,
    distance: float,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray]:
    lx, ly, lz = dims
    height = min(1.5, lz - 0.7)
    margin = 0.55
    az = math.radians(az_deg)
    for _ in range(200):
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
            return head, source
    head = np.array([lx / 2.0, ly / 2.0, height], dtype=np.float64)
    source = head + np.array([distance * math.sin(az), distance * math.cos(az), 0.0])
    source[0] = np.clip(source[0], margin, lx - margin)
    source[1] = np.clip(source[1], margin, ly - margin)
    return head, source


def source_position(head_center: np.ndarray, az_deg: float, distance: float) -> np.ndarray:
    az = math.radians(float(az_deg))
    return head_center + np.array([
        distance * math.sin(az),
        distance * math.cos(az),
        0.0,
    ], dtype=np.float64)


def choose_moving_room_geometry(
    dims: Tuple[float, float, float],
    angle_seq: np.ndarray,
    distance: float,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray]:
    lx, ly, lz = dims
    height = min(1.5, lz - 0.7)
    margin = 0.55
    for _ in range(500):
        head = np.array([
            rng.uniform(1.0, lx - 1.0),
            rng.uniform(1.0, ly - 1.0),
            height,
        ], dtype=np.float64)
        positions = np.stack([source_position(head, a, distance) for a in angle_seq], axis=0)
        if (
            np.all((positions[:, 0] >= margin) & (positions[:, 0] <= lx - margin))
            and np.all((positions[:, 1] >= margin) & (positions[:, 1] <= ly - margin))
            and np.all((positions[:, 2] >= margin) & (positions[:, 2] <= lz - margin))
        ):
            return head, positions
    raise RuntimeError("Could not place moving trajectory inside room without clipping")


def synthesize_mono_room_rir(sample_rate: int, dims: Tuple[float, float, float], rt60: float, head_center: np.ndarray, source_xyz: np.ndarray) -> np.ndarray:
    absorption, max_order = pra.inverse_sabine(rt60, dims)
    max_order = int(max(1, min(max_order, 12)))
    room = pra.ShoeBox(
        dims,
        fs=sample_rate,
        materials=pra.Material(absorption),
        max_order=max_order,
        ray_tracing=False,
        air_absorption=True,
    )
    room.add_source(source_xyz)
    room.add_microphone_array(head_center.reshape(3, 1))
    room.compute_rir()
    rir = np.asarray(room.rir[0][0], dtype=np.float32)
    if len(rir) == 0:
        rir = np.array([1.0], dtype=np.float32)
    return rir / max(float(np.max(np.abs(rir))), 1e-8)


def apply_reverb_to_mono(speech: np.ndarray, rir: np.ndarray, num_samples: int) -> np.ndarray:
    reverbed = fftconvolve(speech, rir, mode="full")[:num_samples].astype(np.float32)
    in_rms = rms(speech)
    out_rms = rms(reverbed)
    if out_rms > 1e-9:
        reverbed *= in_rms / out_rms
    return reverbed.astype(np.float32, copy=False)


def image_source_axis_positions(source_coord: float, room_len: float, max_order: int) -> List[Tuple[float, int]]:
    out = []
    for n in range(-max_order, max_order + 1):
        for q in (0, 1):
            coord = 2.0 * n * room_len + ((-1.0) ** q) * source_coord
            order = abs(2 * n + q)
            if order <= max_order:
                out.append((coord, order))
    return out


def estimate_rt60_from_ir(ir: np.ndarray, sample_rate: int) -> float:
    if len(ir) < 8:
        return 0.0
    energy = np.cumsum(ir[::-1].astype(np.float64) ** 2)[::-1]
    if energy[0] <= 1e-12:
        return 0.0
    edc = 10.0 * np.log10(np.maximum(energy / energy[0], 1e-12))
    t = np.arange(len(edc), dtype=np.float64) / float(sample_rate)
    mask = (edc <= -5.0) & (edc >= -35.0)
    if mask.sum() < 8:
        return 0.0
    slope, _ = np.polyfit(t[mask], edc[mask], 1)
    if slope >= 0:
        return 0.0
    return float(-60.0 / slope)


def rt60_close_to_target_ok(target_rt60: float, estimated_rt60: float) -> bool:
    return abs(float(estimated_rt60) - float(target_rt60)) <= max(0.08, 0.20 * float(target_rt60))


def rt60_within_profile_ok(room_profile: str, estimated_rt60: float) -> bool:
    lo, hi = room_profile_rt60_range(str(room_profile))
    return lo <= float(estimated_rt60) <= hi


def hybrid_gate2_ok(report: Dict[str, object]) -> bool:
    return bool(
        report.get("waveform_late_join_ok", report.get("late_join_ok", False))
        and rt60_close_to_target_ok(float(report.get("target_rt60", 0.0)), float(report.get("estimated_rt60", 0.0)))
        and rt60_within_profile_ok(str(report.get("room_profile", "small")), float(report.get("estimated_rt60", 0.0)))
    )


def robust_late_join_stats(
    full: np.ndarray,
    late_start_sample: int,
    sample_rate: int,
    threshold: float = 1.0,
) -> Tuple[bool, float, float, float]:
    win = max(1, int(round(0.010 * sample_rate)))
    pre = full[:, max(0, late_start_sample - win):late_start_sample]
    post = full[:, late_start_sample:min(full.shape[1], late_start_sample + win)]
    pre_rms = rms(pre) if pre.size else 0.0
    post_rms = rms(post) if post.size else 0.0
    floor = 0.01 * max(rms(full), pre_rms, post_rms, 1e-8)
    metric = abs(pre_rms - post_rms) / max(pre_rms, post_rms, floor)
    return bool(metric < threshold), float(metric), float(pre_rms), float(post_rms)


def synthesize_pathwise_hrtf_brir(
    subject: HRTFSubject,
    sample_rate: int,
    dims: Tuple[float, float, float],
    rt60: float,
    head_center: np.ndarray,
    source_xyz: np.ndarray,
    max_order: int,
    brir_seconds: float,
    chunk_id: int = 0,
    keep_path_debug: bool = False,
) -> Tuple[np.ndarray, Dict[str, object]]:
    c = 343.0
    absorption, sabine_order = pra.inverse_sabine(rt60, dims)
    max_order = int(max_order)
    beta = math.sqrt(max(0.0, 1.0 - float(absorption)))
    brir_len = int(round(brir_seconds * sample_rate))
    brir_l = np.zeros(brir_len, dtype=np.float32)
    brir_r = np.zeros(brir_len, dtype=np.float32)

    xs = image_source_axis_positions(float(source_xyz[0]), dims[0], max_order)
    ys = image_source_axis_positions(float(source_xyz[1]), dims[1], max_order)
    zs = image_source_axis_positions(float(source_xyz[2]), dims[2], max_order)

    num_paths = 0
    direct_report: Dict[str, object] = {}
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
                az = wrap_deg(math.degrees(math.atan2(vec[0], vec[1])))
                el = math.degrees(math.asin(np.clip(vec[2] / dist, -1.0, 1.0)))
                hrir_idx, rendered_az, rendered_el, _ = subject.pick_direction(az, el)
                h_l = subject.ir[hrir_idx, 0]
                h_r = subject.ir[hrir_idx, 1]
                gain = (beta ** order) / max(dist, 1e-6)
                end = min(brir_len, delay + len(h_l))
                n = end - delay
                if n <= 0:
                    continue
                brir_l[delay:end] += (gain * h_l[:n]).astype(np.float32)
                brir_r[delay:end] += (gain * h_r[:n]).astype(np.float32)
                is_direct = order == 0
                if is_direct:
                    direct_report = {
                        "direct_arrival_azimuth_deg": float(az),
                        "direct_arrival_elevation_deg": float(el),
                        "direct_rendered_azimuth_deg": float(rendered_az),
                        "direct_rendered_elevation_deg": float(rendered_el),
                        "direct_selected_hrir_index": int(hrir_idx),
                        "direct_path_distance_m": float(dist),
                        "direct_path_delay_samples": int(delay),
                    }
                if keep_path_debug:
                    path_debug.append({
                        "chunk_id": int(chunk_id),
                        "path_id": int(num_paths),
                        "order": int(order),
                        "is_direct": bool(is_direct),
                        "image_source_x": float(image[0]),
                        "image_source_y": float(image[1]),
                        "image_source_z": float(image[2]),
                        "distance_m": float(dist),
                        "delay_samples": int(delay),
                        "gain": float(gain),
                        "arrival_azimuth_deg": float(az),
                        "arrival_elevation_deg": float(el),
                        "selected_hrir_index": int(hrir_idx),
                        "selected_hrir_azimuth_deg": float(rendered_az),
                        "selected_hrir_elevation_deg": float(rendered_el),
                    })
                num_paths += 1
    peak = max(float(np.max(np.abs(brir_l))), float(np.max(np.abs(brir_r))), 1e-8)
    brir = np.stack([brir_l / peak, brir_r / peak], axis=0).astype(np.float32)
    report = {
        "num_paths": int(num_paths),
        "brir_seconds": float(brir_seconds),
        "brir_max_order": int(max_order),
        "sabine_max_order": int(sabine_order),
        "absorption": float(absorption),
        "reflection_beta": float(beta),
        "estimated_rt60": estimate_rt60_from_ir(0.5 * (brir[0] + brir[1]), sample_rate),
        "path_debug": path_debug,
        **direct_report,
    }
    return brir, report


def stable_late_seed(
    subject_id: str,
    room_profile: str,
    rt60: float,
    head_center: np.ndarray,
    seed: int,
) -> int:
    payload = (
        str(subject_id),
        str(room_profile),
        round(float(rt60), 4),
        tuple(float(x) for x in np.round(head_center, 3)),
        int(seed),
    )
    return int(abs(hash(payload)) % (2**32))


def synthesize_hybrid_pathwise_hrtf_brir_chunk(
    subject: HRTFSubject,
    sample_rate: int,
    dims: Tuple[float, float, float],
    rt60: float,
    room_profile: str,
    head_center: np.ndarray,
    source_xyz: np.ndarray,
    max_order: int,
    brir_seconds: float,
    late_seed: int,
    chunk_id: int = 0,
    early_cut_ms: float = 80.0,
    late_start_ms: float = 80.0,
    keep_path_debug: bool = False,
) -> Tuple[np.ndarray, Dict[str, object]]:
    c = 343.0
    absorption, sabine_order = pra.inverse_sabine(rt60, dims)
    beta = math.sqrt(max(0.0, 1.0 - float(absorption)))
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

    direct_report: Dict[str, object] = {}
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
                az = wrap_deg(math.degrees(math.atan2(vec[0], vec[1])))
                el = math.degrees(math.asin(np.clip(vec[2] / dist, -1.0, 1.0)))
                hrir_idx, rendered_az, rendered_el, _ = subject.pick_direction(az, el)
                h_l = subject.ir[hrir_idx, 0]
                h_r = subject.ir[hrir_idx, 1]
                gain = (beta ** order) / max(dist, 1e-6)
                end = min(brir_len, delay + len(h_l))
                n = end - delay
                if n <= 0:
                    continue
                early_l[delay:end] += (gain * h_l[:n]).astype(np.float32)
                early_r[delay:end] += (gain * h_r[:n]).astype(np.float32)
                is_direct = order == 0
                if is_direct:
                    direct_report = {
                        "direct_arrival_azimuth_deg": float(az),
                        "direct_arrival_elevation_deg": float(el),
                        "direct_rendered_azimuth_deg": float(rendered_az),
                        "direct_rendered_elevation_deg": float(rendered_el),
                        "direct_selected_hrir_index": int(hrir_idx),
                        "direct_path_distance_m": float(dist),
                        "direct_path_delay_samples": int(delay),
                    }
                if keep_path_debug:
                    path_debug.append({
                        "chunk_id": int(chunk_id),
                        "path_id": int(early_path_count),
                        "order": int(order),
                        "is_direct": bool(is_direct),
                        "image_source_x": float(image[0]),
                        "image_source_y": float(image[1]),
                        "image_source_z": float(image[2]),
                        "distance_m": float(dist),
                        "delay_samples": int(delay),
                        "gain": float(gain),
                        "arrival_azimuth_deg": float(az),
                        "arrival_elevation_deg": float(el),
                        "selected_hrir_index": int(hrir_idx),
                        "selected_hrir_azimuth_deg": float(rendered_az),
                        "selected_hrir_elevation_deg": float(rendered_el),
                    })
                early_path_count += 1

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

    late = generate_binaural_diffuse_late_tail(late_duration_samples, sample_rate, rt60, int(late_seed))
    fade_len = min(int(round(0.020 * sample_rate)), late.shape[1])
    if fade_len > 1:
        late[:, :fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)[None, :]
    initial_rms = rms(0.5 * (late[0, : max(fade_len, 1)] + late[1, : max(fade_len, 1)]))
    if anchor_rms > 1e-8 and initial_rms > 1e-8:
        late *= anchor_rms / initial_rms

    full = early.copy()
    late_end = min(brir_len, late_start_sample + late.shape[1])
    late_n = late_end - late_start_sample
    if late_n > 0:
        full[:, late_start_sample:late_end] += late[:, :late_n]

    late_join_ok, late_join_metric, pre_rms, post_rms = robust_late_join_stats(
        full,
        late_start_sample,
        sample_rate,
        threshold=1.0,
    )

    peak = max(float(np.max(np.abs(full))), 1e-8)
    brir = (full / peak).astype(np.float32, copy=False)
    mono_proxy = 0.5 * (brir[0] + brir[1])
    energy_stats = decompose_brir_energies(brir, direct_delay, late_start_sample, sample_rate)
    estimated_drr = (
        float(10.0 * np.log10(max(energy_stats["direct_energy"], 1e-12) / max(energy_stats["reverberant_energy"], 1e-12)))
        if energy_stats["reverberant_energy"] > 1e-12 else float("nan")
    )
    band_corr = bandwise_corr_summary(brir[:, late_start_sample:], sample_rate)
    early_delays = [int(row["delay_samples"]) for row in path_debug]
    early_last_delay_ms = (max(early_delays) / float(sample_rate) * 1000.0) if early_delays else 0.0
    report = {
        "brir_method": "moving_hybrid_pathwise_hrtf_brir_gate2_v1",
        "brir_max_order": int(max_order),
        "sabine_max_order": int(sabine_order),
        "absorption": float(absorption),
        "reflection_beta": float(beta),
        "num_paths": int(early_path_count),
        "early_path_count": int(early_path_count),
        "early_cut_ms": float(early_cut_ms),
        "late_start_ms": float(late_start_ms),
        "late_start_sample": int(late_start_sample),
        "late_tail_type": "binaural_diffuse_statistical",
        "late_seed": int(late_seed),
        "late_seed_scope": "per_trajectory",
        "brir_seconds": float(brir_len / sample_rate),
        "estimated_rt60": estimate_rt60_from_ir(mono_proxy, sample_rate),
        "target_rt60_range_lo": float(room_profile_rt60_range(room_profile)[0]),
        "target_rt60_range_hi": float(room_profile_rt60_range(room_profile)[1]),
        "late_join_ok": late_join_ok,
        "late_join_metric": float(late_join_metric),
        "late_join_pre_rms": float(pre_rms),
        "late_join_post_rms": float(post_rms),
        "late_anchor_window_ms": anchor_window_ms,
        "late_anchor_energy": float(anchor_energy),
        "estimated_drr_db": estimated_drr,
        "target_drr_db": estimated_drr,
        "estimated_early_late_ratio_db": estimate_early_late_ratio_db(brir, late_start_sample),
        "early_last_delay_ms": float(early_last_delay_ms),
        "left_right_corrcoef": float(np.corrcoef(brir[0], brir[1])[0, 1]) if brir.shape[1] > 1 else float("nan"),
        **energy_stats,
        **band_corr,
        **direct_report,
        "path_debug": path_debug,
    }
    return brir, report


class HRTFSubject:
    def __init__(self, sofa_path: Path, sample_rate: int):
        self.sofa_path = sofa_path
        self.subject_id = sofa_path.stem.replace("subject_", "")
        db = sofa.Database.open(str(sofa_path))
        positions = np.asarray(db.Source.Position.get_values(), dtype=np.float64)
        ir = np.asarray(db.Data.IR.get_values(), dtype=np.float32)
        sofa_sr = int(round(float(db.Data.SamplingRate.get_values()[0])))
        if sofa_sr != sample_rate:
            g = math.gcd(sofa_sr, sample_rate)
            ir = resample_poly(ir, sample_rate // g, sofa_sr // g, axis=-1).astype(np.float32)
        self.ir = ir
        self.sofa_azimuths = np.array([wrap_deg(float(a)) for a in positions[:, 0]], dtype=np.float64)
        # Keep the same project convention as the static BRIR generator:
        # 0 deg is front (+y), +90 deg is listener right (+x).  The CIPIC SOFA
        # azimuth sign is opposite for this convention, so lookup/reporting must
        # use the flipped world-convention azimuth while preserving HRIR indices.
        self.azimuths = np.array([wrap_deg(-float(a)) for a in self.sofa_azimuths], dtype=np.float64)
        self.elevations = np.asarray(positions[:, 1], dtype=np.float64)
        self.radii = np.asarray(positions[:, 2], dtype=np.float64)

    def nearest_idx(self, azimuth: float) -> int:
        az_err = np.array([angular_error(a, azimuth) for a in self.azimuths])
        score = az_err + 2.0 * np.abs(self.elevations)
        return int(np.argmin(score))

    def pick_direction(self, azimuth: float, elevation: float = 0.0) -> Tuple[int, float, float, float]:
        az_err = np.array([angular_error(a, azimuth) for a in self.azimuths])
        score = az_err + 2.0 * np.abs(self.elevations - float(elevation))
        idx = int(np.argmin(score))
        return idx, float(self.azimuths[idx]), float(self.elevations[idx]), float(self.radii[idx])

    def nearest_angle(self, azimuth: float) -> float:
        return float(self.azimuths[self.nearest_idx(azimuth)])

    def render_chunk(self, mono: np.ndarray, azimuth: float, out_len: int) -> np.ndarray:
        idx = self.nearest_idx(azimuth)
        h_l = self.ir[idx, 0]
        h_r = self.ir[idx, 1]
        left = fftconvolve(mono, h_l, mode="full")[:out_len]
        right = fftconvolve(mono, h_r, mode="full")[:out_len]
        return np.stack([left, right], axis=1).astype(np.float32)


def generate_trajectory(rng: random.Random, steps: int, kind: str) -> Tuple[np.ndarray, float, str]:
    theta0 = rng.uniform(-180.0, 180.0)
    dt = 0.1
    if kind == "static":
        return np.full(steps, wrap_deg(theta0), dtype=np.float32), 0.0, "static"
    if kind == "linear":
        omega = rng.choice([-1.0, 1.0]) * rng.uniform(10.0, 50.0)
        seq = [wrap_deg(theta0 + omega * i * dt) for i in range(steps)]
        speed_bin = "slow" if abs(omega) < 20.0 else "medium"
        return np.asarray(seq, dtype=np.float32), abs(omega), speed_bin

    num_segments = rng.randint(2, 4)
    breakpoints = sorted(rng.sample(range(8, steps - 4), num_segments - 1))
    breakpoints = [0] + breakpoints + [steps]
    seq = []
    theta = theta0
    speeds = []
    for start, end in zip(breakpoints[:-1], breakpoints[1:]):
        omega = rng.uniform(-60.0, 60.0)
        if rng.random() < 0.25:
            omega = 0.0
        speeds.append(abs(omega))
        for _ in range(start, end):
            seq.append(wrap_deg(theta))
            theta = wrap_deg(theta + omega * dt)
    mean_speed = float(np.mean(speeds))
    speed_bin = "slow" if mean_speed < 20.0 else ("medium" if mean_speed < 60.0 else "fast")
    return np.asarray(seq[:steps], dtype=np.float32), mean_speed, speed_bin


def render_dynamic_clean(
    speech: np.ndarray,
    subject: HRTFSubject,
    angle_seq: np.ndarray,
    sample_rate: int,
    chunk_samples: int,
    crossfade_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    chunks = []
    rendered_angles = []
    for i, angle in enumerate(angle_seq):
        start = i * chunk_samples
        dry = speech[start:start + chunk_samples]
        rendered_angle = subject.nearest_angle(float(angle))
        rendered_angles.append(rendered_angle)
        chunks.append(subject.render_chunk(dry, rendered_angle, chunk_samples))
    out = chunks[0].copy()
    for chunk in chunks[1:]:
        if crossfade_samples > 0:
            fade = np.linspace(0.0, 1.0, crossfade_samples, endpoint=False, dtype=np.float32)[:, None]
            out[-crossfade_samples:] = out[-crossfade_samples:] * (1.0 - fade) + chunk[:crossfade_samples] * fade
            out = np.concatenate([out, chunk[crossfade_samples:]], axis=0)
        else:
            out = np.concatenate([out, chunk], axis=0)
    target_len = len(angle_seq) * chunk_samples
    if len(out) < target_len:
        out = np.pad(out, ((0, target_len - len(out)), (0, 0)), mode="constant")
    return peak_normalize(out[:target_len]), np.asarray(rendered_angles, dtype=np.float32)


def render_dynamic_pathwise_brir(
    speech: np.ndarray,
    subject: HRTFSubject,
    angle_seq: np.ndarray,
    source_positions: np.ndarray,
    sample_rate: int,
    chunk_samples: int,
    dims: Tuple[float, float, float],
    rt60: float,
    room_profile: str,
    head_center: np.ndarray,
    brir_max_order: int,
    brir_seconds: float,
    rendering_mode: str = "image_source_pathwise_dynamic_hrtf_brir",
    late_seed: Optional[int] = None,
    early_cut_ms: float = 80.0,
    late_start_ms: float = 80.0,
    keep_path_debug: bool = False,
) -> Tuple[np.ndarray, Dict[str, object]]:
    brir_len = int(round(brir_seconds * sample_rate))
    out_len = len(angle_seq) * chunk_samples + brir_len
    out = np.zeros((out_len, 2), dtype=np.float32)
    direct_rendered_angles = []
    direct_selected_indices = []
    direct_distances = []
    direct_delays = []
    num_paths_seq = []
    estimated_rt60_seq = []
    estimated_drr_seq = []
    early_late_ratio_seq = []
    late_join_metric_seq = []
    late_join_ok_seq = []
    early_path_count_seq = []
    early_last_delay_ms_seq = []
    low_band_corr_seq = []
    mid_band_corr_seq = []
    high_band_corr_seq = []
    path_debug_rows: List[Dict[str, object]] = []
    for i, source_xyz in enumerate(source_positions):
        start = i * chunk_samples
        dry = speech[start:start + chunk_samples]
        if rendering_mode == "moving_hybrid_pathwise_hrtf_brir_gate2_v1":
            if late_seed is None:
                raise ValueError("late_seed is required for moving hybrid BRIR rendering")
            brir, report = synthesize_hybrid_pathwise_hrtf_brir_chunk(
                subject=subject,
                sample_rate=sample_rate,
                dims=dims,
                rt60=rt60,
                room_profile=room_profile,
                head_center=head_center,
                source_xyz=source_xyz,
                max_order=brir_max_order,
                brir_seconds=brir_seconds,
                late_seed=int(late_seed),
                chunk_id=i,
                early_cut_ms=early_cut_ms,
                late_start_ms=late_start_ms,
                keep_path_debug=keep_path_debug,
            )
        else:
            brir, report = synthesize_pathwise_hrtf_brir(
                subject=subject,
                sample_rate=sample_rate,
                dims=dims,
                rt60=rt60,
                head_center=head_center,
                source_xyz=source_xyz,
                max_order=brir_max_order,
                brir_seconds=brir_seconds,
                chunk_id=i,
                keep_path_debug=keep_path_debug,
            )
        left = fftconvolve(dry, brir[0], mode="full")
        right = fftconvolve(dry, brir[1], mode="full")
        conv = np.stack([left, right], axis=1).astype(np.float32)
        end = min(out_len, start + len(conv))
        out[start:end] += conv[:end - start]
        direct_rendered_angles.append(float(report["direct_rendered_azimuth_deg"]))
        direct_selected_indices.append(int(report["direct_selected_hrir_index"]))
        direct_distances.append(float(report["direct_path_distance_m"]))
        direct_delays.append(int(report["direct_path_delay_samples"]))
        num_paths_seq.append(int(report["num_paths"]))
        estimated_rt60_seq.append(float(report["estimated_rt60"]))
        estimated_drr_seq.append(float(report.get("estimated_drr_db", float("nan"))))
        early_late_ratio_seq.append(float(report.get("estimated_early_late_ratio_db", float("nan"))))
        late_join_metric_seq.append(float(report.get("late_join_metric", float("nan"))))
        late_join_ok_seq.append(bool(report.get("late_join_ok", False)))
        early_path_count_seq.append(int(report.get("early_path_count", report["num_paths"])))
        early_last_delay_ms_seq.append(float(report.get("early_last_delay_ms", float("nan"))))
        low_band_corr_seq.append(float(report.get("low_band_corr", float("nan"))))
        mid_band_corr_seq.append(float(report.get("mid_band_corr", float("nan"))))
        high_band_corr_seq.append(float(report.get("high_band_corr", float("nan"))))
        if keep_path_debug:
            path_debug_rows.extend(report["path_debug"])
    target_len = len(angle_seq) * chunk_samples
    stereo = peak_normalize(out[:target_len])
    mean_estimated_rt60 = float(np.nanmean(estimated_rt60_seq)) if estimated_rt60_seq else 0.0
    mean_late_join_metric = float(np.nanmean(late_join_metric_seq)) if late_join_metric_seq else float("nan")
    late_join_pass_rate = float(np.mean(late_join_ok_seq)) if late_join_ok_seq else 0.0
    rt60_close = rt60_close_to_target_ok(rt60, mean_estimated_rt60)
    rt60_profile = rt60_within_profile_ok(room_profile, mean_estimated_rt60)
    return stereo, {
        "direct_rendered_angle_seq": np.asarray(direct_rendered_angles, dtype=np.float32),
        "direct_selected_hrir_index_seq": direct_selected_indices,
        "direct_path_distance_seq": direct_distances,
        "direct_path_delay_samples_seq": direct_delays,
        "num_paths_seq": num_paths_seq,
        "early_path_count_seq": early_path_count_seq,
        "early_last_delay_ms_seq": early_last_delay_ms_seq,
        "estimated_rt60_seq": estimated_rt60_seq,
        "estimated_rt60": mean_estimated_rt60,
        "estimated_drr_db_seq": estimated_drr_seq,
        "estimated_drr_db": float(np.nanmean(estimated_drr_seq)) if estimated_drr_seq else float("nan"),
        "estimated_early_late_ratio_db_seq": early_late_ratio_seq,
        "estimated_early_late_ratio_db": float(np.nanmean(early_late_ratio_seq)) if early_late_ratio_seq else float("nan"),
        "late_join_metric_seq": late_join_metric_seq,
        "late_join_metric": mean_late_join_metric,
        "late_join_ok_seq": late_join_ok_seq,
        "late_join_pass_rate": late_join_pass_rate,
        "late_join_ok": bool(late_join_pass_rate >= 0.80) if late_join_ok_seq else False,
        "rt60_close_to_target_ok": bool(rt60_close),
        "rt60_within_profile_ok": bool(rt60_profile),
        "quality_gate_ok": bool(rt60_close and rt60_profile and late_join_pass_rate >= 0.80),
        "late_seed": None if late_seed is None else int(late_seed),
        "late_seed_scope": "per_trajectory" if late_seed is not None else "none",
        "low_band_corr": float(np.nanmean(low_band_corr_seq)) if low_band_corr_seq else float("nan"),
        "mid_band_corr": float(np.nanmean(mid_band_corr_seq)) if mid_band_corr_seq else float("nan"),
        "high_band_corr": float(np.nanmean(high_band_corr_seq)) if high_band_corr_seq else float("nan"),
        "path_debug": path_debug_rows,
    }


def choose_kind(rng: random.Random) -> str:
    v = rng.random()
    if v < 0.2:
        return "static"
    if v < 0.8:
        return "linear"
    return "piecewise"


def write_path_debug_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "chunk_id", "path_id", "order", "is_direct",
        "image_source_x", "image_source_y", "image_source_z",
        "distance_m", "delay_samples", "gain",
        "arrival_azimuth_deg", "arrival_elevation_deg",
        "selected_hrir_index", "selected_hrir_azimuth_deg", "selected_hrir_elevation_deg",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_one_moving_sample(
    idx: int,
    subject_id: str,
    subject: HRTFSubject,
    args,
    speech_files,
    rng: random.Random,
    np_rng: np.random.Generator,
    noise_files,
    seed_offset: int,
) -> Tuple[np.ndarray, Dict[str, object], Dict[str, object]]:
    chunk_samples = int(round(args.chunk_seconds * args.sample_rate))
    num_samples = int(round(args.duration_sec * args.sample_rate))
    crossfade_samples = int(round(args.crossfade_ms * args.sample_rate / 1000.0))

    speech_path = speech_files[int(np_rng.integers(0, len(speech_files)))]
    speech = fit_length(load_mono(speech_path, args.sample_rate), num_samples, np_rng)
    kind = choose_kind(rng)
    target_angle_seq, speed, speed_bin = generate_trajectory(rng, args.label_steps, kind)
    target_label_seq = [angle_to_label(a, args.num_classes) for a in target_angle_seq]
    distance = rng.uniform(args.distance_min, args.distance_max)
    room_profile = None
    room_dims = None
    head_center = None
    source_positions = None
    rt60 = 0.0
    brir_report: Dict[str, object] = {}
    noise_report: Dict[str, object] = {
        "noise_scene": "none",
        "noise_ch_left": "none",
        "noise_ch_right": "none",
        "noise_id": "none",
        "noise_mode": "none",
        "snr_db": 999.0,
    }

    if args.condition == "clean_moving":
        stereo, rendered_angle_seq = render_dynamic_clean(
            speech,
            subject,
            target_angle_seq,
            args.sample_rate,
            chunk_samples,
            crossfade_samples,
        )
        direct_selected_hrir_index_seq = [subject.nearest_idx(float(a)) for a in target_angle_seq]
        rendering_mode = "dynamic_hrtf"
    elif args.condition in {"reverb_moving_brir", "noisy_reverb_moving_brir", "moving_hybridbrir_gate2"}:
        for _ in range(200):
            try:
                room_profile = rng.choice(args.room_profiles)
                room_dims, rt60 = room_profile_params(room_profile, rng, args.rt60_min, args.rt60_max)
                head_center, source_positions = choose_moving_room_geometry(
                    room_dims,
                    target_angle_seq,
                    distance,
                    rng,
                )
                break
            except RuntimeError:
                target_angle_seq, speed, speed_bin = generate_trajectory(rng, args.label_steps, kind)
                target_label_seq = [angle_to_label(a, args.num_classes) for a in target_angle_seq]
        if head_center is None or source_positions is None or room_dims is None:
            raise RuntimeError("Failed to sample a valid moving room geometry")
        if args.condition == "moving_hybridbrir_gate2":
            rendering_mode = "moving_hybrid_pathwise_hrtf_brir_gate2_v1"
            late_seed = stable_late_seed(subject_id, room_profile, rt60, head_center, args.seed + seed_offset + idx)
        else:
            rendering_mode = "image_source_pathwise_dynamic_hrtf_brir"
            late_seed = None
        stereo, brir_report = render_dynamic_pathwise_brir(
            speech=speech,
            subject=subject,
            angle_seq=target_angle_seq,
            source_positions=source_positions,
            sample_rate=args.sample_rate,
            chunk_samples=chunk_samples,
            dims=room_dims,
            rt60=rt60,
            room_profile=room_profile,
            head_center=head_center,
            brir_max_order=args.brir_max_order,
            brir_seconds=args.brir_seconds,
            rendering_mode=rendering_mode,
            late_seed=late_seed,
            early_cut_ms=args.early_cut_ms,
            late_start_ms=args.late_start_ms,
            keep_path_debug=args.path_debug_csv,
        )
        rendered_angle_seq = brir_report["direct_rendered_angle_seq"]
        direct_selected_hrir_index_seq = brir_report["direct_selected_hrir_index_seq"]
        if args.condition in {"noisy_reverb_moving_brir", "moving_hybridbrir_gate2"}:
            stereo = peak_normalize(stereo)
            noise_l, noise_r, noise_report = choose_binaural_noise(noise_files, num_samples, args.sample_rate, rng)
            snr_db = rng.uniform(args.snr_min, args.snr_max)
            stereo = peak_normalize(mix_stereo_at_snr(stereo, noise_l, noise_r, snr_db))
            noise_report["snr_db"] = float(snr_db)
    else:
        raise ValueError(f"Unsupported condition: {args.condition}")

    rendered_label_seq = [angle_to_label(a, args.num_classes) for a in rendered_angle_seq]
    if args.label_source == "target":
        doa_label_seq = target_label_seq
        doa_angle_seq = target_angle_seq
    elif args.label_source == "rendered":
        doa_label_seq = rendered_label_seq
        doa_angle_seq = rendered_angle_seq
    else:
        raise ValueError(f"Unsupported label_source: {args.label_source}")

    meta = {
        "condition": args.condition,
        "subject_id": subject_id,
        "speech_path": str(speech_path),
        "angle_seq": [float(a) for a in target_angle_seq],
        "doa_labels": doa_label_seq,
        "doa_angles": [float(a) for a in doa_angle_seq],
        "label_source": args.label_source,
        "target_angle_seq": [float(a) for a in target_angle_seq],
        "rendered_angle_seq": [float(a) for a in rendered_angle_seq],
        "direct_rendered_angle_seq": [float(a) for a in rendered_angle_seq],
        "target_label_seq": target_label_seq,
        "rendered_label_seq": rendered_label_seq,
        "direct_selected_hrir_index_seq": [int(x) for x in direct_selected_hrir_index_seq],
        "trajectory_type": kind,
        "speed": float(speed),
        "speed_bin": speed_bin,
        "distance": float(distance),
        "rt60": float(rt60),
        "target_rt60": float(rt60),
        "estimated_rt60": float(brir_report.get("estimated_rt60", 0.0)),
        "estimated_rt60_seq": brir_report.get("estimated_rt60_seq", []),
        "estimated_drr_db": brir_report.get("estimated_drr_db", None),
        "estimated_drr_db_seq": brir_report.get("estimated_drr_db_seq", []),
        "estimated_early_late_ratio_db": brir_report.get("estimated_early_late_ratio_db", None),
        "estimated_early_late_ratio_db_seq": brir_report.get("estimated_early_late_ratio_db_seq", []),
        "room_profile": room_profile or "none",
        "room_dims_m": (
            "none"
            if room_dims is None
            else f"{room_dims[0]:.2f}x{room_dims[1]:.2f}x{room_dims[2]:.2f}"
        ),
        "listener_position": None if head_center is None else [float(x) for x in head_center],
        "source_position_seq": None if source_positions is None else [[float(v) for v in row] for row in source_positions],
        "direct_path_distance_seq": brir_report.get("direct_path_distance_seq", []),
        "direct_path_delay_samples_seq": brir_report.get("direct_path_delay_samples_seq", []),
        "num_paths_seq": brir_report.get("num_paths_seq", []),
        "early_path_count_seq": brir_report.get("early_path_count_seq", []),
        "early_last_delay_ms_seq": brir_report.get("early_last_delay_ms_seq", []),
        "brir_max_order": int(args.brir_max_order),
        "brir_seconds": float(args.brir_seconds),
        "early_cut_ms": float(args.early_cut_ms),
        "late_start_ms": float(args.late_start_ms),
        "late_tail_type": "binaural_diffuse_statistical" if args.condition == "moving_hybridbrir_gate2" else "none",
        "late_seed": brir_report.get("late_seed", None),
        "late_seed_scope": brir_report.get("late_seed_scope", "none"),
        "late_join_ok": brir_report.get("late_join_ok", None),
        "late_join_ok_seq": brir_report.get("late_join_ok_seq", []),
        "late_join_pass_rate": brir_report.get("late_join_pass_rate", None),
        "late_join_metric": brir_report.get("late_join_metric", None),
        "late_join_metric_seq": brir_report.get("late_join_metric_seq", []),
        "rt60_close_to_target_ok": brir_report.get("rt60_close_to_target_ok", None),
        "rt60_within_profile_ok": brir_report.get("rt60_within_profile_ok", None),
        "quality_gate_ok": brir_report.get("quality_gate_ok", None),
        "low_band_corr": brir_report.get("low_band_corr", None),
        "mid_band_corr": brir_report.get("mid_band_corr", None),
        "high_band_corr": brir_report.get("high_band_corr", None),
        "rendering_mode": rendering_mode,
        "snr": float(noise_report["snr_db"]),
        "snr_db": float(noise_report["snr_db"]),
        "noise_scene": noise_report["noise_scene"],
        "noise_ch_left": noise_report["noise_ch_left"],
        "noise_ch_right": noise_report["noise_ch_right"],
        "noise_id": noise_report["noise_id"],
        "noise_mode": noise_report["noise_mode"],
        "sample_rate": args.sample_rate,
        "duration_sec": args.duration_sec,
        "chunk_seconds": args.chunk_seconds,
        "crossfade_ms": args.crossfade_ms,
    }
    return stereo, meta, brir_report


def generate_split(split_name: str, root: Path, subjects: Sequence[str], args, speech_files, hrtf_cache, noise_files, seed_offset: int) -> None:
    wav_dir = root / "binaural_dev"
    meta_dir = root / "metadata_dev"
    path_debug_dir = root / "path_debug"
    wav_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    if args.path_debug_csv:
        path_debug_dir.mkdir(parents=True, exist_ok=True)
    total = args.samples_per_train if split_name == "train_subjects" else args.samples_per_eval
    start_idx = max(1, int(args.index_start))
    end_idx = int(args.index_end) if args.index_end is not None else total
    end_idx = min(total, end_idx)
    if start_idx > end_idx:
        raise ValueError(f"Empty index range for {split_name}: {start_idx}..{end_idx}")
    shard_total = end_idx - start_idx + 1
    accepted_first_try = 0
    total_attempts = 0
    for local_i, idx in enumerate(range(start_idx, end_idx + 1), start=1):
        sample_seed = args.seed + seed_offset * 1_000_003 + idx
        rng = random.Random(sample_seed)
        np_rng = np.random.default_rng(sample_seed)
        subject_id = subjects[(idx - 1) % len(subjects)]
        subject = hrtf_cache[subject_id]
        last_stereo = None
        last_meta = None
        last_brir_report: Dict[str, object] = {}
        accepted_attempt = 0
        for attempt in range(1, int(args.max_attempts_per_sample) + 1):
            stereo, meta, brir_report = render_one_moving_sample(
                idx=idx,
                subject_id=subject_id,
                subject=subject,
                args=args,
                speech_files=speech_files,
                rng=rng,
                np_rng=np_rng,
                noise_files=noise_files,
                seed_offset=seed_offset + 100000 * attempt,
            )
            last_stereo, last_meta, last_brir_report = stereo, meta, brir_report
            total_attempts += 1
            if (not args.require_quality_gate) or bool(meta.get("quality_gate_ok", True)):
                accepted_attempt = attempt
                break
        if last_stereo is None or last_meta is None:
            raise RuntimeError("Failed to render moving sample")
        if accepted_attempt == 0:
            accepted_attempt = int(args.max_attempts_per_sample)
        if accepted_attempt == 1:
            accepted_first_try += 1

        file_id = f"{idx:06d}"
        last_meta["file_id"] = file_id
        last_meta["quality_gate_required"] = bool(args.require_quality_gate)
        last_meta["accepted_attempt"] = int(accepted_attempt)
        last_meta["max_attempts_per_sample"] = int(args.max_attempts_per_sample)
        sf.write(str(wav_dir / f"binaural{file_id}.wav"), last_stereo, args.sample_rate, subtype="PCM_16")
        if args.path_debug_csv and last_brir_report.get("path_debug"):
            write_path_debug_csv(path_debug_dir / f"paths{file_id}.csv", last_brir_report["path_debug"])
        (meta_dir / f"metadata{file_id}.json").write_text(json.dumps(last_meta, indent=2), encoding="utf-8")
        if local_i % args.log_interval == 0 or idx == end_idx:
            avg_attempts = total_attempts / float(local_i)
            print(
                f"[{split_name}] {idx}/{total} shard={start_idx}-{end_idx} "
                f"local={local_i}/{shard_total} avg_attempts={avg_attempts:.2f} "
                f"first_try={accepted_first_try / float(local_i):.3f}",
                flush=True,
            )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--librispeech_root", type=Path, default=Path("/disk2/bywang/data/LibriSpeech/train-clean-100"))
    p.add_argument("--hrtf_root", type=Path, default=Path("/disk2/bywang/data/HRTF"))
    p.add_argument("--demand_root", type=Path, default=Path("/disk2/bywang/data/demand"))
    p.add_argument("--output_root", type=Path, default=Path("/disk2/bywang/DOA-net/data/librispeech_cipic_moving_clean_v1"))
    p.add_argument(
        "--condition",
        choices=["clean_moving", "reverb_moving_brir", "noisy_reverb_moving_brir", "moving_hybridbrir_gate2"],
        default="clean_moving",
    )
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--duration_sec", type=float, default=4.0)
    p.add_argument("--chunk_seconds", type=float, default=0.1)
    p.add_argument("--label_steps", type=int, default=40)
    p.add_argument("--label_source", choices=["target", "rendered"], default="target")
    p.add_argument("--num_classes", type=int, default=72)
    p.add_argument("--distance_min", type=float, default=1.0)
    p.add_argument("--distance_max", type=float, default=1.5)
    p.add_argument("--total_subjects", type=int, default=30)
    p.add_argument("--train_subjects", type=str, default=None)
    p.add_argument("--val_subjects", type=str, default=None)
    p.add_argument("--test_subjects", type=str, default=None)
    p.add_argument("--samples_per_train", type=int, default=20000)
    p.add_argument("--samples_per_eval", type=int, default=2000)
    p.add_argument("--crossfade_ms", type=float, default=8.0)
    p.add_argument("--rt60_min", type=float, default=None)
    p.add_argument("--rt60_max", type=float, default=None)
    p.add_argument("--room_profiles", nargs="+", default=["small", "medium", "large"])
    p.add_argument("--brir_max_order", type=int, default=3)
    p.add_argument("--brir_seconds", type=float, default=1.8)
    p.add_argument("--early_cut_ms", type=float, default=80.0)
    p.add_argument("--late_start_ms", type=float, default=80.0)
    p.add_argument("--scenes", nargs="+", default=DEMAND_SCENES)
    p.add_argument("--snr_min", type=float, default=-10.0)
    p.add_argument("--snr_max", type=float, default=10.0)
    p.add_argument("--path_debug_csv", action="store_true")
    p.add_argument("--require_quality_gate", action="store_true")
    p.add_argument("--max_attempts_per_sample", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--append", action="store_true", help="Allow writing into an existing output_root without deleting it.")
    p.add_argument(
        "--split_only",
        choices=["all", "train_subjects", "val_subjects", "test_subjects_unseen"],
        default="all",
        help="Generate only one split; useful for parallel sharded generation.",
    )
    p.add_argument("--index_start", type=int, default=1, help="1-based first sample index to generate in the selected split.")
    p.add_argument("--index_end", type=int, default=None, help="1-based last sample index to generate in the selected split.")
    p.add_argument("--skip_manifest", action="store_true", help="Do not write manifest.json; useful for parallel workers.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--log_interval", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.output_root = args.output_root.with_name(args.output_root.name + "_smoke")
        args.total_subjects = min(args.total_subjects, 10)
        args.samples_per_train = min(args.samples_per_train, 96)
        args.samples_per_eval = min(args.samples_per_eval, 24)
        args.log_interval = 24
    if args.output_root.exists():
        if not args.overwrite:
            if not args.append:
                raise FileExistsError(f"{args.output_root} exists; pass --overwrite or --append")
        else:
            shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    speech_files = sorted(args.librispeech_root.rglob("*.flac"))
    if not speech_files:
        raise FileNotFoundError(f"No flac files found under {args.librispeech_root}")
    noise_files = None
    if args.condition in {"noisy_reverb_moving_brir", "moving_hybridbrir_gate2"}:
        noise_files = list_noise_files(args.demand_root, args.scenes)
    if args.train_subjects or args.val_subjects or args.test_subjects:
        split = {
            "train_subjects": parse_subject_list(args.train_subjects or ""),
            "val_subjects": parse_subject_list(args.val_subjects or ""),
            "test_subjects_unseen": parse_subject_list(args.test_subjects or ""),
        }
        missing = [name for name, values in split.items() if not values]
        if missing:
            raise ValueError(f"Explicit subject split requires non-empty values for: {missing}")
    else:
        split = split_subjects(list_subjects(args.hrtf_root), args.total_subjects, args.seed)
    hrtf_cache = {
        sid: HRTFSubject(args.hrtf_root / f"subject_{sid}.sofa", args.sample_rate)
        for ids in split.values()
        for sid in ids
    }
    split_names = ("train_subjects", "val_subjects", "test_subjects_unseen")
    selected_split_names = split_names if args.split_only == "all" else (args.split_only,)
    for offset, split_name in enumerate(split_names):
        if split_name not in selected_split_names:
            continue
        generate_split(split_name, args.output_root / split_name, split[split_name], args, speech_files, hrtf_cache, noise_files, offset)
    manifest = {"split": split, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    if not args.skip_manifest:
        (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done: {args.output_root}")


if __name__ == "__main__":
    main()
