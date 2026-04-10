#!/usr/bin/env python3
"""使用 LibriSpeech + CIPIC SOFA 合成静态双耳 DOA 训练数据。

输出目录结构与当前项目兼容:
  <output_root>/
    binaural_dev/binauralXXXX.wav
    metadata_dev/metadataXXXX.csv

metadata 文件按现有 `dataset/static_dataset.py` 的读取习惯写为:
  frame_idx,x,y,z,0,0,0,0

示例:
  /home/bywang/miniconda3/envs/doa/bin/python synthesize_librispeech_cipic.py \
    --librispeech_root /disk2/bywang/data/LibriSpeech/train-clean-100 \
    --sofa_path /disk2/bywang/data/HRTF/subject_003.sofa \
    --output_root /disk2/bywang/DOA-net/data/librispeech_cipic_subject003 \
    --num_recordings 5000
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sofa
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly


def wrap_deg(angle_deg: float) -> float:
    """将角度映射到 [-180, 180)。"""
    return ((angle_deg + 180.0) % 360.0) - 180.0


def spherical_to_cartesian(az_deg: float, el_deg: float, r: float) -> Tuple[float, float, float]:
    """将球坐标(az, el, r)转为笛卡尔坐标(x, y, z)。

    这里约定 x=右, y=前, z=上，使得 atan2(x, y) == az。
    """
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    x = r * math.cos(el) * math.sin(az)
    y = r * math.cos(el) * math.cos(az)
    z = r * math.sin(el)
    return x, y, z


def build_balanced_measurement_pool(
    source_positions: np.ndarray,
    num_classes: int,
    azimuth_range: Tuple[float, float],
) -> Tuple[Dict[int, List[int]], np.ndarray]:
    """根据 SOFA 测点建立按方位角 bin 的索引池。"""
    az_min, az_max = azimuth_range
    az_resolution = (az_max - az_min) / num_classes

    # CIPIC SOFA 中 Source.Position 默认通常为 [azimuth, elevation, radius]
    azimuths = np.array([wrap_deg(a) for a in source_positions[:, 0]], dtype=np.float64)

    bin_to_indices: Dict[int, List[int]] = {i: [] for i in range(num_classes)}
    for idx, az in enumerate(azimuths):
        rel = (az - az_min) / az_resolution
        bin_idx = int(np.floor(rel))
        if 0 <= bin_idx < num_classes:
            bin_to_indices[bin_idx].append(idx)

    return bin_to_indices, azimuths


def list_librispeech_files(root: Path) -> List[Path]:
    files = sorted(root.rglob("*.flac"))
    if not files:
        raise FileNotFoundError(f"No .flac files found under {root}")
    return files


def load_mono_resampled(path: Path, target_sr: int) -> np.ndarray:
    """读取语音并重采样到 target_sr，返回 float32 单通道。"""
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32, copy=False)
    if sr != target_sr:
        g = math.gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        audio = resample_poly(audio, up=up, down=down).astype(np.float32, copy=False)

    return audio


def fit_to_length(audio: np.ndarray, num_samples: int, rng: np.random.Generator) -> np.ndarray:
    """将单通道语音裁剪/平铺到固定长度。"""
    if len(audio) == 0:
        return np.zeros(num_samples, dtype=np.float32)

    if len(audio) >= num_samples:
        max_start = len(audio) - num_samples
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        return audio[start : start + num_samples]

    reps = int(np.ceil(num_samples / len(audio)))
    tiled = np.tile(audio, reps)
    return tiled[:num_samples].astype(np.float32, copy=False)


def apply_hrir(speech: np.ndarray, hrir_l: np.ndarray, hrir_r: np.ndarray) -> np.ndarray:
    """对单通道语音施加左右耳 HRIR，输出 [N, 2]。"""
    left = fftconvolve(speech, hrir_l, mode="full")
    right = fftconvolve(speech, hrir_r, mode="full")

    n = len(speech)
    stereo = np.stack([left[:n], right[:n]], axis=1).astype(np.float32, copy=False)

    peak = float(np.max(np.abs(stereo)))
    if peak > 1e-7:
        stereo = stereo * (0.95 / peak)

    return stereo


def write_metadata_csv(path: Path, x: float, y: float, z: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([0, x, y, z, 0, 0, 0, 0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize binaural dataset from LibriSpeech + CIPIC SOFA")
    parser.add_argument("--librispeech_root", type=Path, required=True)
    parser.add_argument("--sofa_path", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--num_recordings", type=int, default=5000)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration_sec", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_classes", type=int, default=72)
    parser.add_argument("--az_min", type=float, default=-180.0)
    parser.add_argument("--az_max", type=float, default=180.0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    binaural_dir = args.output_root / "binaural_dev"
    metadata_dir = args.output_root / "metadata_dev"
    binaural_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Scanning LibriSpeech from: {args.librispeech_root}")
    flac_files = list_librispeech_files(args.librispeech_root)
    print(f"      Found {len(flac_files)} speech files")

    print(f"[2/5] Loading SOFA: {args.sofa_path}")
    db = sofa.Database.open(str(args.sofa_path))
    source_positions = db.Source.Position.get_values()  # [M, 3]
    ir = db.Data.IR.get_values()  # [M, 2, L]
    sofa_sr = int(round(float(db.Data.SamplingRate.get_values()[0])))
    print(f"      SOFA positions: {source_positions.shape}, IR: {ir.shape}, sr: {sofa_sr}")

    # HRIR 重采样到目标采样率
    print(f"[3/5] Resampling HRIR to {args.sample_rate} Hz")
    if sofa_sr != args.sample_rate:
        g = math.gcd(sofa_sr, args.sample_rate)
        up = args.sample_rate // g
        down = sofa_sr // g
        hrir = resample_poly(ir, up=up, down=down, axis=-1).astype(np.float32, copy=False)
    else:
        hrir = ir.astype(np.float32, copy=False)

    azimuth_range = (args.az_min, args.az_max)
    bin_to_indices, azimuths = build_balanced_measurement_pool(
        source_positions=source_positions,
        num_classes=args.num_classes,
        azimuth_range=azimuth_range,
    )

    non_empty_bins = [b for b, idxs in bin_to_indices.items() if idxs]
    if not non_empty_bins:
        raise RuntimeError("No valid SOFA measurements mapped to azimuth bins.")

    print(f"      Non-empty azimuth bins: {len(non_empty_bins)}/{args.num_classes}")

    num_samples = int(round(args.duration_sec * args.sample_rate))

    print(f"[4/5] Synthesizing {args.num_recordings} recordings...")
    for i in range(1, args.num_recordings + 1):
        speech_path = flac_files[int(rng.integers(0, len(flac_files)))]
        speech = load_mono_resampled(speech_path, args.sample_rate)
        speech = fit_to_length(speech, num_samples, rng)

        # 尽量均衡地覆盖方位角类别
        target_bin = non_empty_bins[(i - 1) % len(non_empty_bins)]
        m_idx = int(rng.choice(bin_to_indices[target_bin]))

        hrir_l = hrir[m_idx, 0]
        hrir_r = hrir[m_idx, 1]
        stereo = apply_hrir(speech, hrir_l, hrir_r)

        out_id = f"{i:04d}"
        wav_path = binaural_dir / f"binaural{out_id}.wav"
        meta_path = metadata_dir / f"metadata{out_id}.csv"

        sf.write(wav_path, stereo, args.sample_rate, subtype="PCM_16")

        az_deg = float(azimuths[m_idx])
        el_deg = float(source_positions[m_idx, 1])
        r = float(source_positions[m_idx, 2])
        x, y, z = spherical_to_cartesian(az_deg, el_deg, r)
        write_metadata_csv(meta_path, x, y, z)

        if i % 100 == 0 or i == args.num_recordings:
            print(f"      {i}/{args.num_recordings} done")

    print(f"[5/5] Done. Output root: {args.output_root}")


if __name__ == "__main__":
    main()
