#!/usr/bin/env python3
"""Generate subject003 data with room reverberation and optional DEMAND noise.

Output format keeps compatibility with current training pipeline:
- binaural_dev/binauralXXXX.wav
- metadata_dev/metadataXXXX.csv

Per recording, the script samples one of three modes:
- clean
- reverb_only
- reverb_plus_noise

Reverberation is synthesized by a lightweight binaural room impulse response
generator conditioned on room profile (small/medium/large) and RT60 range.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

try:
    import librosa
except Exception:
    librosa = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare reverb + DEMAND mixed subject003 dataset")
    parser.add_argument(
        "--clean_root",
        type=Path,
        default=Path("/disk2/bywang/DOA-net/data/librispeech_cipic_subject003"),
    )
    parser.add_argument(
        "--demand_root",
        type=Path,
        default=Path("/disk2/bywang/data/demand"),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/disk2/bywang/DOA-net/data/librispeech_cipic_subject003_reverb_demand_m10_10"),
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["OOFFICE", "PCAFETER", "TMETRO", "TBUS", "SPSQUARE", "NPARK"],
        help="Selected DEMAND scenes",
    )
    parser.add_argument("--snr_min_db", type=float, default=-10.0)
    parser.add_argument("--snr_max_db", type=float, default=10.0)
    parser.add_argument("--rt60_min", type=float, default=0.2)
    parser.add_argument("--rt60_max", type=float, default=0.8)
    parser.add_argument(
        "--room_profiles",
        nargs="+",
        default=["small", "medium", "large"],
        choices=["small", "medium", "large"],
        help="Room size profiles to sample",
    )
    parser.add_argument("--clean_prob", type=float, default=0.4)
    parser.add_argument("--reverb_only_prob", type=float, default=0.3)
    parser.add_argument("--reverb_noise_prob", type=float, default=0.3)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def resample_if_needed(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return x
    if librosa is None:
        raise RuntimeError("librosa is required for resampling but is not available")

    if x.ndim == 1:
        return librosa.resample(x, orig_sr=orig_sr, target_sr=target_sr)

    ys = [librosa.resample(x[:, c], orig_sr=orig_sr, target_sr=target_sr) for c in range(x.shape[1])]
    min_len = min(len(y) for y in ys)
    return np.stack([y[:min_len] for y in ys], axis=1)


def list_channel_wavs(scene_dir: Path) -> List[Path]:
    wavs = sorted(scene_dir.glob("ch*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No channel wav found in {scene_dir}")
    return wavs


def _room_dims(profile: str, rng: random.Random) -> Tuple[float, float, float]:
    if profile == "small":
        return (
            rng.uniform(3.2, 4.2),
            rng.uniform(3.5, 4.8),
            rng.uniform(2.6, 3.0),
        )
    if profile == "medium":
        return (
            rng.uniform(5.0, 6.8),
            rng.uniform(4.5, 6.2),
            rng.uniform(2.8, 3.2),
        )
    return (
        rng.uniform(7.5, 10.0),
        rng.uniform(6.0, 8.0),
        rng.uniform(2.9, 3.5),
    )


def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 2 ** ((x - 1).bit_length())


def _fft_convolve_1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    out_len = len(x) + len(h) - 1
    n_fft = _next_pow2(out_len)
    X = np.fft.rfft(x, n_fft)
    H = np.fft.rfft(h, n_fft)
    y = np.fft.irfft(X * H, n_fft)
    return y[:out_len].astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + 1e-12))


def synthesize_binaural_rir(
    sr: int,
    room_dims: Tuple[float, float, float],
    rt60: float,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    c = 343.0
    lx, ly, lz = room_dims
    room_diag = math.sqrt(lx * lx + ly * ly + lz * lz)

    src_dist = rng.uniform(1.0, min(4.0, room_diag * 0.5))
    az_deg = rng.uniform(-180.0, 180.0)
    az_rad = math.radians(az_deg)

    base_delay_s = src_dist / c
    head_width = 0.18
    itd_s = (head_width / c) * math.sin(az_rad)
    ild_db = 3.0 * math.sin(az_rad)

    d_l = max(0.0, base_delay_s + 0.5 * itd_s)
    d_r = max(0.0, base_delay_s - 0.5 * itd_s)
    idx_l = int(round(d_l * sr))
    idx_r = int(round(d_r * sr))

    rir_len = int(sr * max(0.25, min(1.2, rt60 * 1.5)))
    rir_l = np.zeros(rir_len, dtype=np.float32)
    rir_r = np.zeros(rir_len, dtype=np.float32)

    g_l = 10.0 ** (0.5 * ild_db / 20.0)
    g_r = 10.0 ** (-0.5 * ild_db / 20.0)
    if idx_l < rir_len:
        rir_l[idx_l] += g_l
    if idx_r < rir_len:
        rir_r[idx_r] += g_r

    vol = lx * ly * lz
    n_reflections = int(18 + vol / 8.0)
    refl_max_s = min(0.25, 0.6 * rt60)
    for _ in range(n_reflections):
        d_refl = rng.uniform(0.003, refl_max_s)
        ilr = rng.uniform(0.85, 1.15)

        amp = rng.uniform(0.12, 0.6)
        amp *= math.exp(math.log(1e-3) * (d_refl / max(rt60, 1e-3)))

        i_l = int(round((d_l + d_refl) * sr))
        i_r = int(round((d_r + d_refl * ilr) * sr))
        if i_l < rir_len:
            rir_l[i_l] += amp * rng.uniform(0.8, 1.2)
        if i_r < rir_len:
            rir_r[i_r] += amp * rng.uniform(0.8, 1.2)

    t = np.arange(rir_len, dtype=np.float32) / float(sr)
    decay_env = np.exp(np.log(1e-3) * (t / max(rt60, 1e-3))).astype(np.float32)
    tail_l = rng.normalvariate(0.0, 1.0)
    tail_r = rng.normalvariate(0.0, 1.0)
    noise_l = (np.random.randn(rir_len).astype(np.float32) * decay_env * 0.02 * abs(tail_l))
    noise_r = (np.random.randn(rir_len).astype(np.float32) * decay_env * 0.02 * abs(tail_r))
    rir_l += noise_l
    rir_r += noise_r

    nrm = max(np.max(np.abs(rir_l)), np.max(np.abs(rir_r)), 1e-6)
    rir_l /= nrm
    rir_r /= nrm

    return rir_l, rir_r, {
        "src_distance_m": src_dist,
        "src_azimuth_deg": az_deg,
    }


def apply_reverb_binaural(clean: np.ndarray, rir_l: np.ndarray, rir_r: np.ndarray) -> np.ndarray:
    out_l = _fft_convolve_1d(clean[:, 0], rir_l)[: len(clean)]
    out_r = _fft_convolve_1d(clean[:, 1], rir_r)[: len(clean)]
    rev = np.stack([out_l, out_r], axis=1)

    for ch in range(2):
        in_rms = _rms(clean[:, ch])
        out_rms = _rms(rev[:, ch])
        if out_rms > 1e-9:
            rev[:, ch] *= in_rms / out_rms
    return np.clip(rev, -1.0, 1.0).astype(np.float32)


def read_noise_segment(noise_path: Path, length: int, rng: random.Random, target_sr: int) -> np.ndarray:
    info = sf.info(str(noise_path))
    total = info.frames
    sr = info.samplerate

    if total >= length:
        start = rng.randint(0, max(0, total - length))
        seg, seg_sr = sf.read(str(noise_path), start=start, frames=length, dtype="float32", always_2d=False)
    else:
        seg, seg_sr = sf.read(str(noise_path), dtype="float32", always_2d=False)

    seg = np.asarray(seg, dtype=np.float32)
    if seg.ndim > 1:
        seg = seg[:, 0]

    if seg_sr != target_sr:
        seg = resample_if_needed(seg, seg_sr, target_sr).astype(np.float32)

    if len(seg) < length:
        reps = int(np.ceil(length / max(1, len(seg))))
        seg = np.tile(seg, reps)

    return seg[:length]


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    clean = clean.astype(np.float32)
    noise = noise.astype(np.float32)

    clean_power = float(np.mean(clean ** 2))
    noise = noise - float(np.mean(noise))
    noise_power = float(np.mean(noise ** 2))

    if clean_power < 1e-12 or noise_power < 1e-12:
        return clean.copy()

    snr_linear = 10.0 ** (snr_db / 10.0)
    scale = np.sqrt(clean_power / (snr_linear * noise_power))
    mixed = clean + scale * noise
    return np.clip(mixed, -1.0, 1.0)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    clean_wav_dir = args.clean_root / "binaural_dev"
    clean_meta_dir = args.clean_root / "metadata_dev"

    out_wav_dir = args.output_root / "binaural_dev"
    out_meta_dir = args.output_root / "metadata_dev"

    if args.output_root.exists() and args.overwrite:
        shutil.rmtree(args.output_root)
    ensure_dir(out_wav_dir)
    ensure_dir(out_meta_dir)

    scene_to_wavs = {}
    for scene in args.scenes:
        scene_dir = args.demand_root / scene
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
        scene_to_wavs[scene] = list_channel_wavs(scene_dir)

    clean_wavs = sorted(clean_wav_dir.glob("binaural*.wav"))
    if not clean_wavs:
        raise FileNotFoundError(f"No clean wav found in {clean_wav_dir}")

    prob_sum = args.clean_prob + args.reverb_only_prob + args.reverb_noise_prob
    if prob_sum <= 0:
        raise ValueError("clean_prob + reverb_only_prob + reverb_noise_prob must be > 0")
    p_clean = args.clean_prob / prob_sum
    p_reverb_only = args.reverb_only_prob / prob_sum

    report_path = args.output_root / "mixing_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_id",
            "mode",
            "room_profile",
            "room_dims_m",
            "rt60_s",
            "scene",
            "noise_ch_left",
            "noise_ch_right",
            "snr_db",
            "src_distance_m",
            "src_azimuth_deg",
            "num_samples",
        ])

        for idx, wav_path in enumerate(clean_wavs, start=1):
            file_id = wav_path.stem.replace("binaural", "")
            meta_path = clean_meta_dir / f"metadata{file_id}.csv"
            if not meta_path.is_file():
                continue

            clean, clean_sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
            clean = np.asarray(clean, dtype=np.float32)
            clean = resample_if_needed(clean, clean_sr, args.sample_rate).astype(np.float32)

            if clean.shape[1] < 2:
                clean = np.repeat(clean, 2, axis=1)
            elif clean.shape[1] > 2:
                clean = clean[:, :2]

            n = clean.shape[0]

            u = rng.random()
            if u < p_clean:
                mode = "clean"
            elif u < p_clean + p_reverb_only:
                mode = "reverb_only"
            else:
                mode = "reverb_plus_noise"

            room_profile = rng.choice(args.room_profiles)
            dims = _room_dims(room_profile, rng)
            rt60 = rng.uniform(args.rt60_min, args.rt60_max)
            rir_l, rir_r, src_info = synthesize_binaural_rir(args.sample_rate, dims, rt60, rng)

            processed = apply_reverb_binaural(clean, rir_l, rir_r)

            scene = ""
            left_noise_wav = ""
            right_noise_wav = ""
            snr_db = ""

            if mode == "clean":
                mixed = clean
                rt60 = 0.0
                room_profile = "none"
                dims = (0.0, 0.0, 0.0)
                src_info = {"src_distance_m": 0.0, "src_azimuth_deg": 0.0}
            elif mode == "reverb_only":
                mixed = processed
            else:
                scene = rng.choice(args.scenes)
                scene_wavs = scene_to_wavs[scene]
                left_noise_wav = rng.choice(scene_wavs).name
                right_noise_wav = rng.choice(scene_wavs).name
                snr_val = rng.uniform(args.snr_min_db, args.snr_max_db)

                noise_l = read_noise_segment(Path(args.demand_root / scene / left_noise_wav), n, rng, args.sample_rate)
                noise_r = read_noise_segment(Path(args.demand_root / scene / right_noise_wav), n, rng, args.sample_rate)

                mixed_l = mix_at_snr(processed[:, 0], noise_l, snr_val)
                mixed_r = mix_at_snr(processed[:, 1], noise_r, snr_val)
                mixed = np.stack([mixed_l, mixed_r], axis=1)
                snr_db = f"{snr_val:.4f}"

            out_wav = out_wav_dir / f"binaural{file_id}.wav"
            sf.write(str(out_wav), mixed, args.sample_rate)
            shutil.copy2(meta_path, out_meta_dir / f"metadata{file_id}.csv")

            writer.writerow([
                file_id,
                mode,
                room_profile,
                f"{dims[0]:.2f}x{dims[1]:.2f}x{dims[2]:.2f}",
                f"{rt60:.4f}",
                scene,
                left_noise_wav,
                right_noise_wav,
                snr_db,
                f"{src_info['src_distance_m']:.4f}",
                f"{src_info['src_azimuth_deg']:.4f}",
                n,
            ])

            if idx % 500 == 0:
                print(f"Processed {idx}/{len(clean_wavs)}")

    print("Done.")
    print(f"Output dataset: {args.output_root}")
    print(f"Mixing report : {report_path}")


if __name__ == "__main__":
    main()
