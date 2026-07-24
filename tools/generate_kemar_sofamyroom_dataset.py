#!/usr/bin/env python3
"""Generate a reverberant KEMAR + SofaMyRoom static DOA dataset.

This generator uses SofaMyRoom as the BRIR renderer. It is intentionally
separate from the older CIPIC/hybrid renderer so the angle convention and room
simulation protocol stay explicit.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly
from anf_generator import generate_signals
from anf_generator.CoherenceMatrix import Parameters as ANFParameters

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prepare_robust_multisubject_dataset import (  # noqa: E402
    fit_to_length,
    list_librispeech_files,
    list_noise_files,
    load_mono_resampled,
    make_balanced_shuffled_bins,
    mix_at_snr,
    peak_normalize,
    read_noise_segment,
    wrap_deg,
)


DEFAULT_SOFAMYROOM_BIN = Path("/disk2/bywang/data/sofamyroom/build_doa/sofamyroom")
DEFAULT_KEMAR_SOFA = Path("/disk2/bywang/data/sofamyroom/data/MIT_KEMAR_normal_pinna.sofa")
DEFAULT_CONDA_LIB = Path("/home/bywang/miniconda3/envs/doa/lib")

TRAIN_NOISE_SCENES = ["OOFFICE", "PCAFETER", "TMETRO"]
TEST_NOISE_SCENES = ["TBUS", "SPSQUARE", "NPARK"]
ROOM_PROFILES = ["small", "medium", "large"]
FREQ_BANDS = [125, 250, 500, 1000, 2000, 4000]
TEST_SNR_VALUES: Sequence[Optional[float]] = (None, 10.0, 5.0, 0.0, -5.0, -10.0)


@dataclass(frozen=True)
class RoomSpec:
    room_id: str
    profile: str
    dims: Tuple[float, float, float]
    target_rt60: float


TEST_ROOMS = [
    RoomSpec("S1", "small", (4.2, 3.8, 2.6), 0.30),
    RoomSpec("S2", "small", (4.8, 4.2, 2.8), 0.40),
    RoomSpec("M1", "medium", (5.8, 4.8, 3.0), 0.45),
    RoomSpec("M2", "medium", (6.8, 5.8, 3.2), 0.60),
    RoomSpec("L1", "large", (8.2, 6.2, 3.2), 0.65),
    RoomSpec("L2", "large", (9.8, 7.8, 3.8), 0.80),
]


def kemar_to_compatible_azimuth(kemar_azimuth_deg: float) -> float:
    return wrap_deg(float(kemar_azimuth_deg))


def kemar_azimuth_to_room_offset(azimuth_deg: float, distance_m: float) -> Tuple[float, float, float]:
    """Map KEMAR azimuth to SofaMyRoom coordinates.

    Verified convention with receiver.orientation=[0 0 0]:
    0 deg front -> +X, 90 deg right -> -Y, 180 deg back -> -X, 270 deg left -> +Y.
    """
    rad = math.radians(float(azimuth_deg))
    return (
        distance_m * math.cos(rad),
        -distance_m * math.sin(rad),
        0.0,
    )


def room_ranges(profile: str) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    if profile == "small":
        return (4.0, 5.0), (3.5, 4.5), (2.5, 3.0), (0.25, 0.45)
    if profile == "medium":
        return (5.5, 7.0), (4.5, 6.0), (2.8, 3.2), (0.35, 0.65)
    if profile == "large":
        return (8.0, 10.0), (6.0, 8.0), (3.0, 4.0), (0.50, 0.80)
    raise ValueError(f"Unknown room profile: {profile}")


def distance_range(profile: str) -> Tuple[float, float]:
    if profile == "small":
        return 1.0, 1.5
    if profile == "medium":
        return 1.0, 1.8
    if profile == "large":
        return 1.0, 2.0
    raise ValueError(f"Unknown room profile: {profile}")


def default_absorption_scale(profile: str) -> float:
    if profile == "small":
        return 0.60
    if profile == "medium":
        return 0.55
    if profile == "large":
        return 0.45
    raise ValueError(f"Unknown room profile: {profile}")


def test_distances(profile: str) -> Sequence[float]:
    if profile == "small":
        return (1.0, 1.25, 1.5)
    if profile == "medium":
        return (1.0, 1.5, 1.8)
    if profile == "large":
        return (1.0, 1.5, 2.0)
    raise ValueError(f"Unknown room profile: {profile}")


def sample_train_room(rng: random.Random) -> RoomSpec:
    profile = rng.choice(ROOM_PROFILES)
    lr, wr, hr, rr = room_ranges(profile)
    dims = (
        rng.uniform(*lr),
        rng.uniform(*wr),
        rng.uniform(*hr),
    )
    rt60 = rng.uniform(*rr)
    return RoomSpec("random", profile, dims, rt60)


def surface_area(dims: Tuple[float, float, float]) -> float:
    lx, ly, lz = dims
    return 2.0 * (lx * ly + lx * lz + ly * lz)


def eyring_absorption_for_rt60(dims: Tuple[float, float, float], rt60: float, scale: float) -> float:
    lx, ly, lz = dims
    volume = lx * ly * lz
    area = surface_area(dims)
    alpha = 1.0 - math.exp(-0.161 * volume / max(area * rt60, 1e-6))
    return float(np.clip(alpha * scale, 0.02, 0.85))


def make_absorption_matrix(alpha: float) -> np.ndarray:
    """Create a mild frequency-shaped 6x6 absorption matrix.

    The same target mean is kept across surfaces, with slightly more absorption
    at high frequencies and on floor/ceiling. This keeps the protocol simple
    while avoiding a perfectly flat material model.
    """
    freq_shape = np.asarray([0.80, 0.90, 1.00, 1.05, 1.10, 1.15], dtype=np.float64)
    surface_shape = np.asarray([1.00, 1.00, 1.00, 1.00, 1.12, 1.08], dtype=np.float64)
    mat = alpha * surface_shape[:, None] * freq_shape[None, :]
    return np.clip(mat, 0.03, 0.90)


def horizontal_wall_clearance(xyz: np.ndarray, dims: Tuple[float, float, float]) -> float:
    lx, ly, _lz = dims
    x, y, _z = xyz
    return float(min(x, lx - x, y, ly - y))


def choose_geometry(
    dims: Tuple[float, float, float],
    profile: str,
    azimuth_deg: float,
    rng: random.Random,
    split: str,
    sample_index: int,
    forced_distance: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    receiver_margin = 0.8
    source_margin = 0.55
    ear_height = 1.4
    lx, ly, lz = dims
    if not (receiver_margin < ear_height < lz - source_margin):
        raise ValueError(f"Room height too small for ear height: dims={dims}")

    if forced_distance is not None:
        distance = float(forced_distance)
    elif split == "test":
        distances = list(test_distances(profile))
        distance = distances[(sample_index - 1) % len(distances)]
    else:
        lo, hi = distance_range(profile)
        distance = rng.uniform(lo, hi)

    offset = np.asarray(kemar_azimuth_to_room_offset(azimuth_deg, distance), dtype=np.float64)
    for _ in range(400):
        receiver = np.asarray([
            rng.uniform(receiver_margin, lx - receiver_margin),
            rng.uniform(receiver_margin, ly - receiver_margin),
            ear_height,
        ], dtype=np.float64)
        source = receiver + offset
        if horizontal_wall_clearance(source, dims) >= source_margin:
            return receiver, source, float(distance)

    # Conservative fallback: center receiver, then reduce distance only if needed.
    receiver = np.asarray([lx / 2.0, ly / 2.0, ear_height], dtype=np.float64)
    rad = math.radians(float(azimuth_deg))
    max_dx = (lx / 2.0 - source_margin) / max(abs(math.cos(rad)), 1e-6)
    max_dy = (ly / 2.0 - source_margin) / max(abs(math.sin(rad)), 1e-6)
    feasible_distance = min(distance, max_dx, max_dy)
    if feasible_distance < 0.8:
        raise RuntimeError(f"Could not place source for dims={dims}, az={azimuth_deg}")
    source = receiver + np.asarray(kemar_azimuth_to_room_offset(azimuth_deg, feasible_distance), dtype=np.float64)
    return receiver, source, float(feasible_distance)


def write_sofamyroom_setup(
    path: Path,
    output_prefix: Path,
    sofa_path: Path,
    room: RoomSpec,
    absorption: np.ndarray,
    source_xyz: np.ndarray,
    receiver_xyz: np.ndarray,
    fs: int,
    response_duration: float,
    reflection_order: int,
    simulate_diffuse: bool,
    number_of_rays: int,
) -> None:
    lx, ly, lz = room.dims
    absorption_rows = ";\n                                 ".join(
        " ".join(f"{v:.6f}" for v in row) for row in absorption
    )
    diffusion = np.full((6, len(FREQ_BANDS)), 0.30, dtype=np.float64)
    diffusion_rows = ";\n                                 ".join(
        " ".join(f"{v:.6f}" for v in row) for row in diffusion
    )
    text = f"""room.dimension              = [ {lx:.6f} {ly:.6f} {lz:.6f} ];
room.humidity               = 0.42;
room.temperature            = 20;

room.surface.frequency      = [ {' '.join(str(v) for v in FREQ_BANDS)} ];
room.surface.absorption     = [ {absorption_rows} ];
room.surface.diffusion      = [ {diffusion_rows} ];

options.fs                  = {int(fs)};
options.responseduration    = {float(response_duration):.6f};
options.bandsperoctave      = 1;
options.referencefrequency  = 125;
options.airabsorption       = true;
options.distanceattenuation = true;
options.subsampleaccuracy   = false;
options.highpasscutoff      = 0;
options.verbose             = false;

options.simulatespecular    = true;
options.reflectionorder     = [ {reflection_order} {reflection_order} {reflection_order} ];

options.simulatediffuse     = {"true" if simulate_diffuse else "false"};
options.numberofrays        = {int(number_of_rays)};
options.diffusetimestep     = 0.010;
options.rayenergyfloordB    = -80;
options.uncorrelatednoise   = true;

options.outputname          = '{output_prefix}';
options.mex_saveaswav       = false;

source(1).location          = [ {source_xyz[0]:.6f} {source_xyz[1]:.6f} {source_xyz[2]:.6f} ];
source(1).orientation       = [ 0 0 0 ];
source(1).description       = 'omnidirectional';

receiver(1).location        = [ {receiver_xyz[0]:.6f} {receiver_xyz[1]:.6f} {receiver_xyz[2]:.6f} ];
receiver(1).orientation     = [ 0 0 0 ];
receiver(1).description     = 'SOFA {sofa_path} interp=1 norm=1 resampling=1';
"""
    path.write_text(text, encoding="utf-8")


def estimate_rt60_from_ir(ir: np.ndarray, sample_rate: int) -> float:
    if ir.size == 0 or np.max(np.abs(ir)) < 1e-10:
        return float("nan")
    edc = np.cumsum(np.square(ir[::-1].astype(np.float64)))[::-1]
    edc_db = 10.0 * np.log10(np.maximum(edc / max(float(np.max(edc)), 1e-12), 1e-12))
    times = np.arange(len(ir), dtype=np.float64) / float(sample_rate)
    mask = (edc_db <= -5.0) & (edc_db >= -35.0)
    if mask.sum() < 8:
        return float("nan")
    slope, _ = np.polyfit(times[mask], edc_db[mask], deg=1)
    if slope >= -1e-9:
        return float("nan")
    return float(-60.0 / slope)


def resample_nd(audio: np.ndarray, orig_sr: int, target_sr: int, axis: int = 0) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    g = math.gcd(int(orig_sr), int(target_sr))
    return resample_poly(audio, target_sr // g, orig_sr // g, axis=axis).astype(np.float32, copy=False)


def render_reverberant_speech(speech: np.ndarray, brir: np.ndarray) -> np.ndarray:
    left = fftconvolve(speech, brir[:, 0], mode="full")[: len(speech)]
    right = fftconvolve(speech, brir[:, 1], mode="full")[: len(speech)]
    return np.stack([left, right], axis=1).astype(np.float32)


def peak_normalize_with_gain(stereo: np.ndarray, peak: float) -> Tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(stereo)))
    if max_abs > peak and max_abs > 1e-8:
        gain = peak / max_abs
        return (stereo * gain).astype(np.float32, copy=False), float(gain)
    return stereo.astype(np.float32, copy=False), 1.0


def generate_diffuse_binaural_noise(
    mono_noise: np.ndarray,
    sample_rate: int,
    ear_spacing_m: float,
    nfft: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate 2-ch spherical diffuse noise from a mono base signal via ANF."""
    mono_noise = np.asarray(mono_noise, dtype=np.float64)
    if mono_noise.ndim != 1:
        raise ValueError(f"Expected mono noise, got shape={mono_noise.shape}")
    if len(mono_noise) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # ANF expects M mutually independent input signals for M output channels.
    # Use decorrelated variants derived from the same scene excerpt.
    noise_a = mono_noise - float(np.mean(mono_noise))
    max_shift = max(1, min(len(noise_a) // 8, int(0.02 * sample_rate)))
    shift = int(rng.integers(1, max_shift + 1))
    noise_b = np.roll(noise_a, shift)
    jitter = 1e-4 * rng.standard_normal(len(noise_a))
    inputs = np.stack([noise_a, noise_b + jitter], axis=0)

    half = float(ear_spacing_m) / 2.0
    mic_positions = np.asarray([
        [-half, 0.0, 0.0],
        [half, 0.0, 0.0],
    ], dtype=np.float64)
    params = ANFParameters(
        mic_positions=mic_positions,
        sc_type="spherical",
        sample_frequency=sample_rate,
        nfft=int(nfft),
    )
    outputs, _coherence_target, _mixing_matrix = generate_signals(
        inputs,
        params,
        decomposition="evd",
        processing="balance+smooth",
    )
    return outputs.T.astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_root", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--num_samples", type=int, default=72)
    p.add_argument("--librispeech_root", type=Path, default=Path("/disk2/bywang/data/LibriSpeech/train-clean-100"))
    p.add_argument("--demand_root", type=Path, default=Path("/disk2/bywang/data/demand"))
    p.add_argument("--sofamyroom_bin", type=Path, default=DEFAULT_SOFAMYROOM_BIN)
    p.add_argument("--sofa_path", type=Path, default=DEFAULT_KEMAR_SOFA)
    p.add_argument("--conda_lib", type=Path, default=DEFAULT_CONDA_LIB)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--brir_fs", type=int, default=48000)
    p.add_argument("--duration_sec", type=float, default=2.0)
    p.add_argument("--brir_duration_sec", type=float, default=1.2)
    p.add_argument("--reflection_order", type=int, default=10)
    p.add_argument("--number_of_rays", type=int, default=2000)
    p.add_argument(
        "--absorption_scale",
        type=float,
        default=None,
        help="Override profile-specific Eyring absorption scale before writing SofaMyRoom setup.",
    )
    p.add_argument("--no_diffuse", action="store_true")
    p.add_argument("--clean_only", action="store_true")
    p.add_argument(
        "--save_mode",
        choices=["full", "train_minimal"],
        default="full",
        help="full saves all intermediate artifacts; train_minimal keeps only binaural wavs and metadata.",
    )
    p.add_argument(
        "--test_grid",
        action="store_true",
        help="For test split, use a room x distance x SNR x azimuth grid. Use --num_samples 0 for the full grid.",
    )
    p.add_argument(
        "--test_snr_values",
        type=str,
        default=None,
        help=(
            "Comma-separated SNR list for --split test --test_grid. "
            "Use 'clean' for the clean bucket, e.g. clean,10,5,0,-5,-10,-15. "
            "If omitted, uses the default clean,10,5,0,-5,-10 protocol."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep_setup", action="store_true")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument(
        "--noise_mode",
        choices=["postmix", "diffusefg"],
        default="postmix",
        help="Noise renderer: postmix uses raw scene channels; diffusefg uses ANF spherical diffuse noise from mono scene noise.",
    )
    p.add_argument("--anf_nfft", type=int, default=512)
    p.add_argument("--ear_spacing_m", type=float, default=0.18)
    return p.parse_args()


def parse_test_snr_values(text: Optional[str]) -> Sequence[Optional[float]]:
    if text is None:
        return TEST_SNR_VALUES
    values: List[Optional[float]] = []
    for item in text.split(","):
        token = item.strip()
        if not token:
            continue
        if token.lower() == "clean":
            values.append(None)
        else:
            values.append(float(token))
    if not values:
        raise ValueError("--test_snr_values must contain at least one value")
    return tuple(values)


def build_test_grid(snr_values: Sequence[Optional[float]] = TEST_SNR_VALUES) -> List[Tuple[RoomSpec, float, Optional[float], int]]:
    cases: List[Tuple[RoomSpec, float, Optional[float], int]] = []
    for room in TEST_ROOMS:
        for distance in test_distances(room.profile):
            for snr_db in snr_values:
                for doa_class in range(72):
                    cases.append((room, float(distance), snr_db, doa_class))
    return cases


def main() -> None:
    args = parse_args()
    test_snr_values = parse_test_snr_values(args.test_snr_values)
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)

    wav_dir = args.output_root / args.split / "binaural"
    clean_dir = args.output_root / args.split / "clean_reverb"
    brir_dir = args.output_root / args.split / "brir"
    setup_dir = args.output_root / args.split / "sofamyroom_setup"
    for d in (wav_dir, clean_dir, brir_dir, setup_dir):
        d.mkdir(parents=True, exist_ok=True)

    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    speech_files = list_librispeech_files(args.librispeech_root)
    noise_scenes = TEST_NOISE_SCENES if args.split == "test" else TRAIN_NOISE_SCENES
    noise_files = list_noise_files(args.demand_root, noise_scenes)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{args.conda_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    num_model_samples = int(round(args.duration_sec * args.sample_rate))
    num_brir_samples = int(round(args.duration_sec * args.brir_fs))
    test_cases = build_test_grid(test_snr_values) if args.split == "test" and args.test_grid else None
    if test_cases is not None:
        if args.num_samples > 0:
            test_cases = test_cases[: args.num_samples]
        args.num_samples = len(test_cases)
    class_schedule = make_balanced_shuffled_bins(args.num_samples, 72, np_rng)
    report_path = args.output_root / args.split / "metadata.csv"

    fieldnames = [
        "file_id", "split", "kemar_azimuth_deg", "azimuth_deg", "doa_class", "elevation_deg",
        "room_size", "room_id", "room_dims_m", "target_rt60", "estimated_rt60",
        "absorption_mean", "absorption_scale", "receiver_xyz", "source_xyz", "source_distance_m",
        "receiver_wall_clearance_m", "source_wall_clearance_m", "sofamyroom_setup_path",
        "brir_path", "clean_reverb_path", "wav_path", "speech_path", "noise_scene",
        "noise_path_l", "noise_path_r", "noise_type", "noise_field_model", "noise_render_method",
        "snr_db", "sample_rate", "brir_fs",
        "duration_sec", "brir_duration_sec", "reflection_order", "simulate_diffuse",
    ]

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(1, args.num_samples + 1):
            forced_distance: Optional[float] = None
            forced_snr: Optional[float] = None
            if test_cases is not None:
                room, forced_distance, forced_snr, doa_class = test_cases[idx - 1]
            else:
                doa_class = int(class_schedule[idx - 1])
                room = TEST_ROOMS[(idx - 1) % len(TEST_ROOMS)] if args.split == "test" else sample_train_room(py_rng)
            kemar_azimuth = float(doa_class * 5)
            azimuth_deg = kemar_to_compatible_azimuth(kemar_azimuth)
            receiver_xyz, source_xyz, distance = choose_geometry(
                room.dims, room.profile, kemar_azimuth, py_rng, args.split, idx, forced_distance
            )
            absorption_scale = (
                default_absorption_scale(room.profile)
                if args.absorption_scale is None
                else float(args.absorption_scale)
            )
            alpha = eyring_absorption_for_rt60(room.dims, room.target_rt60, absorption_scale)
            absorption = make_absorption_matrix(alpha)

            file_id = f"kemar_{args.split}_{idx:06d}"
            setup_path = setup_dir / f"{file_id}.txt"
            output_prefix = brir_dir / file_id
            write_sofamyroom_setup(
                setup_path,
                output_prefix,
                args.sofa_path,
                room,
                absorption,
                source_xyz,
                receiver_xyz,
                args.brir_fs,
                args.brir_duration_sec,
                args.reflection_order,
                not args.no_diffuse,
                args.number_of_rays,
            )
            subprocess.run([str(args.sofamyroom_bin), str(setup_path)], check=True, env=env)
            brir_wav_path = brir_dir / f"{file_id}_receiver_0.wav"
            brir, brir_sr = sf.read(str(brir_wav_path), always_2d=True, dtype="float32")
            if brir_sr != args.brir_fs or brir.shape[1] != 2:
                raise RuntimeError(f"Bad BRIR output {brir_wav_path}: sr={brir_sr}, shape={brir.shape}")
            brir_npy_path = brir_dir / f"{file_id}.npy"
            if args.save_mode == "full":
                np.save(brir_npy_path, brir.astype(np.float32))

            speech_path = speech_files[int(np_rng.integers(0, len(speech_files)))]
            speech = fit_to_length(load_mono_resampled(speech_path, args.brir_fs), num_brir_samples, np_rng)
            clean_reverb_48k = render_reverberant_speech(speech, brir)
            clean_reverb = resample_nd(clean_reverb_48k, args.brir_fs, args.sample_rate, axis=0)[:num_model_samples]
            clean_reverb = peak_normalize(clean_reverb, peak=0.90)

            noise_scene: Optional[str] = None
            noise_l_path: Optional[Path] = None
            noise_r_path: Optional[Path] = None
            noise_type = "clean"
            noise_field_model = ""
            noise_render_method = ""
            snr_db: Optional[float] = None
            mixed = clean_reverb
            if not args.clean_only:
                if forced_snr is not None or (args.split == "test" and test_cases is not None):
                    snr_db = forced_snr
                elif args.split == "test":
                    snr_db = TEST_SNR_VALUES[(idx - 1) % len(TEST_SNR_VALUES)]
                else:
                    snr_db = py_rng.uniform(-10.0, 10.0)
                if snr_db is not None:
                    noise_scene = py_rng.choice(noise_scenes)
                    scene_files = noise_files[noise_scene]
                    if args.noise_mode == "diffusefg":
                        noise_l_path = scene_files[int(np_rng.integers(0, len(scene_files)))]
                        mono_noise = read_noise_segment(noise_l_path, num_model_samples, args.sample_rate, py_rng)
                        noise = generate_diffuse_binaural_noise(
                            mono_noise=mono_noise,
                            sample_rate=args.sample_rate,
                            ear_spacing_m=args.ear_spacing_m,
                            nfft=args.anf_nfft,
                            rng=np_rng,
                        )
                        noise_type = "diffuse_field_generator"
                        noise_field_model = "spherical"
                        noise_render_method = "anf_generator_python"
                    else:
                        noise_l_path = scene_files[int(np_rng.integers(0, len(scene_files)))]
                        noise_r_path = scene_files[int(np_rng.integers(0, len(scene_files)))]
                        noise_l = read_noise_segment(noise_l_path, num_model_samples, args.sample_rate, py_rng)
                        noise_r = read_noise_segment(noise_r_path, num_model_samples, args.sample_rate, py_rng)
                        noise = np.stack([noise_l, noise_r], axis=1)
                        noise_type = "additive_binaural_postmix"
                    mixed = mix_at_snr(clean_reverb, noise, float(snr_db))
                    mixed, mix_gain = peak_normalize_with_gain(mixed, peak=0.95)
                    clean_reverb = (clean_reverb * mix_gain).astype(np.float32, copy=False)

            clean_path = clean_dir / f"{file_id}.npy"
            wav_path = wav_dir / f"{file_id}.wav"
            if args.save_mode == "full":
                np.save(clean_path, clean_reverb.astype(np.float32))
            sf.write(str(wav_path), mixed, args.sample_rate, subtype="PCM_16")

            mono_brir = 0.5 * (brir[:, 0] + brir[:, 1])
            estimated_rt60 = estimate_rt60_from_ir(mono_brir, args.brir_fs)
            if args.save_mode == "train_minimal":
                brir_wav_path.unlink(missing_ok=True)
                setup_path.unlink(missing_ok=True)
            row = {
                "file_id": file_id,
                "split": args.split,
                "kemar_azimuth_deg": f"{kemar_azimuth:.6f}",
                "azimuth_deg": f"{azimuth_deg:.6f}",
                "doa_class": doa_class,
                "elevation_deg": "0.000000",
                "room_size": room.profile,
                "room_id": room.room_id,
                "room_dims_m": f"{room.dims[0]:.3f}x{room.dims[1]:.3f}x{room.dims[2]:.3f}",
                "target_rt60": f"{room.target_rt60:.6f}",
                "estimated_rt60": f"{estimated_rt60:.6f}",
                "absorption_mean": f"{float(np.mean(absorption)):.6f}",
                "absorption_scale": f"{absorption_scale:.6f}",
                "receiver_xyz": f"{receiver_xyz[0]:.6f},{receiver_xyz[1]:.6f},{receiver_xyz[2]:.6f}",
                "source_xyz": f"{source_xyz[0]:.6f},{source_xyz[1]:.6f},{source_xyz[2]:.6f}",
                "source_distance_m": f"{distance:.6f}",
                "receiver_wall_clearance_m": f"{horizontal_wall_clearance(receiver_xyz, room.dims):.6f}",
                "source_wall_clearance_m": f"{horizontal_wall_clearance(source_xyz, room.dims):.6f}",
                "sofamyroom_setup_path": "" if args.save_mode == "train_minimal" else str(setup_path),
                "brir_path": "" if args.save_mode == "train_minimal" else str(brir_npy_path),
                "clean_reverb_path": "" if args.save_mode == "train_minimal" else str(clean_path),
                "wav_path": str(wav_path),
                "speech_path": str(speech_path),
                "noise_scene": noise_scene or "clean",
                "noise_path_l": "" if noise_l_path is None else str(noise_l_path),
                "noise_path_r": "" if noise_r_path is None else str(noise_r_path),
                "noise_type": noise_type,
                "noise_field_model": noise_field_model,
                "noise_render_method": noise_render_method,
                "snr_db": "clean" if snr_db is None else f"{float(snr_db):.6f}",
                "sample_rate": args.sample_rate,
                "brir_fs": args.brir_fs,
                "duration_sec": f"{args.duration_sec:.6f}",
                "brir_duration_sec": f"{args.brir_duration_sec:.6f}",
                "reflection_order": args.reflection_order,
                "simulate_diffuse": int(not args.no_diffuse),
            }
            writer.writerow(row)
            f.flush()
            if not args.keep_setup:
                # Keep the path in metadata stable only when explicitly requested.
                pass
            if idx == 1 or idx % args.log_interval == 0:
                print(f"[{idx}/{args.num_samples}] wrote {file_id} az={kemar_azimuth:.1f} room={room.room_id}/{room.profile}")

    if args.save_mode == "train_minimal":
        for d in (clean_dir, brir_dir, setup_dir):
            try:
                d.rmdir()
            except OSError:
                pass

    print(f"Done: {report_path}")


if __name__ == "__main__":
    main()
