#!/usr/bin/env python3
"""Generate a validated Neidhardt measured-BRIR external test set.

The v2 protocol fixes the label, crop, paired-noise, and metadata issues in the
legacy v1 generator. It never writes into the v1 output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import soundfile as sf
from netCDF4 import Dataset
from scipy.signal import fftconvolve, resample_poly


ROOT = Path("/disk2/bywang/DOA-net")
BRIR_DIR = Path("/disk2/bywang/data/neidhardt_brir")
SPEECH_DIR = Path("/disk2/bywang/data/LibriSpeech/LibriSpeech_test/test-clean")
DEMAND_DIR = Path("/disk2/bywang/data/demand")
DEFAULT_OUTPUT_DIR = ROOT / "data" / "librispeech_neidhardt_measured_brir_test_v2"

DEMAND_SCENES = ["OOFFICE", "PCAFETER", "TMETRO", "TBUS", "SPSQUARE", "NPARK"]
TRAIN_SEEN_SCENES = {"OOFFICE", "PCAFETER", "TMETRO"}
SNR_LEVELS: Sequence[str | int] = ("clean", -10, -5, 0, 5, 10)
NUM_CLASSES = 72
SAMPLE_RATE = 16000
SEGMENT_SECONDS = 2.0
CROPS_PER_BRIR = 2
SEED = 42
MIN_SPEECH_SECONDS = 4.5
ZENODO_DOI = "10.5281/zenodo.2593714"
ZENODO_ZIP_MD5 = "a2b5b8dbd55707f7acc16d2e381af081"


def wrap_deg(angle: float) -> float:
    return ((float(angle) + 180.0) % 360.0) - 180.0


def circular_error_deg(a: float, b: float) -> float:
    return abs(wrap_deg(float(a) - float(b)))


def nearest_grid_label(angle_deg: float) -> int:
    """Map a measured angle to the nearest fixed 5-degree DOA class."""
    wrapped = wrap_deg(angle_deg)
    return int(np.rint((wrapped + 180.0) / 5.0)) % NUM_CLASSES


def label_to_angle(label: int) -> float:
    return -180.0 + 5.0 * int(label)


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resample_1d(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(signal, dtype=np.float32)
    divisor = math.gcd(int(source_rate), int(target_rate))
    return resample_poly(
        signal,
        target_rate // divisor,
        source_rate // divisor,
    ).astype(np.float32)


def load_mono(path: Path, target_rate: int) -> np.ndarray:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return resample_1d(np.asarray(audio, dtype=np.float32), int(source_rate), target_rate)


def peak_normalize(stereo: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_abs = float(np.max(np.abs(stereo)))
    if max_abs > peak and max_abs > 1e-8:
        stereo = stereo * (peak / max_abs)
    return stereo.astype(np.float32, copy=False)


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    signal = signal.astype(np.float32, copy=False)
    noise = noise.astype(np.float32, copy=False)
    noise = noise - np.mean(noise, axis=0, keepdims=True)
    signal_power = float(np.mean(signal.astype(np.float64) ** 2))
    noise_power = float(np.mean(noise.astype(np.float64) ** 2))
    if signal_power < 1e-12 or noise_power < 1e-12:
        raise ValueError("Cannot mix a silent signal or noise segment")
    scale = math.sqrt(signal_power / ((10.0 ** (float(snr_db) / 10.0)) * noise_power))
    return (signal + scale * noise).astype(np.float32, copy=False)


def load_paired_noise(
    scene_files: Dict[str, List[Path]],
    scene: str,
    length: int,
    rng: random.Random,
) -> Tuple[np.ndarray, Dict[str, object]]:
    files = scene_files[scene]
    if len(files) < 2:
        raise ValueError(f"Scene {scene} needs at least two DEMAND channels")
    ch0_path, ch1_path = rng.sample(files, 2)
    ch0_info = sf.info(ch0_path)
    ch1_info = sf.info(ch1_path)
    if ch0_info.samplerate != SAMPLE_RATE or ch1_info.samplerate != SAMPLE_RATE:
        raise ValueError(
            f"DEMAND channels must be {SAMPLE_RATE} Hz: "
            f"{ch0_path}={ch0_info.samplerate}, {ch1_path}={ch1_info.samplerate}"
        )
    available = min(ch0_info.frames, ch1_info.frames)
    if available < length:
        raise ValueError(f"Noise files in {scene} are shorter than {length} samples")
    start = rng.randint(0, available - length)
    ch0, _ = sf.read(ch0_path, start=start, frames=length, dtype="float32", always_2d=False)
    ch1, _ = sf.read(ch1_path, start=start, frames=length, dtype="float32", always_2d=False)
    stereo = np.stack([ch0, ch1], axis=1).astype(np.float32)
    return stereo, {
        "noise_scene": scene,
        "noise_path_ch0": str(ch0_path),
        "noise_path_ch1": str(ch1_path),
        "noise_start_sample": int(start),
        "noise_scene_seen_in_training": scene in TRAIN_SEEN_SCENES,
    }


def collect_speech_files(required: int, rng: random.Random) -> List[Path]:
    eligible: List[Path] = []
    for path in sorted(SPEECH_DIR.rglob("*.flac")):
        info = sf.info(path)
        if info.frames / info.samplerate >= MIN_SPEECH_SECONDS:
            eligible.append(path)
    if len(eligible) < required:
        raise RuntimeError(f"Need {required} eligible speech files, found {len(eligible)}")
    rng.shuffle(eligible)
    return eligible[:required]


def collect_noise_files() -> Dict[str, List[Path]]:
    result: Dict[str, List[Path]] = {}
    for scene in DEMAND_SCENES:
        files = sorted((DEMAND_DIR / scene).glob("ch*.wav"))
        if len(files) < 2:
            raise FileNotFoundError(f"Expected at least two noise channels in {DEMAND_DIR / scene}")
        result[scene] = files
    return result


def source_relative_azimuth(
    source_position: np.ndarray,
    listener_position: np.ndarray,
    listener_view_azimuth: float,
) -> float:
    delta = source_position - listener_position
    source_global_azimuth = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
    return wrap_deg(source_global_azimuth - float(listener_view_azimuth))


def protocol_azimuth(measurement_index: int) -> float:
    """Map SOFA head rotation to the project's KEMAR azimuth handedness."""
    return wrap_deg(5.0 * int(measurement_index))


def speaker_orientation(room_short_name: str) -> str:
    if "towardsListener" in room_short_name:
        return "towards_listener"
    if "awayfromListener" in room_short_name:
        return "away_from_listener"
    raise ValueError(f"Unknown loudspeaker orientation in {room_short_name!r}")


def prepare_output(output_dir: Path, overwrite: bool) -> Tuple[Path, Path]:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output is not empty: {output_dir}; pass --overwrite explicitly")
        shutil.rmtree(output_dir)
    wav_dir = output_dir / "test_all" / "binaural_dev"
    meta_dir = output_dir / "test_all" / "metadata_dev"
    wav_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    return wav_dir, meta_dir


def select_protocol(args: argparse.Namespace, all_sofa_files: List[Path]) -> Tuple[List[Path], List[int]]:
    if not args.smoke:
        return all_sofa_files, list(range(72))
    wanted = {"Pos1_LS_0.sofa", "Pos5_LS_0.sofa"}
    sofa_files = [path for path in all_sofa_files if path.name in wanted]
    if len(sofa_files) != 2:
        raise RuntimeError(f"Smoke test expected {sorted(wanted)}, found {[p.name for p in sofa_files]}")
    return sofa_files, [0, 18, 36, 54]


def validate_side_cues(records: List[Dict[str, object]]) -> None:
    """Check direct-path channel cues in the measured BRIRs, before speech rendering."""
    side_measurements = {
        (str(r["sofa_file"]), int(r["measurement_index"]), float(r["grid_azimuth_deg"]))
        for r in records
        if float(r["grid_azimuth_deg"]) in {-90.0, 90.0}
    }
    if not side_measurements:
        raise AssertionError("No +/-90-degree clean samples were available for channel sanity")
    by_sofa: Dict[str, List[Tuple[int, float]]] = {}
    for sofa_file, measurement_index, angle in side_measurements:
        by_sofa.setdefault(sofa_file, []).append((measurement_index, angle))
    for sofa_file, measurements in by_sofa.items():
        with Dataset(BRIR_DIR / sofa_file, "r") as database:
            ir = database.variables["Data.IR"]
            for measurement_index, angle in measurements:
                direct_energies = []
                for channel in range(2):
                    response = np.asarray(ir[measurement_index, channel, 0, :], dtype=np.float64)
                    magnitude = np.abs(response)
                    onset = int(np.argmax(magnitude > max(float(magnitude.max()) * 0.1, 1e-7)))
                    direct_energies.append(float(np.sum(response[onset : onset + 128] ** 2)))
                if angle == 90.0 and direct_energies[1] <= direct_energies[0]:
                    raise AssertionError(f"Expected channel-1 direct-path dominance at +90 deg in {sofa_file}")
                if angle == -90.0 and direct_energies[0] <= direct_energies[1]:
                    raise AssertionError(f"Expected channel-0 direct-path dominance at -90 deg in {sofa_file}")


def validate_condition_pairing(records: List[Dict[str, object]]) -> None:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record["base_pair_id"]), []).append(record)
    expected_snr_labels = {str(level) for level in SNR_LEVELS}
    for base_pair_id, variants in grouped.items():
        if {str(v["snr_label"]) for v in variants} != expected_snr_labels:
            raise AssertionError(f"Incomplete SNR variants for {base_pair_id}")
        noisy = [v for v in variants if v["snr_label"] != "clean"]
        keys = ("noise_scene", "noise_path_ch0", "noise_path_ch1", "noise_start_sample")
        if len({tuple(v[key] for key in keys) for v in noisy}) != 1:
            raise AssertionError(f"Noisy SNR variants are not paired for {base_pair_id}")


def validate_records(records: List[Dict[str, object]], wav_dir: Path, smoke: bool) -> Dict[str, object]:
    expected = (2 * 4 if smoke else 10 * 72) * CROPS_PER_BRIR * len(SNR_LEVELS)
    if len(records) != expected:
        raise AssertionError(f"Expected {expected} records, found {len(records)}")
    if any(float(r["label_snap_error_deg"]) > 0.2 for r in records):
        raise AssertionError("A measured azimuth is more than 0.2 degrees from the 5-degree grid")
    if any(int(r["crop_start_sample_1"]) - int(r["crop_start_sample_0"]) != 2 * SAMPLE_RATE for r in records):
        raise AssertionError("Non-overlapping crop invariant failed")
    for record in records:
        path = wav_dir / f"binaural{int(record['file_id']):06d}.wav"
        info = sf.info(path)
        if info.frames != 2 * SAMPLE_RATE or info.channels != 2 or info.samplerate != SAMPLE_RATE:
            raise AssertionError(f"Invalid WAV shape: {path}: {info}")
        audio, _ = sf.read(path, dtype="float32", always_2d=True)
        if not np.isfinite(audio).all() or float(np.max(np.abs(audio))) < 1e-6:
            raise AssertionError(f"Invalid or silent WAV: {path}")
        if float(np.max(np.abs(audio))) >= 1.0:
            raise AssertionError(f"Clipped WAV: {path}")
    validate_condition_pairing(records)
    validate_side_cues(records)
    inconsistent_positions = {
        int(r["listener_position"]) for r in records if not bool(r["geometry_metadata_consistent"])
    }
    if inconsistent_positions != {5}:
        raise AssertionError(
            f"Expected only position 5 to have inconsistent SOFA geometry, got {inconsistent_positions}"
        )
    return {
        "expected_records": expected,
        "validated_records": len(records),
        "max_label_snap_error_deg": max(float(r["label_snap_error_deg"]) for r in records),
        "channel_sanity": "passed",
        "paired_noise_across_snr": "passed",
        "sofa_geometry_inconsistent_positions": sorted(inconsistent_positions),
        "crop_non_overlap": "passed",
        "wav_shape": "passed",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--log-interval", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    archive = BRIR_DIR / "BRIR_SOFAfiles.zip"
    if md5sum(archive) != ZENODO_ZIP_MD5:
        raise RuntimeError("BRIR archive checksum does not match Zenodo record 2593714")

    all_sofa_files = sorted(BRIR_DIR.glob("Pos*_LS_*.sofa"))
    if len(all_sofa_files) != 10:
        raise RuntimeError(f"Expected 10 SOFA files, found {len(all_sofa_files)}")
    sofa_files, measurement_indices = select_protocol(args, all_sofa_files)
    total_brirs = len(sofa_files) * len(measurement_indices)
    speech_files = collect_speech_files(total_brirs, rng)
    noise_files = collect_noise_files()
    wav_dir, meta_dir = prepare_output(args.output_dir, args.overwrite)

    records: List[Dict[str, object]] = []
    file_id = 0
    crop_samples = int(SAMPLE_RATE * SEGMENT_SECONDS)
    brir_index = 0

    for sofa_index, sofa_path in enumerate(sofa_files):
        with Dataset(sofa_path, "r") as database:
            ir_variable = database.variables["Data.IR"]
            source_rate = int(round(float(database.variables["Data.SamplingRate"][:].reshape(-1)[0])))
            source_position = np.asarray(database.variables["SourcePosition"][:]).reshape(-1, 3)[0]
            listener_position = np.asarray(database.variables["ListenerPosition"][:]).reshape(-1, 3)[0]
            listener_views = np.asarray(database.variables["ListenerView"][:])[:, 0]
            room_short_name = str(getattr(database, "RoomShortName", ""))
            orientation = speaker_orientation(room_short_name)
            position = int(sofa_path.stem.split("_")[0].replace("Pos", ""))

            for measurement_index in measurement_indices:
                speech_path = speech_files[brir_index]
                speech = load_mono(speech_path, SAMPLE_RATE)
                if len(speech) < 4 * SAMPLE_RATE:
                    raise AssertionError(f"Speech shorter than 4 seconds after filtering: {speech_path}")

                # Preserve SOFA receiver order to match the KEMAR training waveforms.
                brir_ch0 = resample_1d(
                    np.asarray(ir_variable[measurement_index, 0, 0, :]), source_rate, SAMPLE_RATE
                )
                brir_ch1 = resample_1d(
                    np.asarray(ir_variable[measurement_index, 1, 0, :]), source_rate, SAMPLE_RATE
                )
                rendered_ch0 = fftconvolve(speech, brir_ch0, mode="full").astype(np.float32)
                rendered_ch1 = fftconvolve(speech, brir_ch1, mode="full").astype(np.float32)
                rendered = np.stack([rendered_ch0, rendered_ch1], axis=1)

                # Select one random 4-second block inside the active speech interval, then split it exactly.
                max_block_start = len(speech) - 4 * SAMPLE_RATE
                block_start = int(np_rng.integers(0, max_block_start + 1))
                crop_starts = [block_start, block_start + crop_samples]

                project_azimuth = protocol_azimuth(measurement_index)
                sofa_geometry_azimuth = source_relative_azimuth(
                    source_position, listener_position, float(listener_views[measurement_index])
                )
                sofa_geometry_project_azimuth = wrap_deg(-sofa_geometry_azimuth)
                geometry_protocol_error = circular_error_deg(
                    sofa_geometry_project_azimuth, project_azimuth
                )
                label = nearest_grid_label(project_azimuth)
                grid_azimuth = label_to_angle(label)
                snap_error = circular_error_deg(project_azimuth, grid_azimuth)

                for crop_index, crop_start in enumerate(crop_starts):
                    clean = peak_normalize(rendered[crop_start : crop_start + crop_samples])
                    scene = DEMAND_SCENES[(brir_index * CROPS_PER_BRIR + crop_index) % len(DEMAND_SCENES)]
                    paired_noise, noise_info = load_paired_noise(
                        noise_files, scene, crop_samples, rng
                    )
                    base_pair_id = f"{sofa_path.stem}_m{measurement_index:02d}_c{crop_index}"

                    for snr in SNR_LEVELS:
                        if snr == "clean":
                            audio = clean
                            snr_db = None
                            current_noise_info = {
                                "noise_scene": "none",
                                "noise_path_ch0": "none",
                                "noise_path_ch1": "none",
                                "noise_start_sample": None,
                                "noise_scene_seen_in_training": None,
                            }
                        else:
                            audio = peak_normalize(mix_at_snr(clean, paired_noise, float(snr)))
                            snr_db = float(snr)
                            current_noise_info = noise_info

                        file_id += 1
                        record: Dict[str, object] = {
                            "file_id": file_id,
                            "base_pair_id": base_pair_id,
                            "azimuth_deg": float(grid_azimuth),
                            "project_azimuth_deg": float(project_azimuth),
                            "sofa_protocol_azimuth_deg": float(wrap_deg(-5.0 * measurement_index)),
                            "grid_azimuth_deg": float(grid_azimuth),
                            "label_snap_error_deg": float(snap_error),
                            "sofa_geometry_azimuth_deg": float(sofa_geometry_azimuth),
                            "sofa_geometry_project_azimuth_deg": float(sofa_geometry_project_azimuth),
                            "geometry_protocol_error_deg": float(geometry_protocol_error),
                            "geometry_metadata_consistent": bool(geometry_protocol_error < 0.2),
                            "doa_class": int(label),
                            "azimuth_bin": int(label),
                            "measurement_index": int(measurement_index),
                            "listener_view_azimuth_deg": float(listener_views[measurement_index]),
                            "source_position_m": source_position.tolist(),
                            "listener_position_m": listener_position.tolist(),
                            "sofa_file": sofa_path.name,
                            "listener_position": position,
                            "speaker_orientation": orientation,
                            "snr_db": snr_db,
                            "snr_label": str(snr),
                            "speech_path": str(speech_path),
                            "speech_speaker_id": speech_path.parts[-3],
                            "crop_index": crop_index,
                            "crop_start_sample": int(crop_start),
                            "crop_start_sample_0": int(crop_starts[0]),
                            "crop_start_sample_1": int(crop_starts[1]),
                            "brir_source": f"Neidhardt et al., Zenodo {ZENODO_DOI}",
                            "dummy_head": "KEMAR 45BA with small ears",
                            "room": "small conference room",
                            "t60_broadband_s": 0.63,
                            "sample_rate": SAMPLE_RATE,
                            "segment_seconds": SEGMENT_SECONDS,
                            "rendering_mode": "speech_convolved_with_measured_brir",
                            **current_noise_info,
                        }
                        sf.write(
                            wav_dir / f"binaural{file_id:06d}.wav",
                            audio,
                            SAMPLE_RATE,
                            subtype="PCM_16",
                        )
                        (meta_dir / f"metadata{file_id:06d}.json").write_text(
                            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        records.append(record)

                brir_index += 1
                if brir_index % args.log_interval == 0 or brir_index == total_brirs:
                    print(
                        f"BRIR {brir_index}/{total_brirs}; segments {file_id}/{total_brirs * 12}",
                        flush=True,
                    )

    report_path = args.output_dir / "test_all" / "mixing_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    validation = validate_records(records, wav_dir, args.smoke)
    manifest = {
        "dataset": "librispeech_neidhardt_measured_brir_test_v2",
        "protocol_version": 2,
        "evaluation_scope": "external generalization from simulated BRIRs to measured-room BRIRs",
        "recording_nature": "synthetic speech rendered by convolution with measured BRIRs; not in-room speech recordings",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "tools/generate_neidhardt_brir_testset_v2.py",
        "seed": args.seed,
        "smoke": bool(args.smoke),
        "speech_source": str(SPEECH_DIR),
        "speech_min_duration_seconds": MIN_SPEECH_SECONDS,
        "speech_unique_per_brir": True,
        "brir_source": f"Neidhardt et al., Zenodo {ZENODO_DOI}",
        "brir_archive_md5": ZENODO_ZIP_MD5,
        "zenodo_license": "CC BY 4.0",
        "sofa_embedded_license": "CC BY-NC-SA 4.0",
        "license_note": "License fields conflict; use the stricter terms for redistribution until clarified.",
        "dummy_head": "KEMAR 45BA with small ears",
        "room": "small conference room",
        "t60_broadband_s": 0.63,
        "sample_rate": SAMPLE_RATE,
        "segment_seconds": SEGMENT_SECONDS,
        "num_classes": NUM_CLASSES,
        "snr_levels": [str(level) for level in SNR_LEVELS],
        "noise_scenes": DEMAND_SCENES,
        "paired_noise_across_snr": True,
        "crops_per_brir": CROPS_PER_BRIR,
        "crop_non_overlap": True,
        "total_brirs": total_brirs,
        "total_segments": len(records),
        "sofa_files": [path.name for path in sofa_files],
        "measurement_indices": measurement_indices,
        "label_mapping": "project azimuth = wrap(+5 * measurement_index); SOFA azimuth handedness is mirrored to match the KEMAR training convention",
        "handedness_validation": "At +/-90 deg, receiver-channel energy dominance matches the official KEMAR test set after sign conversion",
        "sofa_geometry_note": "SOFA geometry is converted by project_azimuth = -sofa_azimuth. Position 5 still implies a 180-degree offset; both values are recorded for audit and geometry is not used as the label.",
        "class_angle_mapping": "angle = -180 + 5 * class",
        "checkpoint_policy": "select checkpoints on the KEMAR simulated validation set only; no selection or tuning on this external test set",
        "channel_mapping": "output ch0=SOFA receiver0; output ch1=SOFA receiver1, matching KEMAR training channel indices",
        "validation": validation,
    }
    (args.output_dir / "test_all" / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    print(f"Output: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
