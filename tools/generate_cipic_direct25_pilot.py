#!/usr/bin/env python3
"""Generate a small subject-disjoint CIPIC direct-HRIR DOA dataset.

This pilot deliberately contains no room response, additive noise, or HRTF
interpolation.  It tests one question only: can a model trained on multiple
CIPIC subjects classify measured frontal directions for unseen CIPIC subjects?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import sofa
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly


CLASS_ANGLES_DEG = [
    -80, -65, -55, -45, -40, -35, -30, -25, -20, -15, -10, -5,
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 55, 65, 80,
]

# Subject-disjoint protocol used by the DP-RTF study's CIPIC experiment.
SUBJECT_SPLITS = {
    "train": [
        "058", "059", "060", "048", "050", "051", "010", "028", "124",
        "011", "012", "165", "147", "148", "152", "044", "127", "156",
        "015", "017", "018", "134", "135", "137", "158", "162", "163",
        "153", "154", "155",
    ],
    "val": ["061", "065", "119", "126", "131", "133"],
    "test": ["021", "003", "040", "008", "009", "033", "019", "020", "027"],
}


def wrap_deg(angle: float) -> float:
    return ((float(angle) + 180.0) % 360.0) - 180.0


def circular_error_deg(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs((a - float(b) + 180.0) % 360.0 - 180.0)


def resample_1d(x: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return np.asarray(x, dtype=np.float32)
    divisor = math.gcd(int(source_sr), int(target_sr))
    return resample_poly(
        x,
        target_sr // divisor,
        source_sr // divisor,
    ).astype(np.float32, copy=False)


def list_audio_files(root: Path) -> List[Path]:
    files = sorted(root.rglob("*.flac")) + sorted(root.rglob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No FLAC/WAV speech found under {root}")
    return files


def speaker_id(path: Path) -> str:
    # LibriSpeech layout is speaker/chapter/utterance.flac.
    return path.parent.parent.name


def load_active_segment(
    path: Path,
    target_sr: int,
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    audio, source_sr = sf.read(path, dtype="float32", always_2d=True)
    mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
    mono = resample_1d(mono, int(source_sr), target_sr)
    mono = mono - float(np.mean(mono))

    if len(mono) < length:
        mono = np.pad(mono, (0, length - len(mono)))
        return mono.astype(np.float32, copy=False)
    if len(mono) == length:
        return mono.astype(np.float32, copy=False)

    # Select randomly among high-energy windows instead of accepting a silent crop.
    max_start = len(mono) - length
    starts = np.unique(np.linspace(0, max_start, num=min(32, max_start + 1), dtype=np.int64))
    powers = np.asarray([
        np.mean(mono[start : start + length].astype(np.float64) ** 2)
        for start in starts
    ])
    threshold = np.quantile(powers, 0.6)
    candidates = starts[powers >= threshold]
    start = int(rng.choice(candidates if len(candidates) else starts))
    return mono[start : start + length].astype(np.float32, copy=False)


def joint_level_normalize(stereo: np.ndarray, target_rms: float = 0.08) -> np.ndarray:
    """Apply one scalar to both ears, preserving ILD and interaural phase."""
    rms = float(np.sqrt(np.mean(stereo.astype(np.float64) ** 2) + 1e-12))
    if rms > 1e-7:
        stereo = stereo * (target_rms / rms)
    peak = float(np.max(np.abs(stereo)))
    if peak > 0.98:
        stereo = stereo * (0.98 / peak)
    return stereo.astype(np.float32, copy=False)


class CIPICSubject:
    def __init__(self, path: Path, target_sr: int):
        self.path = path
        database = sofa.Database.open(str(path))
        positions = np.asarray(database.Source.Position.get_values(), dtype=np.float64)
        ir = np.asarray(database.Data.IR.get_values(), dtype=np.float32)
        source_sr = int(round(float(database.Data.SamplingRate.get_values()[0])))
        if source_sr != target_sr:
            divisor = math.gcd(source_sr, target_sr)
            ir = resample_poly(
                ir,
                target_sr // divisor,
                source_sr // divisor,
                axis=-1,
            ).astype(np.float32, copy=False)

        self.ir = ir
        self.sofa_azimuths = np.asarray([wrap_deg(v) for v in positions[:, 0]])
        self.elevations = positions[:, 1]

    def measured_frontal_hrir(self, project_angle_deg: float) -> Tuple[int, np.ndarray, float, float]:
        # This repository defines positive azimuth toward the listener's right;
        # the local CIPIC SOFA conversion uses the opposite sign convention.
        target_sofa_az = wrap_deg(-project_angle_deg)
        errors = circular_error_deg(self.sofa_azimuths, target_sofa_az)
        horizontal = np.isclose(self.elevations, 0.0, atol=1e-7)
        matches = np.flatnonzero(horizontal & np.isclose(errors, 0.0, atol=1e-7))
        if len(matches) != 1:
            raise ValueError(
                f"Expected one exact horizontal measurement for {project_angle_deg} deg "
                f"in {self.path}, found {len(matches)}"
            )
        index = int(matches[0])
        return index, self.ir[index], float(self.sofa_azimuths[index]), float(self.elevations[index])


def choose_source_files(files: Sequence[Path], count: int, seed: int) -> List[Path]:
    rng = random.Random(seed)
    by_speaker: Dict[str, List[Path]] = {}
    for path in files:
        by_speaker.setdefault(speaker_id(path), []).append(path)
    speakers = sorted(by_speaker)
    rng.shuffle(speakers)
    for paths in by_speaker.values():
        rng.shuffle(paths)

    chosen: List[Path] = []
    cursor = {spk: 0 for spk in speakers}
    while len(chosen) < count:
        progressed = False
        for spk in speakers:
            idx = cursor[spk]
            if idx < len(by_speaker[spk]):
                chosen.append(by_speaker[spk][idx])
                cursor[spk] += 1
                progressed = True
                if len(chosen) == count:
                    break
        if not progressed:
            raise ValueError(f"Requested {count} source files, only found {len(chosen)}")
    return chosen


def render_split(
    split_name: str,
    subject_ids: Sequence[str],
    speech_root: Path,
    utterances_per_subject: int,
    hrtf_root: Path,
    output_root: Path,
    sample_rate: int,
    duration_sec: float,
    seed: int,
) -> Dict[str, object]:
    split_root = output_root / ({"train": "train_subjects", "val": "val_subjects", "test": "test_subjects_unseen"}[split_name])
    wav_root = split_root / "binaural"
    wav_root.mkdir(parents=True, exist_ok=False)
    metadata_path = split_root / "metadata.csv"
    num_samples = int(round(sample_rate * duration_sec))

    source_files = choose_source_files(
        list_audio_files(speech_root),
        len(subject_ids) * utterances_per_subject,
        seed,
    )
    rows = []
    class_counts: Counter = Counter()
    subject_counts: Counter = Counter()
    used_speakers = set()
    source_cursor = 0

    for subject_offset, subject_id in enumerate(subject_ids):
        subject = CIPICSubject(hrtf_root / f"subject_{subject_id}.sofa", sample_rate)
        for utterance_index in range(utterances_per_subject):
            source_path = source_files[source_cursor]
            source_cursor += 1
            crop_rng = np.random.default_rng(seed * 1_000_003 + source_cursor)
            speech = load_active_segment(source_path, sample_rate, num_samples, crop_rng)
            used_speakers.add(speaker_id(source_path))

            for class_index, angle_deg in enumerate(CLASS_ANGLES_DEG):
                measurement_index, hrir, sofa_az, sofa_el = subject.measured_frontal_hrir(angle_deg)
                left = fftconvolve(speech, hrir[0], mode="full")[:num_samples]
                right = fftconvolve(speech, hrir[1], mode="full")[:num_samples]
                stereo = joint_level_normalize(np.stack([left, right], axis=1))

                file_id = f"{split_name}_s{subject_id}_u{utterance_index:02d}_c{class_index:02d}"
                relative_wav = Path("binaural") / f"{file_id}.wav"
                sf.write(split_root / relative_wav, stereo, sample_rate, subtype="PCM_16")
                rows.append({
                    "file_id": file_id,
                    "wav_path": str(relative_wav),
                    "azimuth_deg": angle_deg,
                    "class_index": class_index,
                    "subject_id": subject_id,
                    "source_path": str(source_path),
                    "source_speaker_id": speaker_id(source_path),
                    "sofa_measurement_index": measurement_index,
                    "sofa_azimuth_deg": f"{sofa_az:.6f}",
                    "sofa_elevation_deg": f"{sofa_el:.6f}",
                })
                class_counts[class_index] += 1
                subject_counts[subject_id] += 1

        print(
            f"[{split_name}] subject {subject_offset + 1}/{len(subject_ids)}: {subject_id}",
            flush=True,
        )

    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    expected_per_class = len(subject_ids) * utterances_per_subject
    if set(class_counts.values()) != {expected_per_class}:
        raise RuntimeError(f"Unbalanced classes in {split_name}: {dict(class_counts)}")

    return {
        "num_clips": len(rows),
        "subjects": list(subject_ids),
        "num_source_speakers": len(used_speakers),
        "source_speakers": sorted(used_speakers),
        "clips_per_class": expected_per_class,
        "clips_per_subject": dict(subject_counts),
    }


def check_disjoint(values: Iterable[Sequence[str]], name: str) -> None:
    groups = [set(group) for group in values]
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            overlap = groups[i] & groups[j]
            if overlap:
                raise RuntimeError(f"{name} leakage between split {i} and {j}: {sorted(overlap)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hrtf_root", type=Path, default=Path("/disk2/bywang/data/HRTF"))
    parser.add_argument("--train_speech_root", type=Path, default=Path("/disk2/bywang/data/LibriSpeech/train-clean-100"))
    parser.add_argument("--val_speech_root", type=Path, default=Path("/disk2/bywang/data/LibriSpeech/LibriSpeech_dev/dev-clean"))
    parser.add_argument("--test_speech_root", type=Path, default=Path("/disk2/bywang/data/LibriSpeech/LibriSpeech_test/test-clean"))
    parser.add_argument("--output_root", type=Path, default=Path("data/librispeech_cipic_direct25_pilot_v1"))
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration_sec", type=float, default=2.0)
    parser.add_argument("--train_utterances_per_subject", type=int, default=5)
    parser.add_argument("--val_utterances_per_subject", type=int, default=2)
    parser.add_argument("--test_utterances_per_subject", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use 1 utterance per subject in every split for a loader/training smoke test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(
            f"Output already exists: {args.output_root}. Choose a new path to avoid mixing datasets."
        )

    for split in SUBJECT_SPLITS.values():
        for subject_id in split:
            path = args.hrtf_root / f"subject_{subject_id}.sofa"
            if not path.is_file():
                raise FileNotFoundError(path)
    check_disjoint(SUBJECT_SPLITS.values(), "CIPIC subject")

    utterance_counts = {
        "train": 1 if args.tiny else args.train_utterances_per_subject,
        "val": 1 if args.tiny else args.val_utterances_per_subject,
        "test": 1 if args.tiny else args.test_utterances_per_subject,
    }
    speech_roots = {
        "train": args.train_speech_root,
        "val": args.val_speech_root,
        "test": args.test_speech_root,
    }

    args.output_root.mkdir(parents=True, exist_ok=False)
    reports = {}
    for split_index, split_name in enumerate(("train", "val", "test")):
        reports[split_name] = render_split(
            split_name=split_name,
            subject_ids=SUBJECT_SPLITS[split_name],
            speech_root=speech_roots[split_name],
            utterances_per_subject=utterance_counts[split_name],
            hrtf_root=args.hrtf_root,
            output_root=args.output_root,
            sample_rate=args.sample_rate,
            duration_sec=args.duration_sec,
            seed=args.seed + split_index * 10_000,
        )

    check_disjoint(
        [reports[name]["source_speakers"] for name in ("train", "val", "test")],
        "speech speaker",
    )
    manifest = {
        "name": "librispeech_cipic_direct25_pilot_v1",
        "purpose": "CIPIC unseen-subject frontal-direction pilot",
        "rendering": "anechoic direct HRIR convolution",
        "hrtf_interpolation": False,
        "room_reverberation": False,
        "additive_noise": False,
        "sample_rate": args.sample_rate,
        "duration_sec": args.duration_sec,
        "class_angles_deg": CLASS_ANGLES_DEG,
        "positive_azimuth": "listener right (project convention)",
        "normalization": "one joint scalar for both ears",
        "speech_roots": {key: str(value) for key, value in speech_roots.items()},
        "splits": reports,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: value["num_clips"] for key, value in reports.items()}, indent=2))
    print(f"Done: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
