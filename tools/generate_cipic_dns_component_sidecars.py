#!/usr/bin/env python3
"""Reconstruct aligned target/interferer stems for directional DNS mixtures."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Mapping

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_cipic_roomsim25 import (
    _load_brir_cached,
    joint_normalize,
    render_context,
    resample_nd,
)
from tools.generate_cipic_roomsim25_directional_dns_v4 import (
    _load_noise_cached,
    mix_at_active_sir,
)


SAMPLE_RATE = 16000
OUTPUT_SAMPLES = 32000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["val"])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verify_limit", type=int, default=32)
    return parser.parse_args()


def _speech_context(row: Mapping[str, str], prefix: int) -> np.ndarray:
    audio, source_sr = sf.read(row["speech_path"], dtype="float32", always_2d=True)
    mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
    mono = resample_nd(mono, int(source_sr), SAMPLE_RATE)
    mono -= float(np.mean(mono))
    start = int(row["speech_target_start_sample"])
    context_start = start - prefix
    if context_start < 0:
        context = np.pad(mono[: start + OUTPUT_SAMPLES], (prefix - start, 0))
    else:
        context = mono[context_start : start + OUTPUT_SAMPLES]
    expected = prefix + OUTPUT_SAMPLES
    if len(context) < expected:
        context = np.pad(context, (0, expected - len(context)))
    return np.asarray(context[:expected], dtype=np.float32)


def reconstruct_components(row: Mapping[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_brir = _load_brir_cached(row["target_brir_path"], SAMPLE_RATE)
    noise_brir = _load_brir_cached(row["noise_brir_path"], SAMPLE_RATE)
    prefix = max(len(target_brir), len(noise_brir)) - 1

    speech = _speech_context(row, prefix)
    clean = render_context(speech, target_brir, prefix, OUTPUT_SAMPLES)

    noise = _load_noise_cached(row["noise_content_path"], SAMPLE_RATE)
    noise_start = int(round(float(row["noise_content_start_sec"]) * SAMPLE_RATE))
    if noise_start < prefix or noise_start + OUTPUT_SAMPLES > len(noise):
        raise RuntimeError(
            f"Noise context out of range for {row['file_id']}: "
            f"start={noise_start}, prefix={prefix}, length={len(noise)}"
        )
    noise_context = np.asarray(
        noise[noise_start - prefix : noise_start + OUTPUT_SAMPLES],
        dtype=np.float32,
    )
    noise_context -= float(np.mean(noise_context))
    directional = render_context(noise_context, noise_brir, prefix, OUTPUT_SAMPLES)
    raw_mix, _achieved_sir, _activity = mix_at_active_sir(
        clean,
        directional,
        float(row["target_sir_db"]),
        SAMPLE_RATE,
    )
    normalized_mix = joint_normalize(raw_mix)

    denominator = float(np.sum(raw_mix.astype(np.float64) ** 2))
    if denominator <= 1.0e-12:
        raise RuntimeError(f"Silent reconstructed mixture: {row['file_id']}")
    scale = float(
        np.sum(raw_mix.astype(np.float64) * normalized_mix.astype(np.float64))
        / denominator
    )
    return clean * scale, (raw_mix - clean) * scale, normalized_mix


def _render_task(payload: tuple[dict[str, str], str, bool]) -> tuple[str, float]:
    row, split_root_string, verify = payload
    split_root = Path(split_root_string)
    file_id = row["file_id"]
    target_path = split_root / "components" / "target" / f"{file_id}.wav"
    interferer_path = split_root / "components" / "interferer" / f"{file_id}.wav"
    if target_path.is_file() and interferer_path.is_file() and not verify:
        return file_id, float("nan")

    target, interferer, reconstructed_mix = reconstruct_components(row)
    if not target_path.is_file():
        sf.write(target_path, target, SAMPLE_RATE, subtype="PCM_16")
    if not interferer_path.is_file():
        sf.write(interferer_path, interferer, SAMPLE_RATE, subtype="PCM_16")

    max_error = float("nan")
    if verify:
        mixture, source_sr = sf.read(
            split_root / row["wav_path"], dtype="float32", always_2d=True
        )
        if int(source_sr) != SAMPLE_RATE or mixture.shape != reconstructed_mix.shape:
            raise RuntimeError(f"Mixture shape/rate mismatch: {file_id}")
        max_error = float(np.max(np.abs(mixture - reconstructed_mix)))
        if max_error > 5.0e-5:
            raise RuntimeError(
                f"Reconstruction mismatch for {file_id}: max_abs_error={max_error}"
            )
    return file_id, max_error


def generate_split(
    split_root: Path,
    workers: int,
    limit: int,
    verify_limit: int,
) -> None:
    with (split_root / "metadata.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if limit > 0:
        rows = rows[:limit]
    for component in ("target", "interferer"):
        (split_root / "components" / component).mkdir(parents=True, exist_ok=True)

    tasks = [
        (dict(row), str(split_root), index < verify_limit)
        for index, row in enumerate(rows)
    ]
    started = time.time()
    verified_errors: list[float] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        for index, (_file_id, error) in enumerate(
            executor.map(_render_task, tasks, chunksize=4), start=1
        ):
            if np.isfinite(error):
                verified_errors.append(error)
            if index == 1 or index % 500 == 0 or index == len(tasks):
                elapsed = max(time.time() - started, 1.0e-6)
                print(
                    f"[{split_root.name}] {index}/{len(tasks)} "
                    f"rate={index / elapsed:.2f} sample/s",
                    flush=True,
                )
    print(
        f"[{split_root.name}] complete samples={len(tasks)} "
        f"verified={len(verified_errors)} "
        f"max_reconstruction_error={max(verified_errors, default=float('nan'))}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    for split in args.splits:
        split_root = args.dataset_root / split
        if not (split_root / "metadata.csv").is_file():
            raise FileNotFoundError(split_root / "metadata.csv")
        generate_split(split_root, args.workers, args.limit, args.verify_limit)


if __name__ == "__main__":
    main()
