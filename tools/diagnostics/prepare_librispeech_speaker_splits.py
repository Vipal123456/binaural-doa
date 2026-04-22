#!/usr/bin/env python3
"""Prepare LibriSpeech speaker-overlap/disjoint splits using symlinks.

The script creates three roots under output_root:
- train_speakers
- test_overlap_speakers (subset of train speakers)
- test_disjoint_speakers (disjoint from train speakers)

Each root mirrors LibriSpeech speaker/chapter folder layout via symlinks.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List


def list_speakers(librispeech_root: Path) -> List[str]:
    speakers = sorted([p.name for p in librispeech_root.iterdir() if p.is_dir()])
    if not speakers:
        raise FileNotFoundError(f"No speaker folders found in {librispeech_root}")
    return speakers


def make_split(
    speakers: List[str],
    train_speakers: int,
    overlap_eval_speakers: int,
    disjoint_eval_speakers: int,
    seed: int,
) -> Dict[str, List[str]]:
    need = train_speakers + disjoint_eval_speakers
    if need > len(speakers):
        raise ValueError(
            f"Need at least {need} speakers, but only {len(speakers)} available"
        )
    if overlap_eval_speakers > train_speakers:
        raise ValueError("overlap_eval_speakers must be <= train_speakers")

    rng = random.Random(seed)
    shuffled = speakers[:]
    rng.shuffle(shuffled)

    train = sorted(shuffled[:train_speakers])
    remaining = shuffled[train_speakers:]
    disjoint = sorted(remaining[:disjoint_eval_speakers])

    overlap_pool = train[:]
    rng.shuffle(overlap_pool)
    overlap = sorted(overlap_pool[:overlap_eval_speakers])

    return {
        "train_speakers": train,
        "test_overlap_speakers": overlap,
        "test_disjoint_speakers": disjoint,
    }


def symlink_split(librispeech_root: Path, dst_root: Path, speaker_ids: List[str]) -> int:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    chapters = 0
    for sid in speaker_ids:
        src_spk = librispeech_root / sid
        if not src_spk.is_dir():
            continue
        dst_spk = dst_root / sid
        dst_spk.mkdir(parents=True, exist_ok=True)

        for chap in sorted([p for p in src_spk.iterdir() if p.is_dir()]):
            dst_chap = dst_spk / chap.name
            dst_chap.mkdir(parents=True, exist_ok=True)

            # Use file-level symlinks to keep pathlib.rglob("*.flac") working.
            for src_file in sorted([p for p in chap.iterdir() if p.is_file()]):
                (dst_chap / src_file.name).symlink_to(src_file.resolve())

            chapters += 1

    return chapters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare LibriSpeech speaker splits")
    p.add_argument(
        "--librispeech_root",
        type=Path,
        default=Path("/disk2/bywang/data/LibriSpeech/train-clean-100"),
    )
    p.add_argument(
        "--output_root",
        type=Path,
        default=Path("/disk2/bywang/DOA-net/data/librispeech_speaker_splits"),
    )
    p.add_argument("--train_speakers", type=int, default=180)
    p.add_argument("--overlap_eval_speakers", type=int, default=30)
    p.add_argument("--disjoint_eval_speakers", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    speakers = list_speakers(args.librispeech_root)
    split = make_split(
        speakers=speakers,
        train_speakers=args.train_speakers,
        overlap_eval_speakers=args.overlap_eval_speakers,
        disjoint_eval_speakers=args.disjoint_eval_speakers,
        seed=args.seed,
    )

    train_root = args.output_root / "train_speakers"
    overlap_root = args.output_root / "test_overlap_speakers"
    disjoint_root = args.output_root / "test_disjoint_speakers"

    n_train_ch = symlink_split(args.librispeech_root, train_root, split["train_speakers"])
    n_overlap_ch = symlink_split(args.librispeech_root, overlap_root, split["test_overlap_speakers"])
    n_disjoint_ch = symlink_split(args.librispeech_root, disjoint_root, split["test_disjoint_speakers"])

    manifest = {
        "seed": args.seed,
        "librispeech_root": str(args.librispeech_root),
        "output_root": str(args.output_root),
        "num_all_speakers": len(speakers),
        "num_train_speakers": len(split["train_speakers"]),
        "num_overlap_eval_speakers": len(split["test_overlap_speakers"]),
        "num_disjoint_eval_speakers": len(split["test_disjoint_speakers"]),
        "num_train_chapters": n_train_ch,
        "num_overlap_eval_chapters": n_overlap_ch,
        "num_disjoint_eval_chapters": n_disjoint_ch,
        "splits": split,
        "roots": {
            "train_speakers": str(train_root),
            "test_overlap_speakers": str(overlap_root),
            "test_disjoint_speakers": str(disjoint_root),
        },
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
