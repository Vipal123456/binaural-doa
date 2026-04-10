#!/usr/bin/env python3
"""Prepare multi-subject LibriSpeech+CIPIC datasets for training.

This script:
1. Selects subjects from available SOFA files.
2. Synthesizes per-subject datasets (if missing) by calling synthesize_librispeech_cipic.py.
3. Merges per-subject datasets into subject-disjoint roots with symlinks:
   - train_subjects
   - val_subjects
   - test_subjects_unseen
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


def list_subject_ids(hrtf_root: Path) -> List[str]:
    sofa_files = sorted(hrtf_root.glob("subject_*.sofa"))
    subject_ids = [p.stem.replace("subject_", "") for p in sofa_files]
    if not subject_ids:
        raise FileNotFoundError(f"No subject_*.sofa found in {hrtf_root}")
    return subject_ids


def pick_subjects(
    all_subjects: List[str],
    total_subjects: int,
    train_subjects: int,
    val_subjects: int,
    test_subjects: int,
    seed: int,
    force_include: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    if train_subjects + val_subjects + test_subjects != total_subjects:
        raise ValueError("train+val+test subject counts must equal total_subjects")
    if len(all_subjects) < total_subjects:
        raise ValueError(f"Need {total_subjects} subjects, but found only {len(all_subjects)}")

    force_include = [s for s in force_include if s in all_subjects]
    if len(set(force_include)) > total_subjects:
        raise ValueError("Too many forced subjects")

    remaining = [s for s in all_subjects if s not in force_include]
    rng = random.Random(seed)
    rng.shuffle(remaining)

    chosen = force_include + remaining[: total_subjects - len(force_include)]

    # Keep deterministic and readable ordering in outputs.
    chosen = sorted(chosen)
    train = chosen[:train_subjects]
    val = chosen[train_subjects : train_subjects + val_subjects]
    test = chosen[train_subjects + val_subjects :]
    return train, val, test


def run_synthesis(
    python_exec: str,
    repo_root: Path,
    librispeech_root: Path,
    sofa_path: Path,
    output_root: Path,
    num_recordings: int,
    sample_rate: int,
    duration_sec: float,
    seed: int,
) -> None:
    cmd = [
        python_exec,
        str(repo_root / "synthesize_librispeech_cipic.py"),
        "--librispeech_root",
        str(librispeech_root),
        "--sofa_path",
        str(sofa_path),
        "--output_root",
        str(output_root),
        "--num_recordings",
        str(num_recordings),
        "--sample_rate",
        str(sample_rate),
        "--duration_sec",
        str(duration_sec),
        "--seed",
        str(seed),
    ]
    subprocess.run(cmd, check=True)


def ensure_subject_dataset(
    python_exec: str,
    repo_root: Path,
    librispeech_root: Path,
    hrtf_root: Path,
    data_root: Path,
    subject_id: str,
    per_subject_recordings: int,
    sample_rate: int,
    duration_sec: float,
    synth_seed: int,
    force_regen: bool,
) -> Path:
    out_root = data_root / f"librispeech_cipic_subject{subject_id}"
    binaural_dir = out_root / "binaural_dev"
    metadata_dir = out_root / "metadata_dev"

    have_enough = (
        binaural_dir.is_dir()
        and metadata_dir.is_dir()
        and len(list(binaural_dir.glob("binaural*.wav"))) >= per_subject_recordings
        and len(list(metadata_dir.glob("metadata*.csv"))) >= per_subject_recordings
    )

    if force_regen and out_root.exists():
        shutil.rmtree(out_root)
        have_enough = False

    if not have_enough:
        sofa_path = hrtf_root / f"subject_{subject_id}.sofa"
        if not sofa_path.is_file():
            raise FileNotFoundError(f"Missing SOFA: {sofa_path}")
        run_synthesis(
            python_exec=python_exec,
            repo_root=repo_root,
            librispeech_root=librispeech_root,
            sofa_path=sofa_path,
            output_root=out_root,
            num_recordings=per_subject_recordings,
            sample_rate=sample_rate,
            duration_sec=duration_sec,
            seed=synth_seed,
        )

    return out_root


def merge_subjects_with_symlinks(
    subject_ids: List[str],
    data_root: Path,
    merged_root: Path,
    per_subject_limit: int,
) -> int:
    if merged_root.exists():
        shutil.rmtree(merged_root)
    (merged_root / "binaural_dev").mkdir(parents=True, exist_ok=True)
    (merged_root / "metadata_dev").mkdir(parents=True, exist_ok=True)

    total = 0
    for sid in subject_ids:
        src_root = data_root / f"librispeech_cipic_subject{sid}"
        src_binaural = src_root / "binaural_dev"
        src_metadata = src_root / "metadata_dev"

        wavs = sorted(src_binaural.glob("binaural*.wav"))[:per_subject_limit]
        for wav in wavs:
            fid = wav.stem.replace("binaural", "")
            meta = src_metadata / f"metadata{fid}.csv"
            if not meta.is_file():
                continue

            new_id = f"{sid}_{fid}"
            dst_wav = merged_root / "binaural_dev" / f"binaural{new_id}.wav"
            dst_meta = merged_root / "metadata_dev" / f"metadata{new_id}.csv"

            dst_wav.symlink_to(wav.resolve())
            dst_meta.symlink_to(meta.resolve())
            total += 1

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare multi-subject LibriSpeech+CIPIC data")
    parser.add_argument("--repo_root", type=Path, default=Path("/disk2/bywang/DOA-net"))
    parser.add_argument("--librispeech_root", type=Path, default=Path("/disk2/bywang/data/LibriSpeech/train-clean-100"))
    parser.add_argument("--hrtf_root", type=Path, default=Path("/disk2/bywang/data/HRTF"))
    parser.add_argument("--data_root", type=Path, default=Path("/disk2/bywang/DOA-net/data"))
    parser.add_argument("--python_exec", type=str, default="/home/bywang/miniconda3/envs/doa/bin/python")

    parser.add_argument("--total_subjects", type=int, default=12)
    parser.add_argument("--train_subjects", type=int, default=8)
    parser.add_argument("--val_subjects", type=int, default=2)
    parser.add_argument("--test_subjects", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_include", nargs="*", default=["003"])

    parser.add_argument("--train_recordings_per_subject", type=int, default=3000)
    parser.add_argument("--val_recordings_per_subject", type=int, default=1500)
    parser.add_argument("--test_recordings_per_subject", type=int, default=1500)
    parser.add_argument("--synth_recordings_per_subject", type=int, default=3000)

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration_sec", type=float, default=10.0)
    parser.add_argument("--synth_seed", type=int, default=42)
    parser.add_argument("--force_regen_subject", action="store_true")
    args = parser.parse_args()

    all_subjects = list_subject_ids(args.hrtf_root)
    train_s, val_s, test_s = pick_subjects(
        all_subjects=all_subjects,
        total_subjects=args.total_subjects,
        train_subjects=args.train_subjects,
        val_subjects=args.val_subjects,
        test_subjects=args.test_subjects,
        seed=args.seed,
        force_include=args.force_include,
    )

    split_info: Dict[str, List[str]] = {
        "train_subjects": train_s,
        "val_subjects": val_s,
        "test_subjects_unseen": test_s,
    }

    print("Selected subjects:")
    print(json.dumps(split_info, indent=2))

    needed = sorted(set(train_s + val_s + test_s))
    for i, sid in enumerate(needed, start=1):
        print(f"[{i}/{len(needed)}] Ensuring subject {sid} dataset...")
        ensure_subject_dataset(
            python_exec=args.python_exec,
            repo_root=args.repo_root,
            librispeech_root=args.librispeech_root,
            hrtf_root=args.hrtf_root,
            data_root=args.data_root,
            subject_id=sid,
            per_subject_recordings=args.synth_recordings_per_subject,
            sample_rate=args.sample_rate,
            duration_sec=args.duration_sec,
            synth_seed=args.synth_seed,
            force_regen=args.force_regen_subject,
        )

    merged_base = args.data_root / "librispeech_cipic_multisubject"
    train_root = merged_base / "train_subjects"
    val_root = merged_base / "val_subjects"
    test_root = merged_base / "test_subjects_unseen"

    print("Merging train subjects...")
    n_train = merge_subjects_with_symlinks(
        subject_ids=train_s,
        data_root=args.data_root,
        merged_root=train_root,
        per_subject_limit=args.train_recordings_per_subject,
    )
    print("Merging val subjects...")
    n_val = merge_subjects_with_symlinks(
        subject_ids=val_s,
        data_root=args.data_root,
        merged_root=val_root,
        per_subject_limit=args.val_recordings_per_subject,
    )
    print("Merging unseen test subjects...")
    n_test = merge_subjects_with_symlinks(
        subject_ids=test_s,
        data_root=args.data_root,
        merged_root=test_root,
        per_subject_limit=args.test_recordings_per_subject,
    )

    manifest = {
        "split": split_info,
        "counts": {
            "train_merged_recordings": n_train,
            "val_merged_recordings": n_val,
            "test_unseen_merged_recordings": n_test,
        },
        "paths": {
            "train_root": str(train_root),
            "val_root": str(val_root),
            "test_unseen_root": str(test_root),
        },
    }
    merged_base.mkdir(parents=True, exist_ok=True)
    manifest_path = merged_base / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Done.")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
