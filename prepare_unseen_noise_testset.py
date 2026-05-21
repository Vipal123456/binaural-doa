#!/usr/bin/env python3
"""Generate a test-only unseen-noise split on top of an existing static dataset.

This script reuses the original subject split and generation protocol from
``prepare_robust_multisubject_dataset.py`` but renders only the unseen-subject
test split with a held-out set of DEMAND noise scenes.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from prepare_robust_multisubject_dataset import (
    HRTFSubject,
    ensure_empty_dir,
    fit_to_length,
    generate_split,
    list_librispeech_files,
    list_noise_files,
    quality_check_root,
)


DEFAULT_HELDOUT_SCENES = [
    "DKITCHEN",
    "NFIELD",
    "OHALLWAY",
    "PSTATION",
    "STRAFFIC",
    "TCAR",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a test-only unseen-noise split")
    parser.add_argument(
        "--base_dataset_root",
        type=Path,
        default=Path("/disk2/bywang/DOA-net/data/librispeech_cipic_multisubject_robust50h_v1"),
        help="Existing robust dataset root containing manifest.json",
    )
    parser.add_argument(
        "--output_dir_name",
        type=str,
        default="test_subjects_unseen_noiseheldout",
        help="Name of the generated test-only split directory under base_dataset_root",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=DEFAULT_HELDOUT_SCENES,
        help="Held-out DEMAND scenes to use for this test-only split",
    )
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.base_dataset_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Base manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_subjects = list(manifest["split"]["test_subjects_unseen"])
    test_recordings = int(manifest["counts"]["test_recordings"])
    if len(test_subjects) == 0:
        raise ValueError("No test subjects found in base manifest")
    if test_recordings % len(test_subjects) != 0:
        raise ValueError("Test recordings count is not divisible by number of test subjects")
    recordings_per_subject = test_recordings // len(test_subjects)

    output_root = args.base_dataset_root / args.output_dir_name
    if output_root.exists():
        if not args.resume:
            if not args.overwrite:
                raise FileExistsError(f"{output_root} already exists; pass --overwrite or --resume")
            ensure_empty_dir(output_root, overwrite=True)
    else:
        output_root.mkdir(parents=True, exist_ok=True)

    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    librispeech_root = Path(manifest["speech_root"])
    hrtf_root = Path(manifest["hrtf_root"])
    demand_root = Path(manifest["demand_root"])
    sample_rate = int(manifest["sample_rate"])
    duration_sec = float(manifest["duration_sec"])
    num_classes = int(manifest["num_classes"])
    source_distance_min = 1.0
    source_distance_max = 1.5

    speech_files = list_librispeech_files(librispeech_root)
    noise_files = list_noise_files(demand_root, args.scenes)
    hrtf_cache = {
        sid: HRTFSubject(hrtf_root / f"subject_{sid}.sofa", sample_rate)
        for sid in sorted(set(test_subjects))
    }

    class Args:
        pass

    gen_args = Args()
    gen_args.output_root = args.base_dataset_root
    gen_args.resume = args.resume
    gen_args.overwrite = args.overwrite
    gen_args.sample_rate = sample_rate
    gen_args.duration_sec = duration_sec
    gen_args.source_distance_min = source_distance_min
    gen_args.source_distance_max = source_distance_max
    gen_args.num_classes = num_classes
    gen_args.recordings_per_subject = recordings_per_subject
    gen_args.scenes = args.scenes
    gen_args.log_interval = args.log_interval

    # generate_split writes to args.output_root / split_dir_name
    generate_split(
        split_name="test_noiseheldout",
        split_dir_name=args.output_dir_name,
        subject_ids=test_subjects,
        args=gen_args,
        speech_files=speech_files,
        noise_files=noise_files,
        hrtf_cache=hrtf_cache,
        py_rng=py_rng,
        np_rng=np_rng,
    )

    split_root = args.base_dataset_root / args.output_dir_name
    qc = quality_check_root(split_root)
    (split_root / "quality_report.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")

    info = {
        "base_dataset_root": str(args.base_dataset_root),
        "base_manifest": str(manifest_path),
        "output_split": args.output_dir_name,
        "seed": args.seed,
        "heldout_scenes": list(args.scenes),
        "test_subjects_unseen": test_subjects,
        "recordings_per_subject": recordings_per_subject,
        "quality_report": qc,
    }
    (split_root / "generation_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
