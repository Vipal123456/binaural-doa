#!/usr/bin/env python3
"""Check LibriSpeech speech-path/speaker overlap across KEMAR metadata files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metadata",
        nargs="+",
        required=True,
        help="Items in name=path form, e.g. train=data/.../metadata.csv",
    )
    p.add_argument("--output_json", type=Path)
    return p.parse_args()


def parse_item(item: str) -> tuple[str, Path]:
    if "=" not in item:
        raise ValueError(f"Expected name=path, got: {item}")
    name, path = item.split("=", 1)
    if not name:
        raise ValueError(f"Empty split name in: {item}")
    return name, Path(path)


def speaker_id(speech_path: str) -> str:
    parts = Path(speech_path).parts
    return parts[-3] if len(parts) >= 3 else ""


def chapter_id(speech_path: str) -> str:
    parts = Path(speech_path).parts
    return "/".join(parts[-3:-1]) if len(parts) >= 3 else ""


def read_speech_paths(path: Path) -> List[str]:
    with path.open(newline="", encoding="utf-8") as f:
        return [row["speech_path"] for row in csv.DictReader(f)]


def main() -> None:
    args = parse_args()
    named_paths = [parse_item(item) for item in args.metadata]

    paths: Dict[str, List[str]] = {}
    for name, meta_path in named_paths:
        paths[name] = read_speech_paths(meta_path)

    unique_paths: Dict[str, Set[str]] = {name: set(values) for name, values in paths.items()}
    speakers: Dict[str, Set[str]] = {
        name: {speaker_id(path) for path in values} for name, values in unique_paths.items()
    }
    chapters: Dict[str, Set[str]] = {
        name: {chapter_id(path) for path in values} for name, values in unique_paths.items()
    }

    summary = {
        "splits": {
            name: {
                "num_rows": len(values),
                "num_unique_speech_paths": len(unique_paths[name]),
                "num_duplicate_rows_within_split": len(values) - len(unique_paths[name]),
                "num_unique_speakers": len(speakers[name]),
                "num_unique_chapters": len(chapters[name]),
            }
            for name, values in paths.items()
        },
        "pairwise_overlap": {},
    }

    names = [name for name, _ in named_paths]
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            path_overlap = unique_paths[left] & unique_paths[right]
            speaker_overlap = speakers[left] & speakers[right]
            chapter_overlap = chapters[left] & chapters[right]
            key = f"{left}__{right}"
            summary["pairwise_overlap"][key] = {
                "num_speech_path_overlap": len(path_overlap),
                f"speech_path_overlap_rate_vs_{left}": (
                    len(path_overlap) / max(len(unique_paths[left]), 1)
                ),
                f"speech_path_overlap_rate_vs_{right}": (
                    len(path_overlap) / max(len(unique_paths[right]), 1)
                ),
                "num_speaker_overlap": len(speaker_overlap),
                "num_chapter_overlap": len(chapter_overlap),
            }

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
