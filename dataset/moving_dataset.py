"""Moving single-speaker binaural DOA sequence dataset."""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .feature_extractor import FeatureExtractor

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None


class MovingDOADataset(Dataset):
    """Load pre-rendered moving binaural samples with 100 ms DOA labels."""

    def __init__(
        self,
        root_dir: str,
        sample_rate: int = 16000,
        segment_seconds: float = 4.0,
        label_steps: int = 40,
        num_classes: int = 72,
        azimuth_range: Tuple[float, float] = (-180.0, 180.0),
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        window: str = "hann",
        audio_subdir: str = "binaural_dev",
        metadata_subdir: str = "metadata_dev",
        label_source: str = "target",
        logger: Any = None,
    ):
        if sf is None:
            raise ImportError("soundfile is required to load MovingDOADataset")
        self.root_dir = Path(root_dir)
        self.sample_rate = int(sample_rate)
        self.segment_seconds = float(segment_seconds)
        self.label_steps = int(label_steps)
        self.num_classes = int(num_classes)
        self.azimuth_range = tuple(azimuth_range)
        self.audio_dir = self.root_dir / audio_subdir
        self.metadata_dir = self.root_dir / metadata_subdir
        if label_source not in {"target", "rendered"}:
            raise ValueError(f"Unsupported label_source: {label_source}")
        self.label_source = label_source
        self.logger = logger
        self.feature_extractor = FeatureExtractor(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
        )
        self.records = self._scan()

    def _scan(self):
        if not self.audio_dir.is_dir():
            raise ValueError(f"Audio directory not found: {self.audio_dir}")
        if not self.metadata_dir.is_dir():
            raise ValueError(f"Metadata directory not found: {self.metadata_dir}")
        records = []
        for meta_path in sorted(self.metadata_dir.glob("metadata*.json")):
            file_id = meta_path.stem.replace("metadata", "")
            audio_path = self.audio_dir / f"binaural{file_id}.wav"
            if audio_path.is_file():
                records.append((file_id, audio_path, meta_path))
        if self.logger is not None:
            self.logger.info(f"Loaded moving index: {len(records)} samples from {self.root_dir}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_id, audio_path, meta_path = self.records[idx]
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        if sr != self.sample_rate:
            raise ValueError(f"Unexpected sample rate {sr} for {audio_path}; expected {self.sample_rate}")
        audio = audio.T
        target_len = int(round(self.segment_seconds * self.sample_rate))
        if audio.shape[1] < target_len:
            audio = np.pad(audio, ((0, 0), (0, target_len - audio.shape[1])), mode="constant")
        audio = audio[:2, :target_len]

        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        target_angle_seq = np.asarray(meta.get("target_angle_seq", meta["angle_seq"]), dtype=np.float32)
        target_label_seq = np.asarray(meta.get("target_label_seq", meta["doa_labels"]), dtype=np.int64)
        rendered_angle_seq = np.asarray(meta.get("rendered_angle_seq", target_angle_seq), dtype=np.float32)
        rendered_label_seq = np.asarray(meta.get("rendered_label_seq", target_label_seq), dtype=np.int64)
        if self.label_source == "rendered":
            angle_seq = rendered_angle_seq
            label_seq = rendered_label_seq
        else:
            angle_seq = target_angle_seq
            label_seq = target_label_seq
        if (
            len(angle_seq) != self.label_steps
            or len(label_seq) != self.label_steps
            or len(target_angle_seq) != self.label_steps
            or len(rendered_angle_seq) != self.label_steps
        ):
            raise ValueError(f"{meta_path} has invalid label length")

        feats = self.feature_extractor.extract(torch.from_numpy(audio).float())
        return {
            "file_id": file_id,
            "log_mag_L": feats["log_mag_L"],
            "log_mag_R": feats["log_mag_R"],
            "spec_real_L": feats["spec_real_L"],
            "spec_imag_L": feats["spec_imag_L"],
            "spec_real_R": feats["spec_real_R"],
            "spec_imag_R": feats["spec_imag_R"],
            "ipd": feats["ipd"],
            "ild": feats["ild"],
            "ipd_sin": feats["ipd_sin"],
            "ipd_cos": feats["ipd_cos"],
            "coherence": feats["coherence"],
            "doa_labels": torch.from_numpy(label_seq),
            "doa_angles": torch.from_numpy(angle_seq),
            "target_labels": torch.from_numpy(target_label_seq),
            "target_angles": torch.from_numpy(target_angle_seq),
            "rendered_labels": torch.from_numpy(rendered_label_seq),
            "rendered_angles": torch.from_numpy(rendered_angle_seq),
            "label_source": self.label_source,
            "trajectory_type": meta.get("trajectory_type", "unknown"),
            "speed": float(meta.get("speed", 0.0)),
            "speed_bin": meta.get("speed_bin", "unknown"),
            "distance": float(meta.get("distance", 0.0)),
            "rt60": float(meta.get("rt60", 0.0)),
            "snr": float(meta.get("snr", 999.0)),
            "condition": meta.get("condition", "unknown"),
        }


def build_moving_datasets(cfg, logger=None):
    ds = cfg.dataset
    m = cfg.model
    f = cfg.feature
    common = dict(
        sample_rate=ds.sample_rate,
        segment_seconds=ds.get("segment_seconds", 4.0),
        label_steps=ds.get("label_steps", 40),
        num_classes=m.num_classes,
        azimuth_range=tuple(m.azimuth_range),
        n_fft=f.n_fft,
        hop_length=f.hop_length,
        win_length=f.win_length,
        window=f.window,
        audio_subdir=ds.get("audio_subdir", "binaural_dev"),
        metadata_subdir=ds.get("metadata_subdir", "metadata_dev"),
        label_source=ds.get("label_source", "target"),
        logger=logger,
    )
    return (
        MovingDOADataset(root_dir=ds.train_root, **common),
        MovingDOADataset(root_dir=ds.val_root, **common),
        MovingDOADataset(root_dir=ds.test_root, **common),
    )
