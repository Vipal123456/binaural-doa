"""LOCATA Task 1 adapter for the project's static binaural models."""

from __future__ import annotations

import glob
import hashlib
import math
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from .feature_extractor import FeatureExtractor


@dataclass(frozen=True)
class LocataRecording:
    recording_id: str
    directory: str
    audio_path: str
    vad_path: str
    source_position_path: str
    array_position_path: str
    native_sample_rate: int
    num_frames: int
    azimuth_deg: float
    elevation_deg: float
    azimuth_std_deg: float


@dataclass(frozen=True)
class LocataSegment:
    recording_id: str
    audio_path: str
    native_sample_rate: int
    start_frame: int
    num_frames: int
    start_sec: float
    duration_sec: float
    vad_ratio: float
    azimuth_deg: float
    elevation_deg: float


def _wrap_deg(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + 180.0) % 360.0 - 180.0


def _circular_error_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(_wrap_deg(np.asarray(a) - np.asarray(b)))


def _read_position_table(path: str) -> np.ndarray:
    rows = np.genfromtxt(path, delimiter="\t", names=True, dtype=np.float64)
    return np.atleast_1d(rows)


def _read_binary_vad(path: str) -> np.ndarray:
    text = Path(path).read_text(encoding="ascii")
    _, separator, values = text.partition("\n")
    if not separator:
        raise ValueError(f"Invalid LOCATA VAD file: {path}")
    vad = np.fromstring(values, dtype=np.float32, sep="\n")
    if vad.size == 0:
        raise ValueError(f"Empty LOCATA VAD file: {path}")
    return vad >= 0.5


def _task1_local_doa(
    array_position_path: str,
    source_position_path: str,
) -> Tuple[float, float, float]:
    """Return KEMAR-compatible azimuth, elevation, and azimuth spread.

    LOCATA stores row-vector rotations such that
    ``local = (source_world - array_world) @ rotation``.  For the dummy head,
    local x points to the listener's right and local y points forward.  The
    project's KEMAR convention is 0 degrees front and +90 degrees right.
    """

    array = _read_position_table(array_position_path)
    source = _read_position_table(source_position_path)
    n_rows = min(len(array), len(source))
    if n_rows == 0:
        raise ValueError("LOCATA position table has no rows")

    array_xyz = np.column_stack([array[name][:n_rows] for name in ("x", "y", "z")])
    source_xyz = np.column_stack([source[name][:n_rows] for name in ("x", "y", "z")])
    rotation = np.empty((n_rows, 3, 3), dtype=np.float64)
    for row in range(3):
        for col in range(3):
            rotation[:, row, col] = array[f"rotation_{row + 1}{col + 1}"][:n_rows]

    local = np.einsum("ni,nij->nj", source_xyz - array_xyz, rotation)
    azimuths = np.degrees(np.arctan2(local[:, 0], local[:, 1]))
    horizontal = np.hypot(local[:, 0], local[:, 1])
    elevations = np.degrees(np.arctan2(local[:, 2], horizontal))

    azimuth_rad = np.radians(azimuths)
    mean_azimuth = math.degrees(
        math.atan2(float(np.sin(azimuth_rad).mean()), float(np.cos(azimuth_rad).mean()))
    )
    spread = float(np.sqrt(np.mean(_circular_error_deg(azimuths, mean_azimuth) ** 2)))
    return float(_wrap_deg(mean_azimuth)), float(np.mean(elevations)), spread


def _validate_dummy_channel_geometry(array_position_path: str) -> Dict[str, float]:
    """Check that mic 1/3 form the expected left/right pair."""

    array = _read_position_table(array_position_path)
    row = array[0]
    origin = np.asarray([row[name] for name in ("x", "y", "z")], dtype=np.float64)
    rotation = np.asarray(
        [[row[f"rotation_{i + 1}{j + 1}"] for j in range(3)] for i in range(3)],
        dtype=np.float64,
    )

    local_mics = []
    for mic in (1, 3):
        world = np.asarray(
            [row[f"mic{mic}_{axis}"] for axis in ("x", "y", "z")],
            dtype=np.float64,
        )
        local_mics.append((world - origin) @ rotation)

    mic1, mic3 = local_mics
    if not (mic1[0] < 0.0 < mic3[0]):
        raise ValueError(
            "Unexpected LOCATA dummy geometry: mic 1/3 are not left/right in local x "
            f"(mic1={mic1.tolist()}, mic3={mic3.tolist()})"
        )
    return {
        "mic1_local_x_m": float(mic1[0]),
        "mic3_local_x_m": float(mic3[0]),
        "interaural_spacing_m": float(np.linalg.norm(mic3 - mic1)),
    }


class LocataTask1Dataset(Dataset):
    """Fixed-window LOCATA Task 1 dummy-head evaluation dataset."""

    def __init__(
        self,
        root_dir: str,
        split: str = "eval",
        sample_rate: int = 16000,
        segment_seconds: float = 2.0,
        hop_seconds: float = 1.0,
        min_vad_ratio: float = 0.5,
        channels: Sequence[int] = (0, 2),
        peak_normalize: bool = True,
        peak_value: float = 0.95,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        window: str = "hann",
        num_classes: int = 72,
        azimuth_range: Tuple[float, float] = (-180.0, 180.0),
        feature_cache_dir: Optional[str] = None,
    ) -> None:
        if split not in {"dev", "eval"}:
            raise ValueError(f"split must be 'dev' or 'eval', got {split}")
        if len(channels) != 2 or channels[0] == channels[1]:
            raise ValueError(f"Expected two distinct channel indices, got {channels}")
        if not 0.0 <= min_vad_ratio <= 1.0:
            raise ValueError("min_vad_ratio must be in [0, 1]")

        self.root_dir = str(Path(root_dir).resolve())
        self.split = split
        self.target_sample_rate = int(sample_rate)
        self.segment_seconds = float(segment_seconds)
        self.hop_seconds = float(hop_seconds)
        self.min_vad_ratio = float(min_vad_ratio)
        self.channels = tuple(int(channel) for channel in channels)
        self.peak_normalize = bool(peak_normalize)
        self.peak_value = float(peak_value)
        self.num_classes = int(num_classes)
        self.azimuth_range = tuple(float(value) for value in azimuth_range)
        self.feature_extractor = FeatureExtractor(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
        )
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else None
        if self.feature_cache_dir is not None:
            self.feature_cache_dir.mkdir(parents=True, exist_ok=True)

        self.recordings: List[LocataRecording] = []
        self.segments: List[LocataSegment] = []
        self.channel_geometry: Dict[str, float] = {}
        self._scan()
        if not self.segments:
            raise ValueError(
                "No LOCATA Task 1 segments passed the fixed window/VAD protocol: "
                f"split={split}, segment={segment_seconds}, hop={hop_seconds}, "
                f"min_vad_ratio={min_vad_ratio}"
            )

    def _scan(self) -> None:
        pattern = os.path.join(self.root_dir, self.split, "task1", "recording*", "dummy")
        directories = sorted(
            glob.glob(pattern),
            key=lambda path: int(Path(path).parent.name.replace("recording", "")),
        )
        if not directories:
            raise FileNotFoundError(f"No LOCATA Task 1 dummy directories matched {pattern}")

        for directory in directories:
            audio_path = os.path.join(directory, "audio_array_dummy.wav")
            vad_paths = glob.glob(os.path.join(directory, "VAD_dummy_*.txt"))
            source_paths = glob.glob(os.path.join(directory, "position_source_*.txt"))
            array_path = os.path.join(directory, "position_array_dummy.txt")
            if not os.path.isfile(audio_path) or not os.path.isfile(array_path):
                continue
            if len(vad_paths) != 1 or len(source_paths) != 1:
                raise ValueError(
                    f"Task 1 must contain one source/VAD in {directory}; "
                    f"found {len(source_paths)} source tables and {len(vad_paths)} VAD files"
                )

            info = sf.info(audio_path)
            if info.channels <= max(self.channels):
                raise ValueError(
                    f"Audio {audio_path} has {info.channels} channels, cannot select {self.channels}"
                )
            azimuth, elevation, azimuth_std = _task1_local_doa(array_path, source_paths[0])
            if azimuth_std > 0.1:
                raise ValueError(
                    f"Task 1 source is not static in {directory}: azimuth std={azimuth_std:.3f} deg"
                )
            geometry = _validate_dummy_channel_geometry(array_path)
            if not self.channel_geometry:
                self.channel_geometry = geometry

            recording_id = Path(directory).parent.name
            recording = LocataRecording(
                recording_id=recording_id,
                directory=directory,
                audio_path=audio_path,
                vad_path=vad_paths[0],
                source_position_path=source_paths[0],
                array_position_path=array_path,
                native_sample_rate=int(info.samplerate),
                num_frames=int(info.frames),
                azimuth_deg=azimuth,
                elevation_deg=elevation,
                azimuth_std_deg=azimuth_std,
            )
            self.recordings.append(recording)
            self._add_recording_segments(recording)

    def _add_recording_segments(self, recording: LocataRecording) -> None:
        vad = _read_binary_vad(recording.vad_path)
        available_frames = min(recording.num_frames, int(vad.size))
        window_frames = int(round(self.segment_seconds * recording.native_sample_rate))
        hop_frames = int(round(self.hop_seconds * recording.native_sample_rate))
        if available_frames < window_frames:
            return

        for start_frame in range(0, available_frames - window_frames + 1, hop_frames):
            end_frame = start_frame + window_frames
            vad_ratio = float(vad[start_frame:end_frame].mean())
            if vad_ratio + 1e-12 < self.min_vad_ratio:
                continue
            self.segments.append(
                LocataSegment(
                    recording_id=recording.recording_id,
                    audio_path=recording.audio_path,
                    native_sample_rate=recording.native_sample_rate,
                    start_frame=start_frame,
                    num_frames=window_frames,
                    start_sec=start_frame / recording.native_sample_rate,
                    duration_sec=self.segment_seconds,
                    vad_ratio=vad_ratio,
                    azimuth_deg=recording.azimuth_deg,
                    elevation_deg=recording.elevation_deg,
                )
            )

    def _cache_path(self, segment: LocataSegment) -> Optional[Path]:
        if self.feature_cache_dir is None:
            return None
        cache_key = "|".join(
            [
                segment.audio_path,
                str(segment.start_frame),
                str(segment.num_frames),
                str(self.channels),
                str(self.target_sample_rate),
                str(self.peak_normalize),
                str(self.peak_value),
                str(self.feature_extractor.n_fft),
                str(self.feature_extractor.hop_length),
                str(self.feature_extractor.win_length),
            ]
        )
        digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:20]
        return self.feature_cache_dir / f"{segment.recording_id}_{digest}.pt"

    def _read_audio(self, segment: LocataSegment) -> np.ndarray:
        audio, native_sr = sf.read(
            segment.audio_path,
            start=segment.start_frame,
            frames=segment.num_frames,
            dtype="float32",
            always_2d=True,
        )
        if int(native_sr) != segment.native_sample_rate:
            raise ValueError(f"Unexpected sample rate change in {segment.audio_path}")
        audio = audio[:, self.channels].T
        if native_sr != self.target_sample_rate:
            audio = resample_poly(audio, self.target_sample_rate, native_sr, axis=1)

        target_samples = int(round(self.segment_seconds * self.target_sample_rate))
        if audio.shape[1] < target_samples:
            audio = np.pad(audio, ((0, 0), (0, target_samples - audio.shape[1])))
        elif audio.shape[1] > target_samples:
            audio = audio[:, :target_samples]

        audio = audio.astype(np.float32, copy=False)
        if self.peak_normalize:
            peak = float(np.max(np.abs(audio)))
            if peak > 1e-8:
                audio = audio * (self.peak_value / peak)
        return audio

    def _extract_features(self, segment: LocataSegment) -> Dict[str, torch.Tensor]:
        cache_path = self._cache_path(segment)
        if cache_path is not None and cache_path.is_file():
            return torch.load(cache_path, map_location="cpu", weights_only=False)

        audio = torch.from_numpy(self._read_audio(segment))
        features = {
            name: tensor.contiguous().cpu()
            for name, tensor in self.feature_extractor.extract(audio).items()
        }
        if cache_path is not None:
            temporary_path = cache_path.with_name(
                f"{cache_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
            )
            torch.save(features, temporary_path)
            os.replace(temporary_path, cache_path)
        return features

    def _nearest_label(self, azimuth_deg: float) -> int:
        lo, hi = self.azimuth_range
        centers = lo + np.arange(self.num_classes, dtype=np.float64) * (
            (hi - lo) / self.num_classes
        )
        errors = _circular_error_deg(centers, azimuth_deg)
        return int(np.argmin(errors))

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        segment = self.segments[index]
        sample: Dict[str, Any] = self._extract_features(segment)
        azimuth = float(segment.azimuth_deg)
        sample.update(
            {
                "file_id": f"{segment.recording_id}_{segment.start_sec:.3f}",
                "recording_id": segment.recording_id,
                "start_sec": torch.tensor(segment.start_sec, dtype=torch.float32),
                "vad_ratio": torch.tensor(segment.vad_ratio, dtype=torch.float32),
                "azimuth_deg": torch.tensor(azimuth, dtype=torch.float32),
                "elevation_deg": torch.tensor(segment.elevation_deg, dtype=torch.float32),
                "azimuth_label": torch.tensor(self._nearest_label(azimuth), dtype=torch.long),
                "front_back_label": torch.tensor(0 if abs(azimuth) <= 90.0 else 1),
            }
        )
        return sample

    def protocol_summary(self) -> Dict[str, Any]:
        return {
            "root_dir": self.root_dir,
            "split": self.split,
            "task": 1,
            "array": "dummy",
            "recording_count": len(self.recordings),
            "segment_count": len(self.segments),
            "target_sample_rate": self.target_sample_rate,
            "segment_seconds": self.segment_seconds,
            "hop_seconds": self.hop_seconds,
            "min_vad_ratio": self.min_vad_ratio,
            "channels_zero_based": list(self.channels),
            "channels_locata_one_based": [channel + 1 for channel in self.channels],
            "peak_normalize": self.peak_normalize,
            "peak_value": self.peak_value if self.peak_normalize else None,
            "azimuth_convention": "0 deg front, +90 deg right, atan2(local_x, local_y)",
            "channel_geometry": self.channel_geometry,
            "recordings": [asdict(recording) for recording in self.recordings],
        }
