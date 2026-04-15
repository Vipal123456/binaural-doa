"""静态双耳DOA数据集加载器。

用于加载格式为:
- binaural_dev/binaural{XXXX}.wav: 双耳音频文件 (24kHz, 2通道, 10秒)
- metadata_dev/metadata{XXXX}.csv: 元数据文件 (帧号, x, y, z, 0, 0, 0, 0)

方位角从笛卡尔坐标 (x, y) 计算: azimuth = atan2(x, y) * 180 / pi
"""

import os
import glob
import warnings
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .feature_extractor import FeatureExtractor

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import librosa
except ImportError:
    librosa = None


@dataclass
class StaticRecording:
    """单条静态录音的元数据。"""
    audio_path: str
    metadata_path: str
    file_id: str  # e.g., "0001"


class StaticDOADataset(Dataset):
    """静态双耳DOA数据集。

    参数
    ----------
    root_dir : str
        数据集根目录，包含 binaural_dev/ 和 metadata_dev/ 子目录
    sample_rate : int
        目标采样率 (Hz)，默认 16000
    segment_seconds : float
        每个片段的时长 (秒)，默认 2.0
    num_classes : int
        方位角分类数量，默认 72 (每5度一类)
    azimuth_range : tuple
        方位角范围，默认 (-180, 180)
    split : str
        数据集划分 ("train", "val", "test")
    train_ratio : float
        训练集比例
    val_ratio : float
        验证集比例
    split_seed : int
        划分随机种子
    """

    def __init__(
        self,
        root_dir: str,
        sample_rate: int = 16000,
        segment_seconds: float = 2.0,
        num_classes: int = 72,
        azimuth_range: Tuple[int, int] = (-180, 180),
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        split_seed: int = 42,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        window: str = "hann",
        add_white_noise: bool = False,
        white_noise_snr_db: float = 10.0,
        white_noise_prob: float = 1.0,
        white_noise_splits: Optional[List[str]] = None,
    ):
        self.root_dir = root_dir
        self.target_sr = sample_rate
        self.segment_seconds = segment_seconds
        self.num_classes = num_classes
        self.azimuth_range = azimuth_range
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.split_seed = split_seed
        self.add_white_noise = add_white_noise
        self.white_noise_snr_db = float(white_noise_snr_db)
        self.white_noise_prob = float(white_noise_prob)
        self.white_noise_splits = set(white_noise_splits or ["train"])

        # 计算方位角分辨率
        self.az_min, self.az_max = azimuth_range
        self.az_resolution = (self.az_max - self.az_min) / num_classes

        # 特征提取器
        self.feature_extractor = FeatureExtractor(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
        )

        # 扫描数据集
        self.recordings = self._scan_recordings()

        # 划分数据集
        self.recordings = self._split_recordings()

        # 构建片段索引
        self.segments: List[Dict[str, Any]] = []
        self._build_segments()

    def _scan_recordings(self) -> List[StaticRecording]:
        """扫描数据集目录，找到所有音频-元数据对。"""
        recordings = []

        audio_dir = os.path.join(self.root_dir, "binaural_dev")
        metadata_dir = os.path.join(self.root_dir, "metadata_dev")

        if not os.path.isdir(audio_dir):
            raise ValueError(f"Audio directory not found: {audio_dir}")
        if not os.path.isdir(metadata_dir):
            raise ValueError(f"Metadata directory not found: {metadata_dir}")

        # 查找所有音频文件
        audio_pattern = os.path.join(audio_dir, "binaural*.wav")
        audio_files = sorted(glob.glob(audio_pattern))

        for audio_path in audio_files:
            # 提取文件ID (e.g., "0001" from "binaural0001.wav")
            basename = os.path.basename(audio_path)
            file_id = basename.replace("binaural", "").replace(".wav", "")

            # 检查对应的元数据文件是否存在
            metadata_path = os.path.join(metadata_dir, f"metadata{file_id}.csv")
            if os.path.isfile(metadata_path):
                recordings.append(StaticRecording(
                    audio_path=audio_path,
                    metadata_path=metadata_path,
                    file_id=file_id,
                ))
            else:
                warnings.warn(f"Metadata not found for {audio_path}")

        return recordings

    def _split_recordings(self) -> List[StaticRecording]:
        """按比例划分数据集。"""
        n = len(self.recordings)
        if n == 0:
            return []

        # 固定随机种子进行划分
        rng = np.random.default_rng(self.split_seed)
        indices = rng.permutation(n)

        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        if self.split == "train":
            selected = indices[:n_train]
        elif self.split == "val":
            selected = indices[n_train:n_train + n_val]
        else:  # test
            selected = indices[n_train + n_val:]

        return [self.recordings[i] for i in selected]

    def _build_segments(self) -> None:
        """将录音切分为固定长度的片段。"""
        segment_samples = int(self.segment_seconds * self.target_sr)

        for rec in self.recordings:
            # 获取音频时长
            duration_sec = self._get_audio_duration_sec(rec.audio_path)
            if duration_sec is None:
                warnings.warn(f"Cannot read duration of {rec.audio_path}; skipping.")
                continue

            # 读取元数据获取方位角（静态场景，取单一值）
            azimuth_deg = self._read_azimuth(rec.metadata_path)
            if azimuth_deg is None:
                warnings.warn(f"Cannot read azimuth from {rec.metadata_path}; skipping.")
                continue

            # 计算方位角标签
            azimuth_label = self._azimuth_to_label(azimuth_deg)

            # 计算片段数量
            num_segments = int(duration_sec // self.segment_seconds)
            if num_segments == 0:
                continue

            for seg_idx in range(num_segments):
                start_sec = seg_idx * self.segment_seconds
                self.segments.append({
                    "audio_path": rec.audio_path,
                    "start_sec": start_sec,
                    "duration_sec": self.segment_seconds,
                    "azimuth_deg": azimuth_deg,
                    "azimuth_label": azimuth_label,
                })

    @staticmethod
    def _get_audio_duration_sec(path: str) -> Optional[float]:
        """获取音频文件时长 (秒)。"""
        if sf is not None:
            try:
                info = sf.info(path)
                return info.frames / info.samplerate
            except Exception:
                pass
        if librosa is not None:
            try:
                return librosa.get_duration(path=path)
            except Exception:
                pass
        return None

    def _read_azimuth(self, metadata_path: str) -> Optional[float]:
        """从元数据文件读取方位角。

        CSV格式: 帧号, x, y, z, 0, 0, 0, 0
        方位角 = atan2(x, y) * 180 / pi
        """
        try:
            data = np.loadtxt(metadata_path, delimiter=",")
            if data.ndim == 1:
                data = data.reshape(1, -1)

            # 取中间帧的坐标
            mid_idx = len(data) // 2
            x = data[mid_idx, 1]
            y = data[mid_idx, 2]

            # 计算方位角
            azimuth_rad = np.arctan2(x, y)
            azimuth_deg = np.degrees(azimuth_rad)

            return float(azimuth_deg)
        except Exception as e:
            warnings.warn(f"Error reading {metadata_path}: {e}")
            return None

    def _azimuth_to_label(self, azimuth_deg: float) -> int:
        """将方位角 (度) 转换为离散类别标签。"""
        # 归一化到 [az_min, az_max) 范围
        az = azimuth_deg
        while az < self.az_min:
            az += 360
        while az >= self.az_max:
            az -= 360

        # 转换为类别索引
        label = int((az - self.az_min) / self.az_resolution)
        label = max(0, min(self.num_classes - 1, label))
        return label

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seg = self.segments[idx]

        # 加载音频片段
        audio = self._load_audio_segment(
            seg["audio_path"],
            seg["start_sec"],
            seg["duration_sec"],
        )

        # 确保是双通道
        if audio.shape[0] != 2:
            if audio.shape[0] > 2:
                audio = audio[:2, :]
            else:
                audio = np.vstack([audio, audio])

        # 可选：按目标SNR添加白噪声
        if self.add_white_noise and self.split in self.white_noise_splits:
            if np.random.random() < self.white_noise_prob:
                audio = self._add_white_noise_with_snr(audio, self.white_noise_snr_db)

        # 转换为 tensor 并提取特征
        audio_tensor = torch.from_numpy(audio).float()
        feats = self.feature_extractor.extract(audio_tensor)

        return {
            "log_mag_L": feats["log_mag_L"],   # [T, F]
            "log_mag_R": feats["log_mag_R"],   # [T, F]
            "ipd": feats["ipd"],               # [T, F]
            "ild": feats["ild"],               # [T, F]
            "ipd_sin": feats["ipd_sin"],       # [T, F]
            "ipd_cos": feats["ipd_cos"],       # [T, F]
            "coherence": feats["coherence"],   # [T, F]
            "azimuth_label": seg["azimuth_label"],
            "azimuth_deg": seg["azimuth_deg"],
        }

    @staticmethod
    def _add_white_noise_with_snr(audio: np.ndarray, snr_db: float) -> np.ndarray:
        """向双通道音频添加白噪声，使其达到目标SNR(dB)。"""
        signal_power = float(np.mean(audio.astype(np.float64) ** 2))
        if signal_power < 1e-12:
            return audio

        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = signal_power / max(snr_linear, 1e-12)
        noise_std = math.sqrt(noise_power)

        noise = np.random.normal(0.0, noise_std, size=audio.shape).astype(np.float32)
        noisy = (audio + noise).astype(np.float32, copy=False)
        noisy = np.clip(noisy, -1.0, 1.0)
        return noisy

    def _load_audio_segment(
        self,
        audio_path: str,
        start_sec: float,
        duration_sec: float,
    ) -> np.ndarray:
        """加载音频片段并重采样。

        返回形状: (channels, samples)
        """
        segment_samples = int(duration_sec * self.target_sr)

        if sf is not None:
            try:
                info = sf.info(audio_path)
                native_sr = info.samplerate

                start_frame = int(start_sec * native_sr)
                frames_to_read = int(duration_sec * native_sr)

                audio, sr = sf.read(
                    audio_path,
                    start=start_frame,
                    frames=frames_to_read,
                    dtype="float32",
                    always_2d=True,
                )
                # (frames, channels) -> (channels, frames)
                audio = audio.T

                # 重采样
                if sr != self.target_sr and librosa is not None:
                    audio = librosa.resample(
                        audio,
                        orig_sr=sr,
                        target_sr=self.target_sr,
                    )

                # 确保正确长度
                if audio.shape[1] < segment_samples:
                    pad = segment_samples - audio.shape[1]
                    audio = np.pad(audio, ((0, 0), (0, pad)), mode="constant")
                elif audio.shape[1] > segment_samples:
                    audio = audio[:, :segment_samples]

                return audio
            except Exception as e:
                warnings.warn(f"Error loading {audio_path}: {e}")

        # Fallback: 返回零数组
        return np.zeros((2, segment_samples), dtype=np.float32)


def build_static_datasets(cfg) -> Tuple[Dataset, Dataset, Dataset]:
    """根据配置构建训练、验证、测试数据集。

    参数
    ----------
    cfg : Config
        配置对象，需包含 cfg.dataset 配置

    返回
    -------
    train_dataset, val_dataset, test_dataset
    """
    ds_cfg = cfg.dataset
    model_cfg = cfg.model
    feat_cfg = cfg.feature

    common_kwargs = dict(
        root_dir=ds_cfg.root_dir,
        sample_rate=ds_cfg.sample_rate,
        segment_seconds=ds_cfg.segment_seconds,
        num_classes=model_cfg.num_classes,
        azimuth_range=tuple(model_cfg.azimuth_range),
        train_ratio=ds_cfg.train_ratio,
        val_ratio=ds_cfg.val_ratio,
        split_seed=ds_cfg.split_seed,
        n_fft=feat_cfg.n_fft,
        hop_length=feat_cfg.hop_length,
        win_length=feat_cfg.win_length,
        window=feat_cfg.window,
        add_white_noise=ds_cfg.get("add_white_noise", False),
        white_noise_snr_db=ds_cfg.get("white_noise_snr_db", 10.0),
        white_noise_prob=ds_cfg.get("white_noise_prob", 1.0),
        white_noise_splits=ds_cfg.get("white_noise_splits", ["train"]),
    )

    train_ds = StaticDOADataset(split="train", **common_kwargs)
    val_ds = StaticDOADataset(split="val", **common_kwargs)
    test_ds = StaticDOADataset(split="test", **common_kwargs)

    return train_ds, val_ds, test_ds
