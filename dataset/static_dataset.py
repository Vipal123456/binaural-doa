"""静态双耳 DOA 数据集加载器。

支持两类协议：
1. 目录内直接做随机 train/val/test 比例划分
2. BinMov2023 / SDEL 风格的固定 5-fold 文件划分（由 ``rnd_files.npy`` 指定）
"""

import os
import glob
import warnings
import math
import pickle
import csv
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict, Any

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
    azimuth_deg: Optional[float] = None


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
        class_angles_deg: Optional[Sequence[float]] = None,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        split_seed: int = 42,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        window: str = "hann",
        spatial_statistics_mode: str = "legacy",
        spatial_statistics_time_frames: int = 1,
        add_white_noise: bool = False,
        white_noise_snr_db: float = 10.0,
        white_noise_prob: float = 1.0,
        white_noise_splits: Optional[List[str]] = None,
        audio_subdir: str = "binaural_dev",
        metadata_subdir: str = "metadata_dev",
        split_strategy: str = "ratio",
        split_folds: Optional[List[int]] = None,
        rnd_files_path: Optional[str] = None,
        azimuth_coordinate_order: str = "xy",
        segment_hop_seconds: Optional[float] = None,
        max_segments_per_recording: Optional[int] = None,
        include_waveform: bool = False,
        component_supervision_enabled: bool = False,
        component_target_subdir: str = "components/target",
        component_interferer_subdir: str = "components/interferer",
        feature_cache_enabled: bool = False,
        feature_cache_dir: Optional[str] = None,
        logger: Optional[Any] = None,
    ):
        self.root_dir = root_dir
        self.target_sr = sample_rate
        self.segment_seconds = segment_seconds
        self.num_classes = num_classes
        self.azimuth_range = azimuth_range
        self.class_angles_deg = (
            None
            if class_angles_deg is None
            else np.asarray(class_angles_deg, dtype=np.float64)
        )
        if self.class_angles_deg is not None:
            if self.class_angles_deg.ndim != 1 or len(self.class_angles_deg) != num_classes:
                raise ValueError(
                    "class_angles_deg must contain exactly num_classes values"
                )
            if len(np.unique(self.class_angles_deg)) != num_classes:
                raise ValueError("class_angles_deg values must be unique")
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.split_seed = split_seed
        self.add_white_noise = add_white_noise
        self.white_noise_snr_db = float(white_noise_snr_db)
        self.white_noise_prob = float(white_noise_prob)
        self.white_noise_splits = set(white_noise_splits or ["train"])
        self.audio_subdir = audio_subdir
        self.metadata_subdir = metadata_subdir
        self.split_strategy = split_strategy
        self.split_folds = sorted(int(f) for f in split_folds) if split_folds else None
        self.rnd_files_path = rnd_files_path
        self.azimuth_coordinate_order = azimuth_coordinate_order
        self.segment_hop_seconds = float(segment_hop_seconds) if segment_hop_seconds is not None else self.segment_seconds
        self.max_segments_per_recording = (
            int(max_segments_per_recording) if max_segments_per_recording is not None else None
        )
        self.include_waveform = bool(include_waveform)
        self.component_supervision_enabled = bool(component_supervision_enabled)
        self.component_target_root = os.path.join(root_dir, component_target_subdir)
        self.component_interferer_root = os.path.join(
            root_dir, component_interferer_subdir
        )
        self.component_supervision_available = (
            self.component_supervision_enabled
            and os.path.isdir(self.component_target_root)
            and os.path.isdir(self.component_interferer_root)
        )
        self.feature_cache_enabled = bool(feature_cache_enabled)
        self.feature_cache_dir = feature_cache_dir
        self.logger = logger

        # 计算方位角分辨率
        self.az_min, self.az_max = azimuth_range
        self.az_resolution = (self.az_max - self.az_min) / num_classes

        # 特征提取器
        self.feature_extractor = FeatureExtractor(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            spatial_statistics_mode=spatial_statistics_mode,
            spatial_statistics_time_frames=spatial_statistics_time_frames,
        )
        self._cache_warning_emitted = False
        self._feature_cache_root = self._init_feature_cache_root()

        self.recordings: List[StaticRecording] = []
        self.segments: List[Dict[str, Any]] = []
        if not self._try_load_cache():
            self.recordings = self._scan_recordings()
            self.recordings = self._split_recordings()
            self._build_segments()
            self._save_cache()

    def _cache_path(self) -> str:
        try:
            cache_dir = os.path.join(self.root_dir, ".doa_cache")
            os.makedirs(cache_dir, exist_ok=True)
        except OSError:
            safe_root = os.path.join(
                os.getcwd(),
                "outputs",
                "dataset_cache",
                Path(self.root_dir).name,
            )
            os.makedirs(safe_root, exist_ok=True)
            cache_dir = safe_root
        seg_ms = int(round(self.segment_seconds * 1000.0))
        az_min, az_max = self.azimuth_range
        filename = (
            f"{self.split}_sr{self.target_sr}_seg{seg_ms}ms_"
            f"cls{self.num_classes}_az{az_min}_{az_max}_seed{self.split_seed}.pkl"
        )
        if self.class_angles_deg is not None:
            angle_digest = hashlib.md5(self.class_angles_deg.tobytes()).hexdigest()[:10]
            filename = filename.replace(".pkl", f"_angles{angle_digest}.pkl")
        hop_ms = int(round(self.segment_hop_seconds * 1000.0))
        maxseg_tag = (
            f"maxseg{self.max_segments_per_recording}"
            if self.max_segments_per_recording is not None
            else "maxsegall"
        )
        filename = filename.replace(".pkl", f"_hop{hop_ms}ms_{maxseg_tag}.pkl")
        if self.split_strategy != "ratio":
            strategy_tag = self.split_strategy
            if self.split_folds:
                strategy_tag += "_f" + "-".join(str(f) for f in self.split_folds)
            filename = filename.replace(".pkl", f"_{strategy_tag}.pkl")
        return os.path.join(cache_dir, filename)

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)

    def _init_feature_cache_root(self) -> Optional[str]:
        if not self.feature_cache_enabled:
            return None
        if self.add_white_noise and self.split in self.white_noise_splits:
            self.feature_cache_enabled = False
            if not self._cache_warning_emitted:
                warnings.warn(
                    "Feature cache is disabled because white-noise augmentation is active "
                    f"for split '{self.split}', which would make cached features stochastic."
                )
                self._cache_warning_emitted = True
            return None

        cache_root = self.feature_cache_dir
        if not cache_root:
            cache_root = os.path.join(self.root_dir, ".feature_cache")
        feature_tag = (
            f"sr{self.target_sr}_seg{int(round(self.segment_seconds * 1000.0))}ms_"
            f"fft{self.feature_extractor.n_fft}_hop{self.feature_extractor.hop_length}_"
            f"win{self.feature_extractor.win_length}_"
            f"stats{self.feature_extractor.spatial_statistics_mode}_"
            f"st{self.feature_extractor.spatial_statistics_time_frames}"
        )
        cache_root = os.path.join(cache_root, feature_tag)
        os.makedirs(cache_root, exist_ok=True)
        self._log(f"[{self.split}] feature cache enabled: {cache_root}")
        return cache_root

    def _feature_cache_path(self, seg: Dict[str, Any]) -> Optional[str]:
        if not self.feature_cache_enabled or self._feature_cache_root is None:
            return None
        cache_key = (
            f"{seg['audio_path']}|{seg['start_sec']:.6f}|{seg['duration_sec']:.6f}|"
            f"{self.target_sr}|{self.feature_extractor.n_fft}|{self.feature_extractor.hop_length}|"
            f"{self.feature_extractor.win_length}|"
            f"{self.feature_extractor.spatial_statistics_mode}|"
            f"{self.feature_extractor.spatial_statistics_time_frames}|"
            f"{self.audio_subdir}|{self.metadata_subdir}"
        )
        digest = hashlib.md5(cache_key.encode("utf-8")).hexdigest()
        prefix = str(seg.get("file_id", "unknown"))
        return os.path.join(self._feature_cache_root, f"{prefix}_{digest}.pt")

    def _load_or_extract_features(self, seg: Dict[str, Any], audio: np.ndarray) -> Dict[str, torch.Tensor]:
        cache_path = self._feature_cache_path(seg)
        if cache_path and os.path.isfile(cache_path):
            return torch.load(cache_path, map_location="cpu")

        audio_tensor = torch.from_numpy(audio).float()
        feats = self.feature_extractor.extract(audio_tensor)
        feats = {k: v.contiguous().cpu() for k, v in feats.items()}

        if cache_path:
            tmp_path = cache_path + f".tmp.{os.getpid()}"
            torch.save(feats, tmp_path)
            os.replace(tmp_path, cache_path)
        return feats

    def _try_load_cache(self) -> bool:
        cache_path = self._cache_path()
        if not os.path.isfile(cache_path):
            return False
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            self.recordings = [
                StaticRecording(**rec_dict) for rec_dict in payload["recordings"]
            ]
            self.segments = payload["segments"]
            self._log(
                f"[{self.split}] 从缓存载入数据索引: "
                f"{len(self.recordings)} recordings / {len(self.segments)} segments"
            )
            return True
        except Exception as exc:
            warnings.warn(f"Failed to load cache {cache_path}: {exc}")
            return False

    def _save_cache(self) -> None:
        cache_path = self._cache_path()
        payload = {
            "recordings": [rec.__dict__ for rec in self.recordings],
            "segments": self.segments,
        }
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._log(
                f"[{self.split}] 数据索引缓存已保存: "
                f"{len(self.recordings)} recordings / {len(self.segments)} segments"
            )
        except Exception as exc:
            warnings.warn(f"Failed to save cache {cache_path}: {exc}")

    def _scan_recordings(self) -> List[StaticRecording]:
        """扫描数据集目录，找到所有音频-元数据对。"""
        recordings = []

        audio_dir = os.path.join(self.root_dir, self.audio_subdir)
        metadata_dir = os.path.join(self.root_dir, self.metadata_subdir)
        flat_metadata_csv = os.path.join(self.root_dir, "metadata.csv")

        # New flat-layout protocol used by the KEMAR + SofaMyRoom dataset:
        # root/
        #   metadata.csv
        #   binaural/*.wav
        if os.path.isfile(flat_metadata_csv):
            with open(flat_metadata_csv, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            self._log(f"[{self.split}] 扫描扁平元数据: {flat_metadata_csv} ({len(rows)} rows)")
            for idx, row in enumerate(rows, start=1):
                wav_path_raw = str(row.get("wav_path", "")).strip()
                if not wav_path_raw:
                    warnings.warn(f"Missing wav_path in {flat_metadata_csv} row {idx}; skipping.")
                    continue
                candidate_paths = []
                if os.path.isabs(wav_path_raw):
                    candidate_paths.append(wav_path_raw)
                else:
                    candidate_paths.append(wav_path_raw)
                    candidate_paths.append(os.path.join(self.root_dir, wav_path_raw))
                wav_path = next((p for p in candidate_paths if os.path.isfile(p)), candidate_paths[-1])
                if not os.path.isfile(wav_path):
                    warnings.warn(f"Audio not found for metadata row: {wav_path}")
                    continue
                try:
                    azimuth_deg = float(row["azimuth_deg"])
                except Exception:
                    warnings.warn(f"Invalid azimuth_deg in {flat_metadata_csv} row {idx}; skipping.")
                    continue
                file_id = str(row.get("file_id", Path(wav_path).stem)).strip() or Path(wav_path).stem
                recordings.append(StaticRecording(
                    audio_path=wav_path,
                    metadata_path=flat_metadata_csv,
                    file_id=file_id,
                    azimuth_deg=azimuth_deg,
                ))
                if idx % 1000 == 0 or idx == len(rows):
                    self._log(f"[{self.split}] 已扫描 {idx}/{len(rows)} 个扁平样本")
            return recordings

        if not os.path.isdir(audio_dir):
            raise ValueError(f"Audio directory not found: {audio_dir}")
        if not os.path.isdir(metadata_dir):
            raise ValueError(f"Metadata directory not found: {metadata_dir}")

        # 查找所有音频文件
        audio_pattern = os.path.join(audio_dir, "binaural*.wav")
        audio_files = sorted(glob.glob(audio_pattern))
        self._log(f"[{self.split}] 扫描音频目录: {audio_dir} ({len(audio_files)} files)")

        for idx, audio_path in enumerate(audio_files, start=1):
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

            if idx % 500 == 0 or idx == len(audio_files):
                self._log(f"[{self.split}] 已扫描 {idx}/{len(audio_files)} 个文件")

        return recordings

    def _recording_fold(self, file_id: str) -> Optional[int]:
        if not self.rnd_files_path:
            return None
        rnd = np.load(self.rnd_files_path)
        fil_nb = int(file_id)
        for fold_idx in range(5):
            start = fold_idx * 500
            end = start + 500
            if fil_nb in rnd[start:end]:
                return fold_idx + 1
        return None

    def _split_recordings(self) -> List[StaticRecording]:
        """按比例划分数据集。"""
        n = len(self.recordings)
        if n == 0:
            return []

        if self.split_strategy == "binmov_fold":
            if not self.split_folds:
                raise ValueError("split_strategy='binmov_fold' requires split_folds")
            if not self.rnd_files_path or not os.path.isfile(self.rnd_files_path):
                raise ValueError("Valid rnd_files_path is required for binmov_fold split")
            selected = []
            for rec in self.recordings:
                fold = self._recording_fold(rec.file_id)
                if fold in self.split_folds:
                    selected.append(rec)
            self._log(
                f"[{self.split}] fold 过滤后保留 {len(selected)}/{n} recordings "
                f"(folds={self.split_folds})"
            )
            return selected

        # 固定随机种子进行划分
        rng = np.random.default_rng(self.split_seed)
        indices = rng.permutation(n)

        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        if self.split == "all":
            selected = indices
        elif self.split == "train":
            selected = indices[:n_train]
        elif self.split == "val":
            selected = indices[n_train:n_train + n_val]
        else:  # test
            selected = indices[n_train + n_val:]

        return [self.recordings[i] for i in selected]

    def _build_segments(self) -> None:
        """将录音切分为固定长度的片段。"""
        total = len(self.recordings)
        self._log(f"[{self.split}] 开始构建片段索引: {total} recordings")

        for idx, rec in enumerate(self.recordings, start=1):
            # 获取音频时长
            duration_sec = self._get_audio_duration_sec(rec.audio_path)
            if duration_sec is None:
                warnings.warn(f"Cannot read duration of {rec.audio_path}; skipping.")
                continue

            # 读取元数据获取方位角（静态场景，取单一值）
            azimuth_deg = rec.azimuth_deg if rec.azimuth_deg is not None else self._read_azimuth(rec.metadata_path)
            if azimuth_deg is None:
                warnings.warn(f"Cannot read azimuth from {rec.metadata_path}; skipping.")
                continue

            # 计算方位角标签
            azimuth_label = self._azimuth_to_label(azimuth_deg)

            # 计算片段数量
            num_segments = int((duration_sec - self.segment_seconds) // self.segment_hop_seconds) + 1
            if self.max_segments_per_recording is not None:
                num_segments = min(num_segments, self.max_segments_per_recording)
            if num_segments == 0:
                continue

            for seg_idx in range(num_segments):
                start_sec = seg_idx * self.segment_hop_seconds
                self.segments.append({
                    "file_id": rec.file_id,
                    "audio_path": rec.audio_path,
                    "start_sec": start_sec,
                    "duration_sec": self.segment_seconds,
                    "azimuth_deg": azimuth_deg,
                    "azimuth_label": azimuth_label,
                })

            if idx % 250 == 0 or idx == total:
                self._log(
                    f"[{self.split}] 已完成 {idx}/{total} recordings, "
                    f"累计 {len(self.segments)} segments"
                )

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

        CSV格式: 帧号, x, y, z, ...
        方位角支持两种约定：
        - ``xy``: atan2(x, y) * 180 / pi  （项目原始约定）
        - ``yx``: atan2(y, x) * 180 / pi  （BinMov2023 / SDEL 约定）
        """
        try:
            data = np.loadtxt(metadata_path, delimiter=",")
            if data.ndim == 1:
                data = data.reshape(1, -1)

            # 取中间帧的坐标
            mid_idx = len(data) // 2
            x = data[mid_idx, 1]
            y = data[mid_idx, 2]

            if self.azimuth_coordinate_order == "yx":
                azimuth_rad = np.arctan2(y, x)
            else:
                azimuth_rad = np.arctan2(x, y)
            azimuth_deg = np.degrees(azimuth_rad)

            return float(azimuth_deg)
        except Exception as e:
            warnings.warn(f"Error reading {metadata_path}: {e}")
            return None

    def _azimuth_to_label(self, azimuth_deg: float) -> int:
        """将方位角 (度) 转换为离散类别标签。"""
        if self.class_angles_deg is not None:
            diffs = np.abs(
                (self.class_angles_deg - float(azimuth_deg) + 180.0) % 360.0 - 180.0
            )
            return int(np.argmin(diffs))

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

    @staticmethod
    def _wrap_azimuth_deg(azimuth_deg: float) -> float:
        """将角度规范化到 [-180, 180) 区间。"""
        wrapped = ((float(azimuth_deg) + 180.0) % 360.0) - 180.0
        return wrapped

    def _azimuth_to_front_back_label(self, azimuth_deg: float) -> int:
        """将方位角映射为前/后标签。

        定义：
        - ``front = 0``: [-90°, 90°]
        - ``back = 1``: 其余区间
        """
        wrapped = self._wrap_azimuth_deg(azimuth_deg)
        return 0 if abs(wrapped) <= 90.0 else 1

    def _compute_front_back_focus_distance_deg(self, azimuth_deg: float) -> float:
        """计算样本到前后轴（0° / 180°）的最小圆周距离。"""
        wrapped = self._wrap_azimuth_deg(azimuth_deg)
        dist_to_front = abs(wrapped)
        dist_to_back = abs(abs(wrapped) - 180.0)
        return min(dist_to_front, dist_to_back)

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

        # 提取特征（可选磁盘缓存）
        feats = self._load_or_extract_features(seg, audio)

        file_id = seg.get("file_id")
        if file_id is None:
            basename = os.path.basename(seg["audio_path"])
            file_id = basename.replace("binaural", "").replace(".wav", "")

        sample = {
            "file_id": file_id,
            "log_mag_L": feats["log_mag_L"],   # [T, F]
            "log_mag_R": feats["log_mag_R"],   # [T, F]
            "spec_real_L": feats["spec_real_L"],  # [T, F]
            "spec_imag_L": feats["spec_imag_L"],  # [T, F]
            "spec_real_R": feats["spec_real_R"],  # [T, F]
            "spec_imag_R": feats["spec_imag_R"],  # [T, F]
            "ipd": feats["ipd"],               # [T, F]
            "ild": feats["ild"],               # [T, F]
            "ipd_sin": feats["ipd_sin"],       # [T, F]
            "ipd_cos": feats["ipd_cos"],       # [T, F]
            "coherence": feats["coherence"],   # [T, F]
            "azimuth_label": seg["azimuth_label"],
            "azimuth_deg": seg["azimuth_deg"],
            "front_back_label": self._azimuth_to_front_back_label(seg["azimuth_deg"]),
            "front_back_focus_distance_deg": self._compute_front_back_focus_distance_deg(seg["azimuth_deg"]),
        }
        if self.include_waveform:
            audio_tensor = torch.from_numpy(audio).float()
            sample["waveform"] = audio_tensor  # [2, samples]
        if self.component_supervision_available:
            target_path = os.path.join(self.component_target_root, f"{file_id}.wav")
            interferer_path = os.path.join(
                self.component_interferer_root, f"{file_id}.wav"
            )
            if not os.path.isfile(target_path) or not os.path.isfile(interferer_path):
                raise FileNotFoundError(
                    f"Missing component supervision for {file_id}: "
                    f"{target_path}, {interferer_path}"
                )
            target_audio = self._load_audio_segment(
                target_path, seg["start_sec"], seg["duration_sec"]
            )
            interferer_audio = self._load_audio_segment(
                interferer_path, seg["start_sec"], seg["duration_sec"]
            )
            target_feats = self.feature_extractor.extract(
                torch.from_numpy(target_audio).float()
            )
            interferer_feats = self.feature_extractor.extract(
                torch.from_numpy(interferer_audio).float()
            )
            for side in ("L", "R"):
                for part in ("real", "imag"):
                    key = f"spec_{part}_{side}"
                    sample[f"target_{key}"] = target_feats[key].contiguous()
                    sample[f"interferer_{key}"] = interferer_feats[key].contiguous()
        return sample

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


def build_static_datasets(cfg, logger=None) -> Tuple[Dataset, Dataset, Dataset]:
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
        class_angles_deg=model_cfg.get("class_angles_deg", None),
        train_ratio=ds_cfg.get("train_ratio", 0.7),
        val_ratio=ds_cfg.get("val_ratio", 0.15),
        split_seed=ds_cfg.get("split_seed", 42),
        n_fft=feat_cfg.n_fft,
        hop_length=feat_cfg.hop_length,
        win_length=feat_cfg.win_length,
        window=feat_cfg.window,
        spatial_statistics_mode=feat_cfg.get("spatial_statistics_mode", "legacy"),
        spatial_statistics_time_frames=feat_cfg.get("spatial_statistics_time_frames", 1),
        add_white_noise=ds_cfg.get("add_white_noise", False),
        white_noise_snr_db=ds_cfg.get("white_noise_snr_db", 10.0),
        white_noise_prob=ds_cfg.get("white_noise_prob", 1.0),
        white_noise_splits=ds_cfg.get("white_noise_splits", ["train"]),
        audio_subdir=ds_cfg.get("audio_subdir", "binaural_dev"),
        metadata_subdir=ds_cfg.get("metadata_subdir", "metadata_dev"),
        split_strategy=ds_cfg.get("split_strategy", "ratio"),
        rnd_files_path=ds_cfg.get("rnd_files_path", None),
        azimuth_coordinate_order=ds_cfg.get("azimuth_coordinate_order", "xy"),
        segment_hop_seconds=ds_cfg.get("segment_hop_seconds", None),
        max_segments_per_recording=ds_cfg.get("max_segments_per_recording", None),
        include_waveform=ds_cfg.get("include_waveform", False),
        component_supervision_enabled=ds_cfg.get(
            "component_supervision_enabled", False
        ),
        component_target_subdir=ds_cfg.get(
            "component_target_subdir", "components/target"
        ),
        component_interferer_subdir=ds_cfg.get(
            "component_interferer_subdir", "components/interferer"
        ),
        feature_cache_enabled=ds_cfg.get("feature_cache_enabled", False),
        feature_cache_dir=ds_cfg.get("feature_cache_dir", None),
        logger=logger,
    )

    train_root = ds_cfg.get("train_root", None)
    val_root = ds_cfg.get("val_root", None)
    test_root = ds_cfg.get("test_root", None)

    if ds_cfg.get("split_strategy", "ratio") == "binmov_fold":
        train_ds = StaticDOADataset(
            split="train",
            split_folds=ds_cfg.get("train_folds", []),
            **common_kwargs,
        )
        val_ds = StaticDOADataset(
            split="val",
            split_folds=ds_cfg.get("val_folds", []),
            **common_kwargs,
        )
        test_ds = StaticDOADataset(
            split="test",
            split_folds=ds_cfg.get("test_folds", []),
            **common_kwargs,
        )
    elif train_root and val_root and test_root:
        train_ds = StaticDOADataset(split="all", **{**common_kwargs, "root_dir": train_root})
        val_ds = StaticDOADataset(split="all", **{**common_kwargs, "root_dir": val_root})
        test_ds = StaticDOADataset(split="all", **{**common_kwargs, "root_dir": test_root})
    else:
        train_ds = StaticDOADataset(split="train", **common_kwargs)
        val_ds = StaticDOADataset(split="val", **common_kwargs)
        test_ds = StaticDOADataset(split="test", **common_kwargs)

    return train_ds, val_ds, test_ds
