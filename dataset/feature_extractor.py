"""双耳（立体声）音频的特征提取。

提取四种时频表示：
  1. 左耳对数幅度谱
  2. 右耳对数幅度谱
  3. IPD（双耳相位差）
  4. ILD（双耳级差）

所有特征共享相同的时频网格，因此可以堆叠或拼接后传入下游模块。

提取后的张量约定
-----------------------------------
每个特征的形状为 ``[T, F]``（时间帧 x 频率bin）。
经 DataLoader 批处理后变为 ``[B, T, F]``。
通道维度在模型中后续添加。
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class FeatureExtractor:
    """无状态的双耳音频特征提取器。

    参数
    ----------
    n_fft : int
        FFT 大小。
    hop_length : int
        STFT 帧之间的跳跃长度。
    win_length : int
        分析窗口长度。
    window : str
        窗函数类型（支持 ``"hann"``；如需其他类型请自行添加）。
    """

    def __init__(self,
                 n_fft: int = 512,
                 hop_length: int = 160,
                 win_length: int = 400,
                 window: str = "hann"):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        if window == "hann":
            self.window = torch.hann_window(win_length)
        else:
            raise ValueError(f"Unsupported window type: {window}")

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def extract(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """从立体声波形中提取特征。

        参数:
            audio: 形状为 ``[2, num_samples]`` 的张量 — 通道 0 = 左耳，
                   通道 1 = 右耳。

        返回:
            字典，包含以下键：

            * ``"log_mag_L"``  — ``[T, F]``  左耳对数幅度谱
            * ``"log_mag_R"``  — ``[T, F]``  右耳对数幅度谱
            * ``"ipd"``        — ``[T, F]``  双耳相位差
            * ``"ild"``        — ``[T, F]``  双耳级差（dB）
        """
        assert audio.ndim == 2 and audio.shape[0] == 2, \
            f"Expected audio shape [2, N], got {audio.shape}"

        left = audio[0]   # [N]
        right = audio[1]  # [N]

        spec_L = self._stft(left)   # [F, T] 复数
        spec_R = self._stft(right)  # [F, T] 复数

        mag_L = spec_L.abs()  # [F, T]
        mag_R = spec_R.abs()  # [F, T]

        # 对数幅度（加 eps 以保证数值稳定性）
        eps = 1e-8
        log_mag_L = torch.log(mag_L + eps).transpose(0, 1)  # [T, F]
        log_mag_R = torch.log(mag_R + eps).transpose(0, 1)  # [T, F]

        # IPD = angle(spec_L * conj(spec_R))
        cross = spec_L * spec_R.conj()  # [F, T]
        ipd = torch.angle(cross).transpose(0, 1)  # [T, F]，弧度制 [-pi, pi]

        # ILD = 20 * log10(|mag_L| / |mag_R|)
        ild = 20.0 * torch.log10((mag_L + eps) / (mag_R + eps))  # [F, T]
        ild = ild.transpose(0, 1)  # [T, F]

        return {
            "log_mag_L": log_mag_L,   # [T, F]
            "log_mag_R": log_mag_R,   # [T, F]
            "ipd": ipd,               # [T, F]
            "ild": ild,               # [T, F]
        }

    @property
    def num_freq_bins(self) -> int:
        """STFT 产生的频率 bin 数量（n_fft // 2 + 1）。"""
        return self.n_fft // 2 + 1

    def num_time_frames(self, num_samples: int) -> int:
        """根据给定的波形长度计算时间帧数。"""
        return num_samples // self.hop_length + 1

    # ------------------------------------------------------------------
    # 私有辅助方法
    # ------------------------------------------------------------------

    def _stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """计算单声道波形的复数 STFT。

        参数:
            waveform: 一维浮点张量。

        返回:
            形状为 ``[F, T]`` 的复数张量，其中 ``F = n_fft // 2 + 1``。
        """
        win = self.window.to(waveform.device)
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=win,
            return_complex=True,
        )  # [F, T]
        return spec
