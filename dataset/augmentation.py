"""双耳音频数据增强模块。

提供多种音频/频谱增强方法以提升模型泛化能力。
"""

import torch
import numpy as np
import torch.nn as nn


class BinauralAugmentation(nn.Module):
    """双耳音频特征增强。

    在频谱特征上进行增强，包括：
    - 时间掩蔽（Time Masking）
    - 频率掩蔽（Frequency Masking）
    - 强度扰动（Magnitude Perturbation）
    - IPD/ILD噪声（相位和能量差扰动）

    参数
    ----------
    time_mask_max_frames : int
        时间掩蔽的最大帧数。
    freq_mask_max_bins : int
        频率掩蔽的最大频率bin数。
    magnitude_std : float
        幅度扰动的标准差。
    ipd_noise_std : float
        IPD噪声标准差（弧度）。
    ild_noise_std : float
        ILD噪声标准差（dB）。
    prob : float
        应用增强的概率（0-1）。
    """

    def __init__(
        self,
        time_mask_max_frames=20,
        freq_mask_max_bins=8,
        magnitude_std=0.1,
        ipd_noise_std=0.05,
        ild_noise_std=0.5,
        prob=0.5,
    ):
        super().__init__()
        self.time_mask_max_frames = time_mask_max_frames
        self.freq_mask_max_bins = freq_mask_max_bins
        self.magnitude_std = magnitude_std
        self.ipd_noise_std = ipd_noise_std
        self.ild_noise_std = ild_noise_std
        self.prob = prob

    def forward(self, batch):
        """应用数据增强。

        参数
        ----------
        batch : dict
            包含 'log_mag_L', 'log_mag_R', 'ipd', 'ild' 的batch字典。

        返回
        -------
        dict
            增强后的batch。
        """
        if not self.training or torch.rand(1).item() > self.prob:
            return batch

        # 复制batch以避免修改原始数据
        batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        B, T, F = batch['log_mag_L'].shape

        feature_keys = ['log_mag_L', 'log_mag_R', 'ipd', 'ild']
        if 'ipd_sin' in batch:
            feature_keys.append('ipd_sin')
        if 'ipd_cos' in batch:
            feature_keys.append('ipd_cos')
        if 'coherence' in batch:
            feature_keys.append('coherence')

        # 1. 时间掩蔽（在所有特征上应用相同的掩蔽）
        if self.time_mask_max_frames > 0 and torch.rand(1).item() < 0.5:
            for b in range(B):
                t_len = np.random.randint(0, min(self.time_mask_max_frames, T // 2))
                if t_len > 0:
                    t_start = np.random.randint(0, T - t_len)
                    for key in feature_keys:
                        batch[key][b, t_start:t_start + t_len, :] = 0

        # 2. 频率掩蔽
        if self.freq_mask_max_bins > 0 and torch.rand(1).item() < 0.5:
            for b in range(B):
                f_len = np.random.randint(0, min(self.freq_mask_max_bins, F // 4))
                if f_len > 0:
                    f_start = np.random.randint(0, F - f_len)
                    for key in feature_keys:
                        batch[key][b, :, f_start:f_start + f_len] = 0

        # 3. 幅度扰动
        if self.magnitude_std > 0 and torch.rand(1).item() < 0.5:
            noise_L = torch.randn_like(batch['log_mag_L']) * self.magnitude_std
            noise_R = torch.randn_like(batch['log_mag_R']) * self.magnitude_std
            batch['log_mag_L'] = batch['log_mag_L'] + noise_L
            batch['log_mag_R'] = batch['log_mag_R'] + noise_R

        # 4. IPD噪声（相位差扰动）
        if self.ipd_noise_std > 0 and torch.rand(1).item() < 0.3:
            ipd_noise = torch.randn_like(batch['ipd']) * self.ipd_noise_std
            batch['ipd'] = batch['ipd'] + ipd_noise
            # IPD范围限制在[-π, π]
            batch['ipd'] = torch.clamp(batch['ipd'], -np.pi, np.pi)
            if 'ipd_sin' in batch:
                batch['ipd_sin'] = torch.sin(batch['ipd'])
            if 'ipd_cos' in batch:
                batch['ipd_cos'] = torch.cos(batch['ipd'])

        # 5. ILD噪声（能量差扰动）
        if self.ild_noise_std > 0 and torch.rand(1).item() < 0.3:
            ild_noise = torch.randn_like(batch['ild']) * self.ild_noise_std
            batch['ild'] = batch['ild'] + ild_noise

        return batch
