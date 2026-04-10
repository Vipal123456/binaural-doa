"""双耳音频数据增强模块。

提供多种音频/频谱增强方法以提升模型泛化能力。
"""

import torch
import torch.nn as nn
import numpy as np


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

        B, F, T = batch['log_mag_L'].shape

        # 1. 时间掩蔽（在所有特征上应用相同的掩蔽）
        if self.time_mask_max_frames > 0 and torch.rand(1).item() < 0.5:
            for b in range(B):
                t_len = np.random.randint(0, min(self.time_mask_max_frames, T // 2))
                if t_len > 0:
                    t_start = np.random.randint(0, T - t_len)
                    batch['log_mag_L'][b, :, t_start:t_start + t_len] = 0
                    batch['log_mag_R'][b, :, t_start:t_start + t_len] = 0
                    batch['ipd'][b, :, t_start:t_start + t_len] = 0
                    batch['ild'][b, :, t_start:t_start + t_len] = 0

        # 2. 频率掩蔽
        if self.freq_mask_max_bins > 0 and torch.rand(1).item() < 0.5:
            for b in range(B):
                f_len = np.random.randint(0, min(self.freq_mask_max_bins, F // 4))
                if f_len > 0:
                    f_start = np.random.randint(0, F - f_len)
                    batch['log_mag_L'][b, f_start:f_start + f_len, :] = 0
                    batch['log_mag_R'][b, f_start:f_start + f_len, :] = 0
                    batch['ipd'][b, f_start:f_start + f_len, :] = 0
                    batch['ild'][b, f_start:f_start + f_len, :] = 0

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

        # 5. ILD噪声（能量差扰动）
        if self.ild_noise_std > 0 and torch.rand(1).item() < 0.3:
            ild_noise = torch.randn_like(batch['ild']) * self.ild_noise_std
            batch['ild'] = batch['ild'] + ild_noise

        return batch


class MixupAugmentation:
    """Mixup数据增强（在batch层面）。

    将两个样本及其标签按比例混合。

    参数
    ----------
    alpha : float
        Beta分布参数，控制混合比例的分布。
    prob : float
        应用mixup的概率。
    """

    def __init__(self, alpha=0.2, prob=0.3):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, batch):
        """应用mixup增强。

        参数
        ----------
        batch : dict
            包含特征和标签的batch。

        返回
        -------
        dict
            混合后的batch（标签不变，由loss函数处理）。
        """
        if self.alpha <= 0 or torch.rand(1).item() > self.prob:
            return batch

        B = batch['log_mag_L'].size(0)
        lam = np.random.beta(self.alpha, self.alpha)

        # 生成随机排列索引
        index = torch.randperm(B, device=batch['log_mag_L'].device)

        # 混合特征
        batch['log_mag_L'] = lam * batch['log_mag_L'] + (1 - lam) * batch['log_mag_L'][index]
        batch['log_mag_R'] = lam * batch['log_mag_R'] + (1 - lam) * batch['log_mag_R'][index]
        batch['ipd'] = lam * batch['ipd'] + (1 - lam) * batch['ipd'][index]
        batch['ild'] = lam * batch['ild'] + (1 - lam) * batch['ild'][index]

        # 注意：对于方位角这种回归任务，mixup标签混合可能不太合适
        # 这里保持原标签不变，让模型学习混合特征下的定位

        return batch
