"""共享的轻量 2D CNN 编码器，用于单耳频谱特征提取。

编码器处理对数幅度频谱图，生成紧凑的特征张量，同时保留时间维度。

输入 shape:  [B, 1, T, F]     — 单通道频谱图
输出 shape:  [B, C_out, T', F']  — 其中 T' ≈ T（时间维保留），F' << F

左右耳使用 **同一个** 编码器实例（共享权重）。
"""

import torch
import torch.nn as nn
from typing import List


class BinauralEncoder(nn.Module):
    """轻量 2D CNN，压缩频率轴同时保留时间轴。

    参数
    ----------
    in_channels : int
        输入通道数（单个频谱图为 1）。
    channels : list[int]
        每个卷积块的输出通道数。
        默认 ``[32, 64, 128]`` → 三个卷积块。
    out_dim : int
        每个时间帧的最终输出特征维度（频率维折叠后）。
    dropout : float
        每个卷积块后的 Dropout 概率。
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: List[int] = None,
        out_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]

        blocks = []
        ch_in = in_channels
        for ch_out in channels:
            blocks.append(
                nn.Sequential(
                    # 卷积核 (3,3), 步幅 (1,2): 时间步幅=1, 频率步幅=2
                    nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3),
                              stride=(1, 2), padding=(1, 1)),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout),
                )
            )
            ch_in = ch_out

        self.conv_blocks = nn.Sequential(*blocks)

        # 卷积块之后频率维被缩减了 2^len(channels) 倍。
        # 使用自适应平均池化将 F 压缩到 1，然后投影到 out_dim。
        self.freq_pool = nn.AdaptiveAvgPool2d((None, 1))  # [B, C, T', 1]
        self.proj = nn.Linear(channels[-1], out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: ``[B, 1, T, F]`` — 单耳对数幅度频谱图。

        返回:
            ``[B, T', out_dim]`` — 单耳特征。
        """
        # 输入 shape: [B, 1, T, F]
        h = self.conv_blocks(x)       # [B, C_last, T', F']
        h = self.freq_pool(h)         # [B, C_last, T', 1]
        h = h.squeeze(-1)             # [B, C_last, T']
        h = h.permute(0, 2, 1)        # [B, T', C_last]
        h = self.proj(h)              # [B, T', out_dim]
        return h


class _BalancedEncoderStage(nn.Module):
    """两步式轻量卷积 stage。

    先用一层常规 3x3 卷积做局部特征提取，再用 depthwise separable
    卷积沿频率轴下采样。相比单层 stride 卷积，这种写法更适合在压缩前
    先完成一点去噪/去混响式的局部建模。
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 2),
                padding=(1, 1),
                groups=out_channels,
            ),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_conv(x)
        x = self.downsample(x)
        return x


class BinauralEncoderV2Balanced(nn.Module):
    """更偏鲁棒性的轻量 2D CNN 编码器。

    每个 stage 由两部分组成：
      1. 常规 3x3 卷积（不下采样）先提局部特征
      2. depthwise separable 3x3 卷积沿频率轴下采样

    这样能在保持较低参数量的同时，让前端在频率压缩前多看一眼局部结构。
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: List[int] = None,
        out_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        if channels is None:
            channels = [24, 40, 64]

        blocks = []
        ch_in = in_channels
        for ch_out in channels:
            blocks.append(_BalancedEncoderStage(ch_in, ch_out, dropout))
            ch_in = ch_out

        self.conv_blocks = nn.Sequential(*blocks)
        self.freq_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.proj = nn.Linear(channels[-1], out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_blocks(x)
        h = self.freq_pool(h)
        h = h.squeeze(-1)
        h = h.permute(0, 2, 1)
        h = self.proj(h)
        return h
