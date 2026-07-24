"""角度运算和 DOA-bin 转换工具。

除非特别说明，所有角度均以 **度** 为单位。
方位角范围约定：[-180, 180)。
"""

import numpy as np
import torch
from typing import Tuple


# ======================================================================
# Bin <-> 角度转换
# ======================================================================

def angle_to_bin(angle_deg: float,
                 num_classes: int = 72,
                 azimuth_range: Tuple[float, float] = (-180.0, 180.0)) -> int:
    """将连续方位角转换为最近的分类 bin 索引。

    参数:
        angle_deg: 方位角（度）。
        num_classes: 离散 bin 数量。
        azimuth_range: 方位角范围 (min, max)。

    返回:
        ``[0, num_classes)`` 范围内的整数 bin 索引。
    """
    lo, hi = azimuth_range
    span = hi - lo  # 例如 360
    bin_width = span / num_classes
    # 将角度包裹到 [lo, hi)
    angle_deg = wrap_angle(angle_deg, lo, hi)
    bin_idx = int((angle_deg - lo) / bin_width)
    # 防止浮点边界溢出
    return min(max(bin_idx, 0), num_classes - 1)


def bin_to_angle(bin_idx: int,
                 num_classes: int = 72,
                 azimuth_range: Tuple[float, float] = (-180.0, 180.0)) -> float:
    """将 bin 索引转换为对应的离散方向点。

    参数:
        bin_idx: 类别索引。
        num_classes: bin 数量。
        azimuth_range: (min, max)。

    返回:
        离散方向角（度）。

    Notes
    -----
    当任务的类别定义为 ``[-180, -175, ..., 175]`` 这类固定
    DOA 候选点时，类别 0 对应 ``-180`` 度，而不是区间中心
    ``-177.5`` 度。这保证正确分类的角误差为 0 度。
    """
    lo, hi = azimuth_range
    bin_width = (hi - lo) / num_classes
    return lo + bin_idx * bin_width


def angles_to_bins(angles: np.ndarray,
                   num_classes: int = 72,
                   azimuth_range: Tuple[float, float] = (-180.0, 180.0)) -> np.ndarray:
    """:func:`angle_to_bin` 的向量化版本。"""
    lo, hi = azimuth_range
    span = hi - lo
    bin_width = span / num_classes
    angles = wrap_angles(angles, lo, hi)
    bins = ((angles - lo) / bin_width).astype(np.int64)
    return np.clip(bins, 0, num_classes - 1)


def bins_to_angles(bins: np.ndarray,
                   num_classes: int = 72,
                   azimuth_range: Tuple[float, float] = (-180.0, 180.0)) -> np.ndarray:
    """:func:`bin_to_angle` 的向量化版本。"""
    lo, hi = azimuth_range
    bin_width = (hi - lo) / num_classes
    return lo + bins * bin_width


# ======================================================================
# 角度包裹 / 误差计算
# ======================================================================

def wrap_angle(angle: float,
               lo: float = -180.0,
               hi: float = 180.0) -> float:
    """将 *angle* 包裹到 ``[lo, hi)`` 范围。"""
    span = hi - lo
    return lo + (angle - lo) % span


def wrap_angles(angles: np.ndarray,
                lo: float = -180.0,
                hi: float = 180.0) -> np.ndarray:
    """:func:`wrap_angle` 的向量化版本。"""
    span = hi - lo
    return lo + (angles - lo) % span


def angular_error(pred_deg: np.ndarray,
                  target_deg: np.ndarray) -> np.ndarray:
    """计算考虑圆周包裹的绝对角度误差。

    结果始终在 ``[0, 180]`` 范围内。

    参数:
        pred_deg: 预测角度（度）。
        target_deg: 真实角度（度）。

    返回:
        逐元素绝对角度误差（度）。
    """
    diff = pred_deg - target_deg
    # 包裹到 [-180, 180)
    diff = (diff + 180) % 360 - 180
    return np.abs(diff)


def angular_error_torch(pred_deg: torch.Tensor,
                        target_deg: torch.Tensor) -> torch.Tensor:
    """:func:`angular_error` 的 PyTorch 版本。"""
    diff = pred_deg - target_deg
    diff = (diff + 180) % 360 - 180
    return diff.abs()
