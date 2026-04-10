"""DOA 分类的评估指标。

指标:
  1. 分类准确率
  2. Top-k 准确率
  3. 平均角度误差（MAE，单位为度，考虑圆周特性）
"""

import numpy as np
from typing import Dict, Optional, Tuple

from utils.angle import bins_to_angles, angular_error


class DOAMetrics:
    """在一个 epoch 上累积预测并计算指标。

    参数
    ----------
    num_classes : int
        方位角分箱数量。
    azimuth_range : tuple
        方位角范围 ``(min, max)``，单位为度。
    top_k : int
        Top-k 准确率中的 k 值。
    """

    def __init__(
        self,
        num_classes: int = 72,
        azimuth_range: Tuple[float, float] = (-180.0, 180.0),
        top_k: int = 3,
    ):
        self.num_classes = num_classes
        self.azimuth_range = azimuth_range
        self.top_k = top_k
        self.reset()

    def reset(self) -> None:
        """清除累积的统计数据。"""
        self._pred_bins = []
        self._true_bins = []
        self._true_degs = []
        self._pred_degs = []  # 回归预测的连续角度
        self._logits = []

    def update(
        self,
        logits: np.ndarray,
        true_bins: np.ndarray,
        true_degs: Optional[np.ndarray] = None,
        pred_degs: Optional[np.ndarray] = None,
    ) -> None:
        """添加一个批次的预测结果。

        参数:
            logits:    ``[B, C]`` 原始 logits（numpy 数组）。
            true_bins: ``[B]`` 整数分箱标签。
            true_degs: ``[B]`` 连续方位角，单位为度（可选）。
            pred_degs: ``[B]`` 回归预测的连续角度，单位为度（可选）。
        """
        pred_bins = logits.argmax(axis=-1)  # [B]
        self._pred_bins.append(pred_bins)
        self._true_bins.append(true_bins)
        self._logits.append(logits)
        if true_degs is not None:
            self._true_degs.append(true_degs)
        if pred_degs is not None:
            self._pred_degs.append(pred_degs)

    def compute(self) -> Dict[str, float]:
        """对累积数据计算所有指标。

        返回:
            包含 ``"accuracy"``、``"top_k_accuracy"``、
            ``"mean_angular_error"`` 等键的字典。
        """
        # 处理空验证集的情况
        if len(self._pred_bins) == 0:
            return {
                "accuracy": 0.0,
                "top_k_accuracy": 0.0,
                "mean_angular_error": float("inf"),
                "median_angular_error": float("inf"),
            }

        pred_bins = np.concatenate(self._pred_bins)
        true_bins = np.concatenate(self._true_bins)
        logits = np.concatenate(self._logits)

        N = len(pred_bins)
        results: Dict[str, float] = {}

        # --- 1. 准确率 ---
        results["accuracy"] = float((pred_bins == true_bins).sum()) / max(N, 1)

        # --- 2. Top-k 准确率 ---
        top_k_preds = np.argsort(logits, axis=-1)[:, -self.top_k:]  # [N, k]
        top_k_hit = np.array([
            true_bins[i] in top_k_preds[i] for i in range(N)
        ])
        results["top_k_accuracy"] = float(top_k_hit.sum()) / max(N, 1)

        # --- 3. 平均角度误差 ---
        # 分类bin预测角度
        cls_pred_degs = bins_to_angles(pred_bins, self.num_classes, self.azimuth_range)

        # 混合预测策略：分类粗定位 + 回归微调
        if len(self._pred_degs) > 0:
            reg_pred_degs = np.concatenate(self._pred_degs)

            # 计算分类和回归预测的角度差
            diff = reg_pred_degs - cls_pred_degs
            diff = (diff + 180) % 360 - 180  # 归一化到 [-180, 180)
            abs_diff = np.abs(diff)

            # 混合策略：如果回归和分类预测接近（<10°），用回归；否则用分类
            threshold = 10.0
            use_regression = abs_diff < threshold

            # 混合预测结果
            pred_degs = np.where(use_regression, reg_pred_degs, cls_pred_degs)

            # 记录使用回归的比例
            regression_ratio = use_regression.mean()
        else:
            # 无回归预测时，使用分类预测
            pred_degs = cls_pred_degs
            regression_ratio = 0.0

        if len(self._true_degs) > 0:
            true_degs = np.concatenate(self._true_degs)
        else:
            true_degs = bins_to_angles(true_bins, self.num_classes, self.azimuth_range)

        ang_err = angular_error(pred_degs, true_degs)  # [N]
        results["mean_angular_error"] = float(ang_err.mean())
        results["median_angular_error"] = float(np.median(ang_err))
        results["std_angular_error"] = float(ang_err.std())
        results["max_angular_error"] = float(ang_err.max())

        # --- 4. 误差分布统计 ---
        results["error_lt_5"] = float((ang_err < 5).sum()) / max(N, 1)
        results["error_lt_10"] = float((ang_err < 10).sum()) / max(N, 1)
        results["error_lt_20"] = float((ang_err < 20).sum()) / max(N, 1)
        results["error_lt_30"] = float((ang_err < 30).sum()) / max(N, 1)

        # --- 5. 混合预测统计 ---
        if len(self._pred_degs) > 0:
            results["regression_usage_ratio"] = float(regression_ratio)

        return results

    def confusion_matrix(self) -> np.ndarray:
        """构建 ``(num_classes, num_classes)`` 混淆矩阵。"""
        pred_bins = np.concatenate(self._pred_bins)
        true_bins = np.concatenate(self._true_bins)
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for t, p in zip(true_bins, pred_bins):
            cm[t, p] += 1
        return cm
