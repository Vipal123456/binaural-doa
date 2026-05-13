"""DOA 分类的评估指标。"""

import numpy as np
from typing import Dict, Optional, Tuple

from utils.angle import bins_to_angles, angular_error, wrap_angles


class DOAMetrics:
    """在一个 epoch 上累积预测并计算指标。

    参数
    ----------
    num_classes : int
        方位角分箱数量。
    azimuth_range : tuple
        方位角范围 ``(min, max)``，单位为度。
    """

    def __init__(
        self,
        num_classes: int = 72,
        azimuth_range: Tuple[float, float] = (-180.0, 180.0),
    ):
        self.num_classes = num_classes
        self.azimuth_range = azimuth_range
        self.reset()

    def reset(self) -> None:
        """清除累积的统计数据。"""
        self._pred_bins = []
        self._true_bins = []
        self._true_degs = []
        self._pred_degs = []  # 回归预测的连续角度
        self._logits = []

    @staticmethod
    def _front_back_labels(angles_deg: np.ndarray) -> np.ndarray:
        """前后半平面标签。

        约定:
        - front = 0: [-90°, 90°]
        - back = 1:  其余区间
        """
        wrapped = wrap_angles(np.asarray(angles_deg, dtype=np.float64))
        return (np.abs(wrapped) > 90.0).astype(np.int64)

    @staticmethod
    def _region_labels(angles_deg: np.ndarray) -> np.ndarray:
        """将角度划分为 front / side / back 三个区域。

        为了让 front/back/side MAE 形成互斥分区，这里使用:
        - front: |angle| <= 60°
        - side:  60° < |angle| < 120°
        - back:  |angle| >= 120°
        """
        wrapped = wrap_angles(np.asarray(angles_deg, dtype=np.float64))
        abs_angles = np.abs(wrapped)
        regions = np.full(abs_angles.shape, "side", dtype=object)
        regions[abs_angles <= 60.0] = "front"
        regions[abs_angles >= 120.0] = "back"
        return regions

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
            包含 ``"accuracy"``、``"f1_score"``、
            ``"mean_angular_error"`` 等键的字典。
        """
        # 处理空验证集的情况
        if len(self._pred_bins) == 0:
            return {
                "accuracy": 0.0,
                "f1_score": 0.0,
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

        # --- 1.1 宏平均 P/R/F1 ---
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for t, p in zip(true_bins, pred_bins):
            cm[t, p] += 1

        tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(axis=0).astype(np.float64) - tp
        fn = cm.sum(axis=1).astype(np.float64) - tp

        precision_per_class = np.divide(
            tp,
            tp + fp,
            out=np.zeros_like(tp, dtype=np.float64),
            where=(tp + fp) > 0,
        )
        recall_per_class = np.divide(
            tp,
            tp + fn,
            out=np.zeros_like(tp, dtype=np.float64),
            where=(tp + fn) > 0,
        )
        f1_per_class = np.divide(
            2.0 * precision_per_class * recall_per_class,
            precision_per_class + recall_per_class,
            out=np.zeros_like(tp, dtype=np.float64),
            where=(precision_per_class + recall_per_class) > 0,
        )

        results["macro_precision"] = float(precision_per_class.mean())
        results["macro_recall"] = float(recall_per_class.mean())
        # 宏平均 F1-score，适合观察 72 个角度类别上的整体均衡性。
        results["f1_score"] = float(f1_per_class.mean())

        # --- 2. 平均角度误差 ---
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

        # --- 3. 误差分布统计 ---
        # 容忍度准确率：允许预测角度与真值相差不超过给定阈值。
        results["acc_at_5deg"] = float((ang_err <= 5.0).sum()) / max(N, 1)
        results["acc_at_10deg"] = float((ang_err <= 10.0).sum()) / max(N, 1)
        results["acc_at_20deg"] = float((ang_err <= 20.0).sum()) / max(N, 1)
        results["acc_at_30deg"] = float((ang_err <= 30.0).sum()) / max(N, 1)

        # --- 3.1 诊断性错误统计 ---
        pred_fb = self._front_back_labels(pred_degs)
        true_fb = self._front_back_labels(true_degs)
        results["front_back_halfplane_error_rate"] = float((pred_fb != true_fb).sum()) / max(N, 1)

        circular_bin_diff = np.abs(pred_bins - true_bins)
        circular_bin_diff = np.minimum(circular_bin_diff, self.num_classes - circular_bin_diff)
        half_bins = self.num_classes // 2

        # near-bin: 落在真实bin及其相邻bin内（圆周意义下）
        results["within_1bin_acc"] = float((circular_bin_diff <= 1).sum()) / max(N, 1)

        # opposite: 典型180°翻转，允许 ±1 bin 容差
        results["opposite_error_rate"] = float(
            (np.abs(circular_bin_diff - half_bins) <= 1).sum()
        ) / max(N, 1)

        # large error: 明显大错，默认定义为角误差 >= 45°
        results["large_error_rate"] = float((ang_err >= 45.0).sum()) / max(N, 1)

        # --- 3.2 按空间区域统计 MAE ---
        region_labels = self._region_labels(true_degs)
        for region_name in ("front", "back", "side"):
            mask = region_labels == region_name
            if mask.any():
                results[f"{region_name}_mae"] = float(ang_err[mask].mean())
            else:
                results[f"{region_name}_mae"] = float("nan")

        # --- 4. 混合预测统计 ---
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
