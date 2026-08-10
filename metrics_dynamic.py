"""Frame-wise metrics for moving single-speaker DOA sequence estimation."""

from collections import defaultdict
from typing import Dict, Iterable

import numpy as np


def wrap_deg(angle: np.ndarray) -> np.ndarray:
    return ((angle + 180.0) % 360.0) - 180.0


def labels_to_angles(labels: np.ndarray, num_classes: int = 72, azimuth_range=(-180.0, 180.0)) -> np.ndarray:
    labels = np.asarray(labels)
    width = (azimuth_range[1] - azimuth_range[0]) / num_classes
    return azimuth_range[0] + (labels.astype(np.float64) + 0.5) * width


def circular_error(pred_angle: np.ndarray, true_angle: np.ndarray) -> np.ndarray:
    diff = np.abs(wrap_deg(pred_angle - true_angle))
    return np.minimum(diff, 360.0 - diff)


class DynamicDOAMetrics:
    """Accumulate frame-wise sequence metrics and grouped MAE."""

    def __init__(self, num_classes: int = 72, azimuth_range=(-180.0, 180.0)):
        self.num_classes = num_classes
        self.azimuth_range = tuple(azimuth_range)
        self.reset()

    def reset(self) -> None:
        self.pred_cls = []
        self.true_cls = []
        self.true_angles = []
        self.pred_angles = []
        self.jitters = []
        self.groups = defaultdict(list)

    @staticmethod
    def _front(angles: np.ndarray) -> np.ndarray:
        wrapped = wrap_deg(angles)
        return (np.abs(wrapped) <= 90.0).astype(np.int64)

    def update(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        true_angles: np.ndarray,
        group_values: Dict[str, Iterable] = None,
    ) -> None:
        pred_cls = logits.argmax(axis=-1)
        pred_angles = labels_to_angles(pred_cls, self.num_classes, self.azimuth_range)
        err = circular_error(pred_angles, true_angles)

        self.pred_cls.append(pred_cls.reshape(-1))
        self.true_cls.append(labels.reshape(-1))
        self.true_angles.append(true_angles.reshape(-1))
        self.pred_angles.append(pred_angles.reshape(-1))

        if pred_angles.shape[1] > 1:
            self.jitters.append(circular_error(pred_angles[:, 1:], pred_angles[:, :-1]).reshape(-1))

        if group_values:
            for group_name, values in group_values.items():
                for b, value in enumerate(values):
                    self.groups[f"{group_name}_mae/{value}"].extend(err[b].reshape(-1).tolist())

    def compute(self) -> Dict[str, float]:
        if not self.pred_cls:
            return {"frame_mae": float("inf"), "frame_accuracy": 0.0}
        pred_cls = np.concatenate(self.pred_cls)
        true_cls = np.concatenate(self.true_cls)
        pred_angles = np.concatenate(self.pred_angles)
        true_angles = np.concatenate(self.true_angles)
        err = circular_error(pred_angles, true_angles)

        pred_fb = self._front(pred_angles)
        true_fb = self._front(true_angles)
        results = {
            "frame_mae": float(err.mean()),
            "median_angular_error": float(np.median(err)),
            "frame_accuracy": float((pred_cls == true_cls).mean()),
            "acc_at_5deg": float((err <= 5.0).mean()),
            "acc_at_10deg": float((err <= 10.0).mean()),
            "front_back_error_rate": float((pred_fb != true_fb).mean()),
            "large_error_rate": float((err > 90.0).mean()),
            "opposite_error_rate": float((err > 150.0).mean()),
            "jitter": float(np.concatenate(self.jitters).mean()) if self.jitters else 0.0,
        }
        for key, values in self.groups.items():
            if values:
                results[key] = float(np.mean(values))
        return results
