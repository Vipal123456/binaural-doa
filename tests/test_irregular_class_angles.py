import numpy as np
import torch

from losses import DOALoss
from metrics import DOAMetrics
from utils.angle import bins_to_angles


ANGLES = [-80, -65, -55, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 55, 65, 80]


def test_bins_to_nonuniform_angles_uses_lookup_table():
    actual = bins_to_angles(
        np.asarray([0, 1, 12, 23, 24]),
        num_classes=25,
        class_angles_deg=ANGLES,
    )
    np.testing.assert_array_equal(actual, [-80, -65, 0, 65, 80])


def test_nonuniform_metrics_correct_class_has_zero_mae():
    logits = np.eye(25, dtype=np.float32)
    targets = np.arange(25, dtype=np.int64)
    metrics = DOAMetrics(num_classes=25, class_angles_deg=ANGLES)
    metrics.update(logits, targets, np.asarray(ANGLES, dtype=np.float32))
    results = metrics.compute()

    assert results["accuracy"] == 1.0
    assert results["mean_angular_error"] == 0.0
    assert results["acc_at_5deg"] == 1.0


def test_nonuniform_metrics_uses_physical_not_index_distance():
    logits = np.zeros((1, 25), dtype=np.float32)
    logits[0, 1] = 1.0  # -65 deg, adjacent to class 0 but 15 degrees away
    metrics = DOAMetrics(num_classes=25, class_angles_deg=ANGLES)
    metrics.update(logits, np.asarray([0]), np.asarray([-80.0]))
    results = metrics.compute()

    assert results["mean_angular_error"] == 15.0
    assert results["acc_at_10deg"] == 0.0


def test_expected_angle_distance_loss_uses_physical_class_spacing():
    criterion = DOALoss(
        num_classes=25,
        class_angles_deg=ANGLES,
        expected_angle_distance_weight=0.2,
        angular_distance_normalizer_deg=160.0,
    )
    target = torch.tensor([0])  # -80 deg
    near_logits = torch.full((1, 25), -20.0)
    far_logits = torch.full((1, 25), -20.0)
    near_logits[0, 1] = 20.0   # -65 deg: 15-degree error
    far_logits[0, 24] = 20.0   # +80 deg: 160-degree error

    near = criterion(near_logits, target)
    far = criterion(far_logits, target)

    assert abs(near["expected_angle_distance"] - 15.0) < 1.0e-4
    assert abs(far["expected_angle_distance"] - 160.0) < 1.0e-4
    assert far["total"] > near["total"]


def test_frame_auxiliary_loss_accepts_quality_weighted_frame_logits():
    criterion = DOALoss(
        num_classes=25,
        class_angles_deg=ANGLES,
        frame_aux_weight=0.1,
    )
    logits = torch.randn(2, 25, requires_grad=True)
    frame_logits = torch.randn(2, 7, 25, requires_grad=True)
    frame_weights = torch.softmax(torch.randn(2, 7, 1), dim=1)
    targets = torch.tensor([3, 17])

    result = criterion(
        logits,
        targets,
        frame_logits=frame_logits,
        frame_weights=frame_weights,
    )

    assert result["frame_aux"] is not None
    result["total"].backward()
    assert frame_logits.grad is not None
