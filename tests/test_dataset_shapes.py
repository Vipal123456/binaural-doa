"""冒烟测试：特征提取器和数据集的形状验证。

运行方式：  python -m pytest tests/test_dataset_shapes.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pytest

from dataset.feature_extractor import FeatureExtractor
from utils.angle import (
    angle_to_bin, bin_to_angle, bins_to_angles, angular_error, wrap_angle, angles_to_bins,
)


class TestFeatureExtractor:
    """验证特征提取的形状和一致性。"""

    @pytest.fixture
    def fe(self):
        return FeatureExtractor(n_fft=512, hop_length=160, win_length=400)

    def test_output_keys(self, fe):
        audio = torch.randn(2, 16000)  # 1秒的立体声音频
        feats = fe.extract(audio)
        expected_keys = {
            "log_mag_L", "log_mag_R", "ipd", "ild", "ipd_sin", "ipd_cos",
            "coherence", "spec_real_L", "spec_imag_L", "spec_real_R", "spec_imag_R",
        }
        assert set(feats.keys()) == expected_keys

    def test_output_shapes_match(self, fe):
        audio = torch.randn(2, 32000)  # 2秒
        feats = fe.extract(audio)
        T, F = feats["log_mag_L"].shape
        assert feats["log_mag_R"].shape == (T, F)
        assert feats["ipd"].shape == (T, F)
        assert feats["ild"].shape == (T, F)

    def test_freq_bins(self, fe):
        audio = torch.randn(2, 16000)
        feats = fe.extract(audio)
        F = feats["log_mag_L"].shape[1]
        assert F == fe.num_freq_bins  # 512 // 2 + 1 = 257

    def test_time_frames(self, fe):
        num_samples = 32000
        audio = torch.randn(2, num_samples)
        feats = fe.extract(audio)
        T = feats["log_mag_L"].shape[0]
        expected_T = fe.num_time_frames(num_samples)
        assert T == expected_T

    def test_ipd_range(self, fe):
        """IPD 的值域应在 [-pi, pi] 范围内。"""
        audio = torch.randn(2, 16000)
        feats = fe.extract(audio)
        assert feats["ipd"].min() >= -np.pi - 0.01
        assert feats["ipd"].max() <= np.pi + 0.01

    def test_cpsd_cue_preserves_legacy_coherence(self):
        torch.manual_seed(0)
        audio = torch.randn(2, 16000)
        legacy = FeatureExtractor(n_fft=512, hop_length=160, win_length=400)
        cpsd = FeatureExtractor(
            n_fft=512,
            hop_length=160,
            win_length=400,
            spatial_statistics_mode="cpsd_cue",
            spatial_statistics_time_frames=5,
        )
        legacy_feats = legacy.extract(audio)
        cpsd_feats = cpsd.extract(audio)
        assert torch.equal(legacy_feats["coherence"], cpsd_feats["coherence"])
        assert not torch.allclose(legacy_feats["ild"], cpsd_feats["ild"])
        assert not torch.allclose(legacy_feats["ipd"], cpsd_feats["ipd"])

    def test_cpsd_all_outputs_are_finite_and_coherence_is_bounded(self):
        audio = torch.randn(2, 16000)
        fe = FeatureExtractor(
            n_fft=512,
            hop_length=160,
            win_length=400,
            spatial_statistics_mode="cpsd_all",
            spatial_statistics_time_frames=5,
        )
        feats = fe.extract(audio)
        for key in ("ild", "ipd_sin", "ipd_cos", "coherence"):
            assert torch.isfinite(feats[key]).all()
        assert feats["coherence"].min() >= 0.0
        assert feats["coherence"].max() <= 1.0

    @pytest.mark.parametrize("time_frames", [0, 2, 4])
    def test_cpsd_window_must_be_positive_and_odd(self, time_frames):
        with pytest.raises(ValueError):
            FeatureExtractor(
                spatial_statistics_mode="cpsd_cue",
                spatial_statistics_time_frames=time_frames,
            )


class TestAngleUtils:
    """验证角度转换和误差计算。"""

    def test_bin_round_trip(self):
        for angle in np.arange(-180, 180, 5):
            b = angle_to_bin(angle, 72)
            recovered = bin_to_angle(b, 72)
            assert recovered == pytest.approx(float(angle)), \
                f"angle={angle} → bin={b} → recovered={recovered}"

    def test_correct_class_has_zero_angular_error(self):
        bins = np.arange(72)
        angles = np.asarray([bin_to_angle(int(b), 72) for b in bins])
        recovered = bins_to_angles(bins, 72)
        errors = angular_error(recovered, angles)
        assert np.all(errors == 0.0)

    def test_angular_error_wrap(self):
        """在 ±180° 附近的误差应该很小，而不是约 360°。"""
        err = angular_error(np.array([179.0]), np.array([-179.0]))
        assert err[0] < 5.0, f"Expected ~2°, got {err[0]}"

    def test_angular_error_zero(self):
        err = angular_error(np.array([45.0]), np.array([45.0]))
        assert err[0] == 0.0

    def test_wrap_angle(self):
        assert wrap_angle(181) == pytest.approx(-179.0)
        assert wrap_angle(-181) == pytest.approx(179.0)
        assert wrap_angle(0) == pytest.approx(0.0)

    def test_batch_bins(self):
        angles = np.array([-180, -90, 0, 90, 179], dtype=np.float64)
        bins = angles_to_bins(angles, 72)
        assert len(bins) == 5
        assert all(0 <= b < 72 for b in bins)
