import numpy as np

from dataset.locata_dataset import _circular_error_deg, _wrap_deg


def test_wrap_deg_uses_half_open_circle():
    values = np.asarray([-540.0, -180.0, 180.0, 540.0, 181.0])
    np.testing.assert_allclose(_wrap_deg(values), [-180.0, -180.0, -180.0, -180.0, -179.0])


def test_circular_error_crosses_180_boundary():
    errors = _circular_error_deg(np.asarray([175.0, -179.0]), np.asarray([-175.0, 179.0]))
    np.testing.assert_allclose(errors, [10.0, 2.0])
