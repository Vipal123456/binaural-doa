import numpy as np

from evaluate_locata import _decode_mirror_pairs, _mirror_pair_groups


def test_mirror_pair_groups_cover_all_72_classes_once():
    lateral_angles, groups = _mirror_pair_groups(72, (-180.0, 180.0))

    assert lateral_angles.shape == (37,)
    np.testing.assert_array_equal(np.sort(np.concatenate(groups)), np.arange(72))
    assert sorted(group.size for group in groups).count(1) == 2
    assert sorted(group.size for group in groups).count(2) == 35


def test_mirror_pair_decoder_separates_lateral_and_front_back_decisions():
    probabilities = np.zeros((1, 72), dtype=np.float64)
    front_bin = 40  # +20 degrees
    back_bin = 68   # +160 degrees, the front/back mirror of +20 degrees
    probabilities[0, front_bin] = 0.4
    probabilities[0, back_bin] = 0.5
    probabilities[0, 42] = 0.6  # +30 degrees, weaker after mirror-pair summation

    local_bins, _ = _decode_mirror_pairs(probabilities, 72, (-180.0, 180.0))
    front_bins, _ = _decode_mirror_pairs(
        probabilities,
        72,
        (-180.0, 180.0),
        np.asarray([0]),
    )
    back_bins, _ = _decode_mirror_pairs(
        probabilities,
        72,
        (-180.0, 180.0),
        np.asarray([1]),
    )

    np.testing.assert_array_equal(local_bins, [back_bin])
    np.testing.assert_array_equal(front_bins, [front_bin])
    np.testing.assert_array_equal(back_bins, [back_bin])

