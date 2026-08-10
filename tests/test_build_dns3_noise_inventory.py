import numpy as np

from tools.build_dns3_noise_inventory import (
    TARGET_SAMPLE_RATE,
    descendants,
    source_identity,
    valid_window_starts,
)


def test_descendants_walks_complete_hierarchy():
    nodes = {
        "root": {"child_ids": ["a", "b"]},
        "a": {"child_ids": ["c"]},
        "b": {"child_ids": []},
        "c": {"child_ids": []},
    }
    assert descendants(nodes, "root") == {"root", "a", "b", "c"}


def test_source_identity_groups_freesound_chunks():
    from pathlib import Path

    first = source_identity(Path("breath_spit_Freesound_validated_422419_0.wav"))
    second = source_identity(Path("other_Freesound_validated_422419_7.wav"))
    assert first == ("freesound", "freesound:422419")
    assert second == first
    named = source_identity(Path("munching_Freesound_validated_B32_Chips_ebrahim_10.wav"))
    assert named == ("freesound", "freesound:munching_Freesound_validated_B32_Chips_ebrahim")
    assert source_identity(Path("x8fp6MYE41s.wav")) == ("audioset", "audioset:x8fp6MYE41s")


def test_constant_dc_tail_is_not_an_active_noise_event():
    signal = np.full(4 * TARGET_SAMPLE_RATE, 0.02, dtype=np.float32)
    assert valid_window_starts(signal) == []


def test_sparse_end_transient_is_not_mistaken_for_continuous_activity():
    signal = np.zeros(4 * TARGET_SAMPLE_RATE, dtype=np.float32)
    transient_start = 3 * TARGET_SAMPLE_RATE - int(0.009 * TARGET_SAMPLE_RATE)
    signal[transient_start : 3 * TARGET_SAMPLE_RATE] = np.linspace(
        -0.95, 0.95, 3 * TARGET_SAMPLE_RATE - transient_start, dtype=np.float32
    )
    assert valid_window_starts(signal) == []


def test_continuous_noise_with_dc_offset_has_valid_windows():
    rng = np.random.default_rng(42)
    signal = 0.02 + 0.03 * rng.standard_normal(4 * TARGET_SAMPLE_RATE).astype(np.float32)
    assert valid_window_starts(signal)
