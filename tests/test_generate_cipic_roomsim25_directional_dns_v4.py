from collections import Counter

import numpy as np

from tools.generate_cipic_roomsim25_directional_dns_v4 import (
    CLASS_ANGLES_DEG,
    EVAL_SIR_DB,
    TEST_CONDITIONS,
    TRAIN_SIR_BINS,
    make_bundles,
    mix_at_active_sir,
    noise_angle_schedule,
)


def test_test_conditions_use_milliseconds() -> None:
    """Keep protocol constants explicit when comparing them to rt60_s metadata."""
    assert {(rt / 1000.0, sir) for rt, sir in TEST_CONDITIONS} == {
        (0.6, -5),
        (0.6, 0),
        (0.6, 5),
        (0.6, 10),
        (0.6, 15),
        (0.2, 5),
        (0.4, 5),
        (0.8, 5),
    }


def fake_noise_inventory(count=97):
    return {
        split: [
            {
                "path": f"/{split}/noise_{index:04d}.wav",
                "source_id": f"{split}:source_{index:04d}",
                "source_kind": "test",
                "active_starts_sec": [1.0, 2.0, 3.0],
            }
            for index in range(count)
        ]
        for split in ("train", "val", "test")
    }


def test_full_counts_match_fixed_protocol():
    bundles = make_bundles("full", 42, fake_noise_inventory())
    counts = {
        split: sum(len(bundle["recipes"]) for bundle in values)
        for split, values in bundles.items()
    }
    assert counts == {"train": 120000, "val": 12000, "test": 64800}
    assert len(bundles["test"]) == 8100
    assert len({bundle["task_id"] for bundle in bundles["test"]}) == 8100
    assert all(len(bundle["recipes"]) == len(TEST_CONDITIONS) for bundle in bundles["test"])


def test_train_sir_strata_and_validation_levels_are_balanced():
    bundles = make_bundles("full", 42, fake_noise_inventory())
    train_rows = bundles["train"][0]["recipes"]
    bin_counts = Counter(
        next(index for index, (low, high) in enumerate(TRAIN_SIR_BINS) if low <= row["sir_db"] < high)
        for row in train_rows
    )
    assert bin_counts == {0: 40, 1: 40, 2: 40, 3: 40}
    val_rows = bundles["val"][0]["recipes"]
    assert Counter(row["sir_db"] for row in val_rows) == {float(value): 16 for value in EVAL_SIR_DB}


def test_test_grid_is_complete_and_content_is_paired():
    rows = make_bundles("full", 42, fake_noise_inventory())["test"][0]["recipes"]
    assert {(row["rt60_ms"], int(row["sir_db"])) for row in rows} == set(TEST_CONDITIONS)
    shared_fields = (
        "subject_id", "target_angle_deg", "noise_angle_deg", "target_distance_index",
        "noise_distance_index", "speech_index", "noise_path", "noise_target_start_sec",
    )
    assert all(len({row[field] for row in rows}) == 1 for field in shared_fields)


def test_noise_angles_have_minimum_separation_and_cover_available_strata():
    for target in CLASS_ANGLES_DEG:
        values = noise_angle_schedule(target, 120, 42 + target)
        assert min(abs(value - target) for value in values) >= 20
        assert len(set(values)) >= 3


def test_active_sir_scaling_is_exact_and_uses_one_binaural_gain():
    rng = np.random.default_rng(42)
    target = rng.standard_normal((32000, 2)).astype(np.float32)
    interferer = rng.standard_normal((32000, 2)).astype(np.float32)
    mixed, achieved, active_ratio = mix_at_active_sir(target, interferer, -5.0, 16000)
    assert mixed.shape == target.shape
    assert abs(achieved + 5.0) < 1e-6
    assert 0.0 < active_ratio <= 1.0
