from collections import Counter

import numpy as np

from tools.generate_cipic_roomsim25_compound_demand_v1 import (
    CONDITIONS,
    DEMAND_SCENES,
    make_bundles,
    mix_compound,
)


def fake_noise_inventory(count=4000):
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


def fake_demand_inventory():
    return {
        "scenes": {
            scene: {
                "path": f"/demand/{scene}/ch01.wav",
                "eligible_starts_samples": [0, 40000, 80000],
            }
            for scene in DEMAND_SCENES
        }
    }


def test_full_protocol_has_2700_base_realizations_and_16200_clips():
    bundles = make_bundles("full", 20260809, fake_noise_inventory(), fake_demand_inventory())["test"]
    assert len(bundles) == 2700
    assert sum(len(bundle["conditions"]) for bundle in bundles) == 16200
    assert len({bundle["task_id"] for bundle in bundles}) == 2700
    assert all(tuple(bundle["conditions"]) == CONDITIONS for bundle in bundles)


def test_full_protocol_balances_scene_angle_subject_and_two_distances():
    bundles = make_bundles("full", 20260809, fake_noise_inventory(), fake_demand_inventory())["test"]
    assert Counter(row["demand_scene"] for row in bundles) == {scene: 450 for scene in DEMAND_SCENES}
    assert set(Counter(row["target_angle_deg"] for row in bundles).values()) == {108}
    assert set(Counter(row["subject_id"] for row in bundles).values()) == {300}
    assert all(abs(row["target_angle_deg"] - row["noise_angle_deg"]) >= 20 for row in bundles)
    per_subject = {}
    for row in bundles:
        per_subject.setdefault(row["subject_id"], set()).add(row["target_distance_index"])
    assert all(len(values) == 2 for values in per_subject.values())


def test_smoke_protocol_keeps_all_six_conditions():
    bundles = make_bundles("smoke", 20260809, fake_noise_inventory(), fake_demand_inventory())["test"]
    assert len(bundles) == 6
    assert sum(len(bundle["conditions"]) for bundle in bundles) == 36


def test_compound_scaling_is_exact_and_reference_omits_diffuse():
    rng = np.random.default_rng(42)
    target = rng.standard_normal((32000, 2)).astype(np.float32)
    directional = rng.standard_normal((32000, 2)).astype(np.float32)
    diffuse = rng.standard_normal((32000, 2)).astype(np.float32)
    reference, sir, snr, active = mix_compound(target, directional, diffuse, 0.0, None, 16000)
    assert reference.shape == target.shape
    assert abs(sir) < 1e-6
    assert snr is None
    assert 0.0 < active <= 1.0
    compound, sir, snr, _ = mix_compound(target, directional, diffuse, -5.0, 5.0, 16000)
    assert compound.shape == target.shape
    assert abs(sir + 5.0) < 1e-6
    assert abs(snr - 5.0) < 1e-6
