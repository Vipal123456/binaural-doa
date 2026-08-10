from collections import Counter

from tools.generate_cipic_roomsim25_diffuse_v2 import (
    CLASS_ANGLES_DEG,
    SNR_DB,
    TEST_NOISE_SCENES,
    TEST_RT60_MS,
    TRAIN_NOISE_SCENES,
    make_bundles,
)


def test_full_split_counts():
    bundles = make_bundles("full", 42)
    counts = {split: sum(len(bundle["recipes"]) for bundle in values) for split, values in bundles.items()}
    assert counts == {"train": 101250, "val": 9000, "test": 40500}


def test_train_subject_angle_is_balanced_by_snr_and_scene():
    rows = make_bundles("full", 42)["train"][0]["recipes"]
    assert len(rows) == 135
    assert Counter(row["snr_db"] for row in rows) == {value: 27 for value in SNR_DB}
    assert Counter(row["noise_scene"] for row in rows) == {value: 45 for value in TRAIN_NOISE_SCENES}
    assert Counter((row["snr_db"], row["noise_scene"]) for row in rows) == {
        (snr, scene): 9 for snr in SNR_DB for scene in TRAIN_NOISE_SCENES
    }


def test_r0_counts_are_exact():
    bundles = make_bundles("full", 42)
    assert sum(row["rt60_ms"] == 0 for bundle in bundles["train"] for row in bundle["recipes"]) == 10125
    assert sum(row["rt60_ms"] == 0 for bundle in bundles["val"] for row in bundle["recipes"]) == 900


def test_test_grid_is_complete_and_paired():
    bundles = make_bundles("full", 42)["test"]
    assert len(bundles) == 3 * 3 * 25 * len(TEST_NOISE_SCENES) * (4 + 3 + 2) // 3
    assert sum(len(bundle["recipes"]) for bundle in bundles) == 40500
    first = bundles[0]["recipes"]
    assert {(row["rt60_ms"], row["snr_db"]) for row in first} == {
        (rt, snr) for rt in TEST_RT60_MS for snr in SNR_DB
    }
    assert len({row["speech_index"] for row in first}) == 1
    assert len({(row["noise_channel_a"], row["noise_channel_b"], row["noise_u_a"], row["noise_u_b"])
                for row in first}) == 1


def test_every_split_is_class_balanced():
    bundles = make_bundles("full", 42)
    for values in bundles.values():
        counts = Counter(CLASS_ANGLES_DEG.index(row["angle_deg"])
                         for bundle in values for row in bundle["recipes"])
        assert len(set(counts.values())) == 1
