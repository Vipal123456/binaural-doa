from collections import Counter
from pathlib import Path

import numpy as np

from tools.generate_cipic_roomsim25 import (
    CLASS_ANGLES_DEG,
    ROOM_BY_SUBJECT,
    SPLIT_SUBJECTS,
    balanced_schedule,
    brir_path,
    brir_source_index,
    make_tasks,
    mix_at_snr,
    project_to_rir_angle,
)


def test_project_angle_is_mirrored_for_roomsim_convention():
    assert project_to_rir_angle(80) == -80
    assert project_to_rir_angle(-80) == 80
    assert brir_source_index(0, 80) == 3
    assert brir_source_index(0, -80) == 35
    assert brir_source_index(1, 80) == 40


def test_brir_path_uses_subject_room_rt_distance_and_mirrored_angle():
    room = ROOM_BY_SUBJECT["003"]
    path = brir_path(Path("/rir"), "003", room, 600, 2, 80)
    assert path == Path("/rir/H003_507030/R600_S77.mat")


def test_balanced_schedule_differs_by_at_most_one():
    schedule = balanced_schedule((-5, 0, 5, 10, 15), 163, 42)
    counts = Counter(schedule)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_full_task_counts_and_subject_disjointness():
    tasks = make_tasks("full", 42)
    assert len(tasks["train"]) == 30 * 160
    assert len(tasks["val"]) == 6 * 80
    assert len(tasks["test"]) == 2592
    groups = [set(SPLIT_SUBJECTS[name]) for name in ("train", "val", "test")]
    assert not (groups[0] & groups[1])
    assert not (groups[0] & groups[2])
    assert not (groups[1] & groups[2])


def test_train_snr_is_exactly_balanced_per_subject():
    tasks = make_tasks("full", 42)["train"]
    first_subject = SPLIT_SUBJECTS["train"][0]
    values = [task["snr_db"] for task in tasks if task["subject_id"] == first_subject]
    assert Counter(values) == {-5: 32, 0: 32, 5: 32, 10: 32, 15: 32}


def test_mix_at_snr_uses_joint_stereo_power():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=(3200, 2)).astype(np.float32)
    noise = rng.normal(size=(3200, 2)).astype(np.float32)
    _mixed, achieved = mix_at_snr(signal, noise, -5.0)
    assert abs(achieved + 5.0) < 1e-6
