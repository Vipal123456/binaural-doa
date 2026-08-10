from tools.evaluate_cipic_compound_grouped import (
    CONDITION_ORDER,
    paired_reference_summary,
    separation_bin,
)


def test_separation_bins_match_protocol():
    assert separation_bin(20) == "20_40"
    assert separation_bin(40) == "20_40"
    assert separation_bin(45) == "45_80"
    assert separation_bin(80) == "45_80"
    assert separation_bin(85) == "85_160"


def test_paired_reference_summary_uses_matching_realizations():
    rows = []
    for key, reference in (("a", 2.0), ("b", 4.0)):
        for index, condition in enumerate(CONDITION_ORDER):
            rows.append({
                "paired_test_key": key,
                "condition": condition,
                "error_deg": reference + index,
            })
    result = {row["condition"]: row for row in paired_reference_summary(rows)}
    assert result[CONDITION_ORDER[0]]["delta_mae_vs_reference"] == 0.0
    assert result[CONDITION_ORDER[2]]["delta_mae_vs_reference"] == 2.0
    assert result[CONDITION_ORDER[2]]["worsened_fraction"] == 1.0
