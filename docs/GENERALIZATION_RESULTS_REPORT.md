# Generalization Diagnostics Result Report (2026-04-21)

This report summarizes the completed 3-way diagnostics (2 seeds: 42,43; DIAG_EPOCHS=3):

1. speaker overlap vs speaker disjoint
2. clean-trained vs mixed-trained
3. single-subject vs cross-subject

Source outputs:

- outputs/diagnostics/speaker_overlap_vs_disjoint.md
- outputs/diagnostics/clean_trained_vs_mixed_trained.md
- outputs/diagnostics/single_subject_vs_cross_subject.md

## 1) Core Delta Summary

We use MAE delta as primary sensitivity indicator:

- Speaker sensitivity:
  - Delta_speaker = MAE(disjoint_test) - MAE(overlap_test)
  - = 32.2503 - 34.2873 = -2.0370 deg
- Noise/reverb sensitivity:
  - Delta_noise = MAE(clean_trained_on_robust_test) - MAE(mixed_trained_on_robust_test)
  - = 30.8971 - 12.3871 = +18.5100 deg
- HRTF/generalization sensitivity:
  - Delta_subject = MAE(single_subject_on_unseen_subject_test) - MAE(cross_subject_on_unseen_subject_test)
  - = 36.0037 - 19.6063 = +16.3975 deg

Sensitivity ranking (by MAE delta magnitude):

1. Noise/Reverb robustness (largest, +18.51 deg)
2. Cross-subject HRTF generalization (+16.40 deg)
3. Speaker overlap/disjoint (not dominant in this run)

## 2) Secondary Metrics (error<5 and macro_recall)

- Noise/reverb:
  - error<5 gain: +0.2698 (mixed-trained better)
  - macro_recall gain: +0.1935 (mixed-trained better)
- Subject generalization:
  - error<5 gain: +0.2156 (cross-subject better)
  - macro_recall gain: +0.1481 (cross-subject better)
- Speaker split:
  - error<5 gain: +0.0226 (disjoint slightly better)
  - macro_recall gain: +0.0087 (disjoint slightly better)

Interpretation:

- Noise/reverb and subject generalization both show large and consistent gains under the robust/cross-subject setups.
- Speaker overlap vs disjoint does not show expected degradation in this run; likely split difficulty imbalance or sample-selection effect.

## 3) Practical Decision (What to optimize first)

Based on this diagnostic round:

1. First priority: noise/reverb robustness
2. Second priority: cross-subject HRTF generalization
3. Third priority: speaker disjoint verification (redo with stricter balancing)

## 4) Recommended Next Validation Round

To reduce diagnosis bias from short training:

- Keep same three diagnostics and same metrics.
- Run confirmation with DIAG_EPOCHS=20 and seeds >= 3.
- For speaker diagnostic, enforce stricter split balancing:
  - same number of utterances/chapters per speaker
  - same total recording counts per overlap/disjoint test roots

This should verify whether the current speaker result is stable or an artifact.
