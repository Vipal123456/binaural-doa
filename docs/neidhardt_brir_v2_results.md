# Neidhardt Measured-BRIR External Test Results

## Protocol

- Test set: `data/librispeech_neidhardt_measured_brir_test_v2/test_all`
- Speech is convolved with measured KEMAR 45BA BRIRs; it is not an in-room speech recording.
- Project azimuth uses `wrap(+5 * measurement_index)`. This converts the SOFA azimuth handedness to the convention used by the KEMAR training and test sets.
- The sign conversion was independently verified from receiver-channel energy at +/-90 degrees before formal evaluation.
- Decoding is fixed argmax classification with `angle = -180 + 5 * class`.

Results produced before the handedness correction (approximately 104 degrees MAE for the main model) are invalid and must not be reported.

## Main Model

| Seed | Accuracy | MAE | Acc@5 | Acc@10 | Front/back error | Opposite error |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 36.31% | 44.83 deg | 49.04% | 50.84% | 30.61% | 7.66% |
| 43 | 36.03% | 44.96 deg | 48.95% | 51.41% | 30.50% | 8.08% |
| 44 | 36.55% | 46.84 deg | 46.66% | 48.76% | 30.45% | 8.22% |
| Mean | **36.30%** | **45.54 deg** | **48.22%** | **50.34%** | **30.52%** | **7.98%** |

### Main Model by SNR (three-seed mean)

| SNR | Accuracy | MAE | Acc@10 |
|---|---:|---:|---:|
| clean | 42.48% | 41.15 deg | 56.60% |
| 10 dB | 40.21% | 42.34 deg | 54.86% |
| 5 dB | 38.66% | 43.91 deg | 53.26% |
| 0 dB | 35.83% | 45.28 deg | 50.46% |
| -5 dB | 32.45% | 48.35 deg | 45.93% |
| -10 dB | 28.15% | 52.23 deg | 40.93% |

The three-seed mean MAE over positions 1-4 is 39.27 degrees. Position 5 has a mean MAE of 70.63 degrees. Position 5 is retained in the overall primary metric, but its SOFA geometry metadata has an independently detected 180-degree inconsistency and must be disclosed.

## Model Comparison

| Model | Checkpoint selection | Accuracy | MAE | Acc@5 | Acc@10 | Front/back error |
|---|---|---:|---:|---:|---:|---:|
| Main, seed42 | best validation Accuracy | 36.31% | 44.83 deg | 49.04% | 50.84% | 30.61% |
| Spectral-FB, seed42 | best validation Accuracy | 38.17% | 43.67 deg | 49.09% | 51.53% | 30.27% |
| SDEL, seed43 | legacy `best.pth` (validation MAE) | **39.31%** | 37.30 deg | **54.06%** | 57.13% | **27.88%** |
| FN-SSL, seed44 | best validation Accuracy | 35.58% | **34.67 deg** | 53.17% | **58.52%** | 29.19% |
| DP-RTF, seed42 | legacy `best.pth` (validation MAE) | 27.06% | 53.62 deg | 42.55% | 46.38% | 38.89% |
| BiL, seed42 | legacy `best.pth` (validation MAE) | 27.99% | 60.62 deg | 37.30% | 40.71% | 44.63% |

The comparison is informative but not fully checkpoint-controlled: older SDEL, DP-RTF, and BiL runs did not retain `best_acc.pth`. Formal claims must disclose this difference or retrain those models with the same checkpoint policy.

## Spectral-FB Analysis

On the measured-BRIR test, Spectral-FB improves over the same-seed main model by 1.86 percentage points Accuracy and 1.16 degrees MAE. On the original KEMAR official test, it is worse:

| Model, seed42 | Parameters | KEMAR Accuracy | KEMAR MAE | KEMAR Acc@10 |
|---|---:|---:|---:|---:|
| Main | 152,803 | **96.888%** | **2.284 deg** | **98.148%** |
| Spectral-FB | 163,075 | 96.656% | 2.692 deg | 97.801% |

Therefore Spectral-FB is not a general replacement for the main model based on this seed. It is evidence that an explicit spectral front/back auxiliary branch may slightly improve measured-domain transfer, but it needs more seeds before any robustness claim.

## Interpretation Boundaries

Confirmed: the main model is strong on the matched simulated KEMAR protocol but loses substantial accuracy on measured BRIRs. SDEL and FN-SSL reduce measured-domain angular error, while DP-RTF and BiL perform worse than the main model.

Reasonable interpretation: front/back spectral-cue mismatch, loudspeaker orientation, listener position, and measured-vs-simulated transfer are more important than additive noise alone. This is supported by the clean/noisy gap, position results, and the consistently worse away-facing condition.

Not established: this single measured room does not prove broad real-world generalization, and it also does not prove that the main architecture is intrinsically poor. The result specifically exposes a cross-measurement-domain limitation.
