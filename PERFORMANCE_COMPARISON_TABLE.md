# Performance Comparison Table

This file is an editable summary table for the main experiments in this repository.
It focuses on the `robust50h multisubject / subject-disjoint unseen-subject` protocol.

## Main Table

| Model | Family | Input Features | Backbone | Output | Training Objective | Params | Accuracy | Top-3 | MAE | Median AE | FB Err | Opp Err | Large Err | Error < 10° | Notes |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DOA-Net mainline cls | Native DOA-Net | `log_mag_L/R + IPD + ILD + sin/cos(IPD) + coherence` | Shared binaural encoder + difference prior + cross-attention + attention bias + gating + BiGRU + attention pooling | `72-class logits + fbaux` | CE + `fbaux` + anti-confusion | ~1.75M | 0.6661 | 0.9317 | 11.92 | 2.50 | 0.0967 | 0.0089 | 0.0667 | 0.8720 | Current native classification mainline |
| DOA-Net mainline + reg | Native DOA-Net | `log_mag_L/R + IPD + ILD + sin/cos(IPD) + coherence` | Same as above | `72-class logits + angle + fbaux` | CE + angular regression + `fbaux` + anti-confusion | ~1.78M | 0.6640 | 0.9360 | 12.40 | 3.54 | 0.0892 | 0.0057 | 0.0679 | 0.8609 | Multi-task classification + regression |
| DOA-Net pure-reg + fbaux | Native DOA-Net | `log_mag_L/R + IPD + ILD + sin/cos(IPD) + coherence` | Same as above | `angle_vec + fbaux` | Cosine vector regression + `fbaux` | ~1.76M | 0.4457 | 0.8370 | 11.01 | 2.40 | 0.0816 | 0.0063 | 0.0578 | 0.8834 | Native backbone pure regression |
| SDEL-DOA-Reg | External baseline | `MBMS proxy + ILD + cos(IPD) + sin(IPD)` | 3-layer CNN + BiGRU + bidirectional multiplicative fusion + MLP | `angle_vec` | Vector regression | ~0.92M | 0.4112 | 0.8651 | 7.42 | 2.25 | 0.0428 | 0.0022 | 0.0316 | 0.9246 | Lowest MAE baseline |
| SDEL-DOA-Cls | External baseline | `MBMS proxy + ILD + cos(IPD) + sin(IPD)` | Same as above | `72-class logits` | CE | ~0.93M | 0.6910 | 0.9558 | 12.77 | 2.50 | 0.0704 | 0.0118 | 0.0620 | 0.8843 | Strong classification baseline |
| SDEL-DOA-Cls + fbaux | External baseline | `MBMS proxy + ILD + cos(IPD) + sin(IPD)` | Same as above | `72-class logits + fbaux` | CE + `fbaux` | ~0.93M | 0.7201 | 0.9727 | 9.02 | 2.50 | 0.0422 | 0.0053 | 0.0381 | 0.9221 | Strongest classification baseline overall |

## Notes

- `FB Err` = `front_back_halfplane_error_rate`
- `Opp Err` = `opposite_error_rate`
- `Large Err` = `large_error_rate`
- `Top-3` for regression models is derived from quantized nearest-bin predictions and should be treated as an auxiliary metric.
- `Params` are approximate rounded values for table readability.

## Recommended Use

- If the focus is the native model line, use:
  - `DOA-Net mainline cls`
  - `DOA-Net mainline + reg`
  - `DOA-Net pure-reg + fbaux`
- If the focus is external comparison, use:
  - `SDEL-DOA-Reg`
  - `SDEL-DOA-Cls`
  - `SDEL-DOA-Cls + fbaux`
