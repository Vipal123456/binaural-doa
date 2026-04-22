# Generalization Diagnostics Runbook

This runbook implements three focused diagnostics:

1. speaker overlap vs speaker disjoint
2. clean-trained vs mixed-trained
3. single-subject vs cross-subject

All diagnostics use the same model family and report:

- accuracy, top_k_accuracy
- macro_precision, macro_recall, macro_f1
- mean_angular_error, median_angular_error
- error_lt_5, error_lt_10

## 0) Quick prerequisites

- Python env: /home/bywang/miniconda3/envs/doa
- Main config baseline: configs/train_librispeech_subject003_cipic_reverb_demand50h_v5_bias_gating_attnpool_csl.yaml
- Required scripts:
  - tools/diagnostics/prepare_librispeech_speaker_splits.py
  - tools/diagnostics/prepare_diagnostic_datasets.sh
  - tools/diagnostics/run_generalization_diagnostics.sh

## 1) Prepare datasets for diagnostics

### 1.1 Speaker overlap/disjoint roots (with mixed corruption)

```bash
bash tools/diagnostics/prepare_diagnostic_datasets.sh speaker_data
```

Generated roots (defaults):

- /disk2/bywang/DOA-net/data/diag_speaker_overlap_train_mixed
- /disk2/bywang/DOA-net/data/diag_speaker_overlap_test_mixed
- /disk2/bywang/DOA-net/data/diag_speaker_disjoint_test_mixed

### 1.2 Cross-subject roots (with mixed corruption)

```bash
bash tools/diagnostics/prepare_diagnostic_datasets.sh multisubject
```

Generated roots (defaults):

- /disk2/bywang/DOA-net/data/librispeech_cipic_multisubject/train_subjects_mixed
- /disk2/bywang/DOA-net/data/librispeech_cipic_multisubject/test_subjects_unseen_mixed

## 2) Run diagnostics

Set seeds and training length for diagnostics:

```bash
export DIAG_EPOCHS=20
```

### 2.1 speaker overlap vs disjoint

```bash
export SPK_TRAIN_ROOT=/disk2/bywang/DOA-net/data/diag_speaker_overlap_train_mixed
export SPK_TEST_OVERLAP_ROOT=/disk2/bywang/DOA-net/data/diag_speaker_overlap_test_mixed
export SPK_TEST_DISJOINT_ROOT=/disk2/bywang/DOA-net/data/diag_speaker_disjoint_test_mixed
bash tools/diagnostics/run_generalization_diagnostics.sh speaker 42,43
```

Output:

- outputs/diagnostics/speaker_overlap_vs_disjoint.csv
- outputs/diagnostics/speaker_overlap_vs_disjoint.md

### 2.2 clean-trained vs mixed-trained

```bash
export CLEAN_TRAIN_ROOT=/disk2/bywang/DOA-net/data/librispeech_cipic_subject003_50h_clean
export MIXED_TRAIN_ROOT=/disk2/bywang/DOA-net/data/librispeech_cipic_subject003_reverb_demand50h_v2
export ROBUST_TEST_ROOT=/disk2/bywang/DOA-net/data/librispeech_cipic_subject003_reverb_demand50h_v2
bash tools/diagnostics/run_generalization_diagnostics.sh noise 42,43
```

Output:

- outputs/diagnostics/clean_trained_vs_mixed_trained.csv
- outputs/diagnostics/clean_trained_vs_mixed_trained.md

### 2.3 single-subject vs cross-subject

```bash
export SINGLE_SUBJECT_TRAIN_ROOT=/disk2/bywang/DOA-net/data/librispeech_cipic_subject003_reverb_demand50h_v2
export CROSS_SUBJECT_TRAIN_ROOT=/disk2/bywang/DOA-net/data/librispeech_cipic_multisubject/train_subjects_mixed
export CROSS_SUBJECT_TEST_ROOT=/disk2/bywang/DOA-net/data/librispeech_cipic_multisubject/test_subjects_unseen_mixed
bash tools/diagnostics/run_generalization_diagnostics.sh subject 42,43
```

Output:

- outputs/diagnostics/single_subject_vs_cross_subject.csv
- outputs/diagnostics/single_subject_vs_cross_subject.md

## 3) Decision rule (what to optimize first)

- If disjoint_test MAE - overlap_test MAE is largest: prioritize speaker robustness.
- If clean_trained_on_robust_test MAE - mixed_trained_on_robust_test MAE is largest: prioritize noise/reverb robustness.
- If single_subject_trained_on_unseen_subject_test MAE - cross_subject_trained_on_unseen_subject_test MAE is largest: prioritize HRTF generalization.

Use MAE and error_lt_5 as primary decision metrics; macro_recall is the secondary classifier-level check.
