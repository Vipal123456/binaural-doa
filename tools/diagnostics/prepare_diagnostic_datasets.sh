#!/usr/bin/env bash
set -euo pipefail

# Prepare dataset roots used by the three diagnostics.
# Modes:
#   speaker_data   : overlap/disjoint speaker-based roots (with mixed corruption)
#   multisubject   : cross-subject roots (with mixed corruption)
#   all            : both

cd /disk2/bywang/DOA-net

MODE="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-/home/bywang/miniconda3/envs/doa/bin/python}"

LIBRISPEECH_ROOT="${LIBRISPEECH_ROOT:-/disk2/bywang/data/LibriSpeech/train-clean-100}"
HRTF_SOFA="${HRTF_SOFA:-/disk2/bywang/data/HRTF/subject_003.sofa}"
DEMAND_ROOT="${DEMAND_ROOT:-/disk2/bywang/data/demand}"

# Speaker split roots
SPK_SPLIT_ROOT="${SPK_SPLIT_ROOT:-/disk2/bywang/DOA-net/data/librispeech_speaker_splits}"

SPK_OVERLAP_TRAIN_CLEAN="${SPK_OVERLAP_TRAIN_CLEAN:-/disk2/bywang/DOA-net/data/diag_speaker_overlap_train_clean}"
SPK_OVERLAP_TEST_CLEAN="${SPK_OVERLAP_TEST_CLEAN:-/disk2/bywang/DOA-net/data/diag_speaker_overlap_test_clean}"
SPK_DISJOINT_TEST_CLEAN="${SPK_DISJOINT_TEST_CLEAN:-/disk2/bywang/DOA-net/data/diag_speaker_disjoint_test_clean}"

SPK_OVERLAP_TRAIN_MIXED="${SPK_OVERLAP_TRAIN_MIXED:-/disk2/bywang/DOA-net/data/diag_speaker_overlap_train_mixed}"
SPK_OVERLAP_TEST_MIXED="${SPK_OVERLAP_TEST_MIXED:-/disk2/bywang/DOA-net/data/diag_speaker_overlap_test_mixed}"
SPK_DISJOINT_TEST_MIXED="${SPK_DISJOINT_TEST_MIXED:-/disk2/bywang/DOA-net/data/diag_speaker_disjoint_test_mixed}"

# synthesis scale (diagnosis-friendly)
NUM_TRAIN_REC="${NUM_TRAIN_REC:-3000}"
NUM_TEST_REC="${NUM_TEST_REC:-1500}"

prepare_mixed() {
  local clean_root="$1"
  local mixed_root="$2"

  "$PYTHON_BIN" prepare_demand_mixed_data.py \
    --clean_root "$clean_root" \
    --demand_root "$DEMAND_ROOT" \
    --output_root "$mixed_root" \
    --overwrite
}

prepare_speaker_data() {
  echo "[speaker_data] prepare speaker split manifests"
  "$PYTHON_BIN" tools/diagnostics/prepare_librispeech_speaker_splits.py \
    --librispeech_root "$LIBRISPEECH_ROOT" \
    --output_root "$SPK_SPLIT_ROOT" \
    --train_speakers 180 \
    --overlap_eval_speakers 30 \
    --disjoint_eval_speakers 30 \
    --seed 42

  echo "[speaker_data] synthesize overlap-train clean"
  "$PYTHON_BIN" synthesize_librispeech_cipic.py \
    --librispeech_root "$SPK_SPLIT_ROOT/train_speakers" \
    --sofa_path "$HRTF_SOFA" \
    --output_root "$SPK_OVERLAP_TRAIN_CLEAN" \
    --num_recordings "$NUM_TRAIN_REC" \
    --sample_rate 16000 \
    --duration_sec 10.0 \
    --seed 42

  echo "[speaker_data] synthesize overlap-test clean"
  "$PYTHON_BIN" synthesize_librispeech_cipic.py \
    --librispeech_root "$SPK_SPLIT_ROOT/test_overlap_speakers" \
    --sofa_path "$HRTF_SOFA" \
    --output_root "$SPK_OVERLAP_TEST_CLEAN" \
    --num_recordings "$NUM_TEST_REC" \
    --sample_rate 16000 \
    --duration_sec 10.0 \
    --seed 43

  echo "[speaker_data] synthesize disjoint-test clean"
  "$PYTHON_BIN" synthesize_librispeech_cipic.py \
    --librispeech_root "$SPK_SPLIT_ROOT/test_disjoint_speakers" \
    --sofa_path "$HRTF_SOFA" \
    --output_root "$SPK_DISJOINT_TEST_CLEAN" \
    --num_recordings "$NUM_TEST_REC" \
    --sample_rate 16000 \
    --duration_sec 10.0 \
    --seed 44

  echo "[speaker_data] prepare mixed roots"
  prepare_mixed "$SPK_OVERLAP_TRAIN_CLEAN" "$SPK_OVERLAP_TRAIN_MIXED"
  prepare_mixed "$SPK_OVERLAP_TEST_CLEAN" "$SPK_OVERLAP_TEST_MIXED"
  prepare_mixed "$SPK_DISJOINT_TEST_CLEAN" "$SPK_DISJOINT_TEST_MIXED"
}

prepare_multisubject_data() {
  echo "[multisubject] prepare clean subject-disjoint roots"
  "$PYTHON_BIN" prepare_multisubject_data.py \
    --repo_root /disk2/bywang/DOA-net \
    --librispeech_root "$LIBRISPEECH_ROOT" \
    --hrtf_root /disk2/bywang/data/HRTF \
    --data_root /disk2/bywang/DOA-net/data \
    --python_exec "$PYTHON_BIN" \
    --total_subjects 12 \
    --train_subjects 8 \
    --val_subjects 2 \
    --test_subjects 2 \
    --seed 42 \
    --force_include 003

  echo "[multisubject] prepare mixed train/unseen-test roots"
  prepare_mixed \
    /disk2/bywang/DOA-net/data/librispeech_cipic_multisubject/train_subjects \
    /disk2/bywang/DOA-net/data/librispeech_cipic_multisubject/train_subjects_mixed

  prepare_mixed \
    /disk2/bywang/DOA-net/data/librispeech_cipic_multisubject/test_subjects_unseen \
    /disk2/bywang/DOA-net/data/librispeech_cipic_multisubject/test_subjects_unseen_mixed
}

case "$MODE" in
  speaker_data)
    prepare_speaker_data
    ;;
  multisubject)
    prepare_multisubject_data
    ;;
  all)
    prepare_speaker_data
    prepare_multisubject_data
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: bash tools/diagnostics/prepare_diagnostic_datasets.sh [speaker_data|multisubject|all]"
    exit 1
    ;;
esac

echo "Dataset preparation done."
