#!/usr/bin/env bash
set -euo pipefail

# Generalization diagnostics runner (pragmatic version)
# Modes:
#   speaker     : speaker overlap vs disjoint (same training model, two test sets)
#   noise       : clean-trained vs mixed-trained (same robust test set)
#   subject     : single-subject vs cross-subject (same unseen-subject test set)
#   all         : run all three

cd /disk2/bywang/DOA-net

MODE="${1:-all}"
SEEDS_CSV="${2:-42,43}"
DIAG_EPOCHS="${DIAG_EPOCHS:-20}"
PYTHON_BIN="${PYTHON_BIN:-/home/bywang/miniconda3/envs/doa/bin/python}"
BASE_CFG="${BASE_CFG:-configs/train_librispeech_subject003_cipic_reverb_demand50h_v5_bias_gating_attnpool_csl.yaml}"

OUT_ROOT="outputs/diagnostics"
mkdir -p "$OUT_ROOT"

# ---------- dataset roots (override via env) ----------
# 1) speaker overlap/disjoint
SPK_TRAIN_ROOT="${SPK_TRAIN_ROOT:-/disk2/bywang/DOA-net/data/diag_speaker_overlap_train_mixed}"
SPK_TEST_OVERLAP_ROOT="${SPK_TEST_OVERLAP_ROOT:-/disk2/bywang/DOA-net/data/diag_speaker_overlap_test_mixed}"
SPK_TEST_DISJOINT_ROOT="${SPK_TEST_DISJOINT_ROOT:-/disk2/bywang/DOA-net/data/diag_speaker_disjoint_test_mixed}"

# 2) clean-trained vs mixed-trained
CLEAN_TRAIN_ROOT="${CLEAN_TRAIN_ROOT:-/disk2/bywang/DOA-net/data/librispeech_cipic_subject003_50h_clean}"
MIXED_TRAIN_ROOT="${MIXED_TRAIN_ROOT:-/disk2/bywang/DOA-net/data/librispeech_cipic_subject003_reverb_demand50h_v2}"
ROBUST_TEST_ROOT="${ROBUST_TEST_ROOT:-/disk2/bywang/DOA-net/data/librispeech_cipic_subject003_reverb_demand50h_v2}"

# 3) single-subject vs cross-subject
SINGLE_SUBJECT_TRAIN_ROOT="${SINGLE_SUBJECT_TRAIN_ROOT:-/disk2/bywang/DOA-net/data/librispeech_cipic_subject003_reverb_demand50h_v2}"
CROSS_SUBJECT_TRAIN_ROOT="${CROSS_SUBJECT_TRAIN_ROOT:-/disk2/bywang/DOA-net/data/librispeech_cipic_multisubject/train_subjects}"
CROSS_SUBJECT_TEST_ROOT="${CROSS_SUBJECT_TEST_ROOT:-/disk2/bywang/DOA-net/data/librispeech_cipic_multisubject/test_subjects_unseen}"

METRICS="accuracy,top_k_accuracy,macro_precision,macro_recall,macro_f1,mean_angular_error,median_angular_error,error_lt_5,error_lt_10"

ensure_root_exists() {
  local p="$1"
  if [[ ! -d "$p" ]]; then
    echo "[ERROR] Missing dataset root: $p"
    exit 1
  fi
}

train_model() {
  local tag="$1"
  local train_root="$2"
  local seed="$3"

  local save_dir="${OUT_ROOT}/checkpoints_${tag}_seed${seed}"
  local log_dir="${OUT_ROOT}/logs_${tag}_seed${seed}"

  mkdir -p "$save_dir" "$log_dir"

  "$PYTHON_BIN" train.py \
    --config "$BASE_CFG" \
    --dataset.root_dir "$train_root" \
    --dataset.split_seed "$seed" \
    --train.epochs "$DIAG_EPOCHS" \
    --output.save_dir "$save_dir" \
    --output.log_dir "$log_dir" \
    > /dev/null

  echo "$save_dir/best.pth"
}

eval_model() {
  local tag="$1"
  local ckpt="$2"
  local test_root="$3"
  local seed="$4"

  local eval_dir="${OUT_ROOT}/logs_${tag}_seed${seed}_eval"
  mkdir -p "$eval_dir"

  "$PYTHON_BIN" evaluate.py \
    --checkpoint "$ckpt" \
    --config "$BASE_CFG" \
    --dataset.root_dir "$test_root" \
    --dataset.train_ratio 0.0 \
    --dataset.val_ratio 0.0 \
    --dataset.test_ratio 1.0 \
    --dataset.split_seed "$seed" \
    --output.log_dir "$eval_dir" \
    > /dev/null

  echo "$eval_dir/train.log"
}

extract_metrics_to_csv() {
  local log_path="$1"
  local out_csv="$2"
  local seed="$3"
  local condition="$4"

  "$PYTHON_BIN" - "$log_path" "$out_csv" "$seed" "$condition" << 'PY'
import re
import sys
from pathlib import Path

log_path, out_csv, seed, condition = sys.argv[1:5]
keys = [
    'accuracy', 'top_k_accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
    'mean_angular_error', 'median_angular_error', 'error_lt_5', 'error_lt_10'
]
pat = re.compile(r"\b([a-zA-Z0-9_]+):\s+([0-9]*\.?[0-9]+)")
vals = {}
for line in Path(log_path).read_text(encoding='utf-8').splitlines():
    m = pat.search(line)
    if not m:
        continue
    k, v = m.group(1), m.group(2)
    if k in keys:
        vals[k] = float(v)

missing = [k for k in keys if k not in vals]
if missing:
    raise SystemExit(f"Missing metrics in {log_path}: {missing}")

p = Path(out_csv)
if not p.exists():
    p.write_text(
        "seed,condition,accuracy,top_k_accuracy,macro_precision,macro_recall,macro_f1,"
        "mean_angular_error,median_angular_error,error_lt_5,error_lt_10\n",
        encoding='utf-8',
    )
with p.open('a', encoding='utf-8') as f:
    f.write(
        f"{seed},{condition},{vals['accuracy']:.6f},{vals['top_k_accuracy']:.6f},"
        f"{vals['macro_precision']:.6f},{vals['macro_recall']:.6f},{vals['macro_f1']:.6f},"
        f"{vals['mean_angular_error']:.6f},{vals['median_angular_error']:.6f},"
        f"{vals['error_lt_5']:.6f},{vals['error_lt_10']:.6f}\n"
    )
PY
}

aggregate_csv_to_md() {
  local csv_path="$1"
  local md_path="$2"
  local title="$3"

  "$PYTHON_BIN" - "$csv_path" "$md_path" "$title" << 'PY'
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

csv_path, md_path, title = sys.argv[1:4]
rows = list(csv.DictReader(Path(csv_path).open('r', encoding='utf-8')))
if not rows:
    raise SystemExit(f"No rows in {csv_path}")

metrics = [
    'accuracy', 'top_k_accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
    'mean_angular_error', 'median_angular_error', 'error_lt_5', 'error_lt_10'
]

grouped = defaultdict(list)
for r in rows:
    grouped[r['condition']].append(r)

with Path(md_path).open('w', encoding='utf-8') as f:
    f.write(f"# {title}\n\n")
    f.write("## Per-seed\n\n")
    f.write("| seed | condition | acc | top3 | macro_p | macro_r | macro_f1 | MAE | median | err<5 | err<10 |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        f.write(
            f"| {r['seed']} | {r['condition']} | {float(r['accuracy']):.4f} | {float(r['top_k_accuracy']):.4f} | "
            f"{float(r['macro_precision']):.4f} | {float(r['macro_recall']):.4f} | {float(r['macro_f1']):.4f} | "
            f"{float(r['mean_angular_error']):.4f} | {float(r['median_angular_error']):.4f} | "
            f"{float(r['error_lt_5']):.4f} | {float(r['error_lt_10']):.4f} |\n"
        )

    f.write("\n## Mean ± Std\n\n")
    f.write("| condition | acc | top3 | macro_p | macro_r | macro_f1 | MAE | median | err<5 | err<10 |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for cond, cond_rows in sorted(grouped.items()):
        vals = {m: [float(r[m]) for r in cond_rows] for m in metrics}
        mean = {m: statistics.mean(v) for m, v in vals.items()}
        std = {m: (statistics.stdev(v) if len(v) > 1 else 0.0) for m, v in vals.items()}
        f.write(
            f"| {cond} | {mean['accuracy']:.4f} ± {std['accuracy']:.4f} | {mean['top_k_accuracy']:.4f} ± {std['top_k_accuracy']:.4f} | "
            f"{mean['macro_precision']:.4f} ± {std['macro_precision']:.4f} | {mean['macro_recall']:.4f} ± {std['macro_recall']:.4f} | "
            f"{mean['macro_f1']:.4f} ± {std['macro_f1']:.4f} | {mean['mean_angular_error']:.4f} ± {std['mean_angular_error']:.4f} | "
            f"{mean['median_angular_error']:.4f} ± {std['median_angular_error']:.4f} | {mean['error_lt_5']:.4f} ± {std['error_lt_5']:.4f} | "
            f"{mean['error_lt_10']:.4f} ± {std['error_lt_10']:.4f} |\n"
        )
PY
}

run_speaker_diag() {
  ensure_root_exists "$SPK_TRAIN_ROOT"
  ensure_root_exists "$SPK_TEST_OVERLAP_ROOT"
  ensure_root_exists "$SPK_TEST_DISJOINT_ROOT"

  local csv_path="${OUT_ROOT}/speaker_overlap_vs_disjoint.csv"
  rm -f "$csv_path"

  IFS=',' read -r -a seeds <<< "$SEEDS_CSV"
  for seed in "${seeds[@]}"; do
    seed="$(echo "$seed" | xargs)"
    echo "[speaker] seed=$seed train overlap model"
    ckpt="$(train_model "diag_speaker_train" "$SPK_TRAIN_ROOT" "$seed")"

    log_overlap="$(eval_model "diag_speaker_overlap_test" "$ckpt" "$SPK_TEST_OVERLAP_ROOT" "$seed")"
    extract_metrics_to_csv "$log_overlap" "$csv_path" "$seed" "overlap_test"

    log_disjoint="$(eval_model "diag_speaker_disjoint_test" "$ckpt" "$SPK_TEST_DISJOINT_ROOT" "$seed")"
    extract_metrics_to_csv "$log_disjoint" "$csv_path" "$seed" "disjoint_test"
  done

  aggregate_csv_to_md "$csv_path" "${OUT_ROOT}/speaker_overlap_vs_disjoint.md" "Speaker Overlap vs Disjoint"
}

run_noise_diag() {
  ensure_root_exists "$CLEAN_TRAIN_ROOT"
  ensure_root_exists "$MIXED_TRAIN_ROOT"
  ensure_root_exists "$ROBUST_TEST_ROOT"

  local csv_path="${OUT_ROOT}/clean_trained_vs_mixed_trained.csv"
  rm -f "$csv_path"

  IFS=',' read -r -a seeds <<< "$SEEDS_CSV"
  for seed in "${seeds[@]}"; do
    seed="$(echo "$seed" | xargs)"

    echo "[noise] seed=$seed train clean"
    ckpt_clean="$(train_model "diag_clean_trained" "$CLEAN_TRAIN_ROOT" "$seed")"
    log_clean="$(eval_model "diag_clean_on_robust_test" "$ckpt_clean" "$ROBUST_TEST_ROOT" "$seed")"
    extract_metrics_to_csv "$log_clean" "$csv_path" "$seed" "clean_trained_on_robust_test"

    echo "[noise] seed=$seed train mixed"
    ckpt_mixed="$(train_model "diag_mixed_trained" "$MIXED_TRAIN_ROOT" "$seed")"
    log_mixed="$(eval_model "diag_mixed_on_robust_test" "$ckpt_mixed" "$ROBUST_TEST_ROOT" "$seed")"
    extract_metrics_to_csv "$log_mixed" "$csv_path" "$seed" "mixed_trained_on_robust_test"
  done

  aggregate_csv_to_md "$csv_path" "${OUT_ROOT}/clean_trained_vs_mixed_trained.md" "Clean-trained vs Mixed-trained"
}

run_subject_diag() {
  ensure_root_exists "$SINGLE_SUBJECT_TRAIN_ROOT"
  ensure_root_exists "$CROSS_SUBJECT_TRAIN_ROOT"
  ensure_root_exists "$CROSS_SUBJECT_TEST_ROOT"

  local csv_path="${OUT_ROOT}/single_subject_vs_cross_subject.csv"
  rm -f "$csv_path"

  IFS=',' read -r -a seeds <<< "$SEEDS_CSV"
  for seed in "${seeds[@]}"; do
    seed="$(echo "$seed" | xargs)"

    echo "[subject] seed=$seed train single-subject"
    ckpt_single="$(train_model "diag_single_subject" "$SINGLE_SUBJECT_TRAIN_ROOT" "$seed")"
    log_single="$(eval_model "diag_single_on_unseen_subject" "$ckpt_single" "$CROSS_SUBJECT_TEST_ROOT" "$seed")"
    extract_metrics_to_csv "$log_single" "$csv_path" "$seed" "single_subject_trained_on_unseen_subject_test"

    echo "[subject] seed=$seed train cross-subject"
    ckpt_cross="$(train_model "diag_cross_subject" "$CROSS_SUBJECT_TRAIN_ROOT" "$seed")"
    log_cross="$(eval_model "diag_cross_on_unseen_subject" "$ckpt_cross" "$CROSS_SUBJECT_TEST_ROOT" "$seed")"
    extract_metrics_to_csv "$log_cross" "$csv_path" "$seed" "cross_subject_trained_on_unseen_subject_test"
  done

  aggregate_csv_to_md "$csv_path" "${OUT_ROOT}/single_subject_vs_cross_subject.md" "Single-subject vs Cross-subject"
}

case "$MODE" in
  speaker)
    run_speaker_diag
    ;;
  noise)
    run_noise_diag
    ;;
  subject)
    run_subject_diag
    ;;
  all)
    run_speaker_diag
    run_noise_diag
    run_subject_diag
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: bash tools/diagnostics/run_generalization_diagnostics.sh [speaker|noise|subject|all] [seeds_csv]"
    exit 1
    ;;
esac

echo "Diagnostics completed. Reports are under: ${OUT_ROOT}"
