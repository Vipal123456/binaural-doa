#!/usr/bin/env bash
set -euo pipefail

cd /disk2/bywang/DOA-net

label="${1:-a7}"
mode="${2:-smoke}"
seeds_csv="${3:-42,43}"
smoke_epochs="${SMOKE_EPOCHS:-3}"

base_cfg="configs/train_librispeech_subject003_cipic_reverb_demand50h_v5_bias_gating_attnpool_csl.yaml"
save_dir="outputs/checkpoints_librispeech_subject003_cipic_reverb_demand50h_ablation_${label}"
log_dir="outputs/logs_librispeech_subject003_cipic_reverb_demand50h_ablation_${label}"

case "$label" in
  a0)
    extra_args=(
      --model.use_attention_bias false
      --model.use_independent_gating false
      --model.use_residual_gating false
      --model.use_attention_pooling false
      --train.circular_soft_label_weight 0.0
      --train.anti_confusion_weight 1.0
    )
    ;;
  a1)
    extra_args=(
      --model.use_attention_bias true
      --model.use_independent_gating false
      --model.use_residual_gating false
      --model.use_attention_pooling false
      --train.circular_soft_label_weight 0.0
      --train.anti_confusion_weight 1.0
    )
    ;;
  a2)
    extra_args=(
      --model.use_attention_bias false
      --model.use_independent_gating true
      --model.use_residual_gating true
      --model.use_attention_pooling false
      --train.circular_soft_label_weight 0.0
      --train.anti_confusion_weight 1.0
    )
    ;;
  a3)
    extra_args=(
      --model.use_attention_bias false
      --model.use_independent_gating false
      --model.use_residual_gating false
      --model.use_attention_pooling true
      --train.circular_soft_label_weight 0.0
      --train.anti_confusion_weight 1.0
    )
    ;;
  a4)
    extra_args=(
      --model.use_attention_bias false
      --model.use_independent_gating false
      --model.use_residual_gating false
      --model.use_attention_pooling false
      --train.circular_soft_label_weight 0.2
      --train.circular_kappa 4.0
      --train.anti_confusion_weight 1.0
    )
    ;;
  a5)
    extra_args=(
      --model.use_attention_bias true
      --model.use_independent_gating true
      --model.use_residual_gating true
      --model.use_attention_pooling false
      --train.circular_soft_label_weight 0.0
      --train.anti_confusion_weight 1.0
    )
    ;;
  a6)
    extra_args=(
      --model.use_attention_bias true
      --model.use_independent_gating true
      --model.use_residual_gating true
      --model.use_attention_pooling true
      --train.circular_soft_label_weight 0.0
      --train.anti_confusion_weight 1.0
    )
    ;;
  a7)
    extra_args=(
      --model.use_attention_bias true
      --model.use_independent_gating true
      --model.use_residual_gating true
      --model.use_attention_pooling true
      --train.circular_soft_label_weight 0.2
      --train.circular_kappa 4.0
      --train.anti_confusion_weight 1.0
    )
    ;;
  *)
    echo "Unknown ablation label: $label"
    exit 1
    ;;
esac

mkdir -p "$save_dir" "$log_dir"

if [[ "$mode" == "smoke" ]]; then
  if [[ "$smoke_epochs" -lt 3 || "$smoke_epochs" -gt 5 ]]; then
    echo "SMOKE_EPOCHS must be in [3, 5], got: $smoke_epochs"
    exit 1
  fi

  IFS=',' read -r -a seeds <<< "$seeds_csv"
  if [[ "${#seeds[@]}" -lt 2 ]]; then
    echo "Need at least 2 seeds for smoke runs, got: ${#seeds[@]}"
    exit 1
  fi

  summary_csv="${log_dir}/smoke_summary.csv"
  summary_md="${log_dir}/smoke_summary.md"
  : > "$summary_csv"
  echo "seed,accuracy,top_k_accuracy,macro_precision,macro_recall,macro_f1,mean_angular_error,median_angular_error,error_lt_5,error_lt_10" >> "$summary_csv"

  for seed in "${seeds[@]}"; do
    seed="$(echo "$seed" | xargs)"
    run_save_dir="${save_dir}_seed${seed}"
    run_log_dir="${log_dir}_seed${seed}"
    run_eval_dir="${run_log_dir}_test_smoke"

    mkdir -p "$run_save_dir" "$run_log_dir" "$run_eval_dir"

    echo "[SMOKE] label=$label seed=$seed epochs=$smoke_epochs"

    /home/bywang/miniconda3/envs/doa/bin/python train.py \
      --config "$base_cfg" \
      --output.save_dir "$run_save_dir" \
      --output.log_dir "$run_log_dir" \
      --train.epochs "$smoke_epochs" \
      --train.batch_size 16 \
      --train.num_workers 2 \
      --dataset.split_seed "$seed" \
      "${extra_args[@]}"

    /home/bywang/miniconda3/envs/doa/bin/python evaluate.py \
      --checkpoint "$run_save_dir/best.pth" \
      --config "$base_cfg" \
      --output.log_dir "$run_eval_dir" \
      --dataset.split_seed "$seed" \
      "${extra_args[@]}"

    eval_log="$run_eval_dir/train.log"
    /home/bywang/miniconda3/envs/doa/bin/python - "$seed" "$eval_log" "$summary_csv" << 'PY'
import re
import sys

seed, log_path, out_csv = sys.argv[1:4]
keys = [
    "accuracy",
    "top_k_accuracy",
  "macro_precision",
  "macro_recall",
  "macro_f1",
    "mean_angular_error",
    "median_angular_error",
    "error_lt_5",
    "error_lt_10",
]
pat = re.compile(r"\b([a-zA-Z0-9_]+):\s+([0-9]*\.?[0-9]+)")
vals = {}
with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        m = pat.search(line)
        if not m:
            continue
        k, v = m.group(1), m.group(2)
        if k in keys:
            vals[k] = float(v)

missing = [k for k in keys if k not in vals]
if missing:
    raise SystemExit(f"Missing keys in {log_path}: {missing}")

with open(out_csv, "a", encoding="utf-8") as f:
    f.write(
      f"{seed},{vals['accuracy']:.6f},{vals['top_k_accuracy']:.6f},"
      f"{vals['macro_precision']:.6f},{vals['macro_recall']:.6f},{vals['macro_f1']:.6f},"
        f"{vals['mean_angular_error']:.6f},{vals['median_angular_error']:.6f},"
        f"{vals['error_lt_5']:.6f},{vals['error_lt_10']:.6f}\n"
    )
PY
  done

  /home/bywang/miniconda3/envs/doa/bin/python - "$summary_csv" "$summary_md" "$label" "$smoke_epochs" "$seeds_csv" << 'PY'
import csv
import math
import statistics
import sys

csv_path, md_path, label, smoke_epochs, seeds_csv = sys.argv[1:]
rows = []
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

metrics = [
    "accuracy",
    "top_k_accuracy",
  "macro_precision",
  "macro_recall",
  "macro_f1",
    "mean_angular_error",
    "median_angular_error",
    "error_lt_5",
    "error_lt_10",
]

means = {}
vars_ = {}
for k in metrics:
    values = [float(r[k]) for r in rows]
    means[k] = statistics.mean(values)
    vars_[k] = statistics.variance(values) if len(values) > 1 else 0.0

with open(md_path, "w", encoding="utf-8") as f:
    f.write(f"# Ablation {label} Smoke Summary\\n\\n")
    f.write(f"- epochs: {smoke_epochs}\\n")
    f.write(f"- seeds: {seeds_csv}\\n\\n")
    f.write("## Per-seed\\n\\n")
    f.write("| seed | accuracy | top_k_accuracy | macro_precision | macro_recall | macro_f1 | mean_angular_error | median_angular_error | error_lt_5 | error_lt_10 |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
      f.write(
        f"| {r['seed']} | {float(r['accuracy']):.4f} | {float(r['top_k_accuracy']):.4f} | "
        f"{float(r['macro_precision']):.4f} | {float(r['macro_recall']):.4f} | {float(r['macro_f1']):.4f} | "
          f"{float(r['mean_angular_error']):.4f} | {float(r['median_angular_error']):.4f} | "
          f"{float(r['error_lt_5']):.4f} | {float(r['error_lt_10']):.4f} |\\n"
      )

    f.write("\\n## Mean and Variance\\n\\n")
    f.write("| metric | mean | variance |\\n")
    f.write("|---|---:|---:|\\n")
    for k in metrics:
        f.write(f"| {k} | {means[k]:.6f} | {vars_[k]:.6f} |\\n")

print(f"Wrote summary: {md_path}")
PY

  cp "$summary_md" "$log_dir/smoke_report.md"
  echo "Smoke summary: $summary_md"

elif [[ "$mode" == "full" ]]; then
  nohup /home/bywang/miniconda3/envs/doa/bin/python -u train.py \
    --config "$base_cfg" \
    --output.save_dir "$save_dir" \
    --output.log_dir "$log_dir" \
    "${extra_args[@]}" \
    > "$log_dir/train_full.log" 2>&1 &
  echo "Started full training for $label in background."
else
  echo "Unknown mode: $mode"
  exit 1
fi