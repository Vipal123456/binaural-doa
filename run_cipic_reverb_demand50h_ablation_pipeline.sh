#!/usr/bin/env bash
set -euo pipefail

cd /disk2/bywang/DOA-net

label="${1:-a7}"
mode="${2:-smoke}"

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
  /home/bywang/miniconda3/envs/doa/bin/python train.py \
    --config "$base_cfg" \
    --output.save_dir "$save_dir" \
    --output.log_dir "$log_dir" \
    --train.epochs 1 \
    --train.batch_size 16 \
    --train.num_workers 2 \
    "${extra_args[@]}"

  /home/bywang/miniconda3/envs/doa/bin/python evaluate.py \
    --checkpoint "$save_dir/best.pth" \
    --config "$base_cfg" \
    --output.log_dir "${log_dir}_test_smoke" \
    "${extra_args[@]}"

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