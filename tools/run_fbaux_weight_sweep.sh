#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/disk2/bywang/DOA-net"
RUNNER="$ROOT_DIR/tools/run_training_background.sh"

CONFIGS=(
  "configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux_w010.yaml"
  "configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux_w015.yaml"
  "configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux_w020.yaml"
  "configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux_only.yaml"
)

if [[ ! -x "$RUNNER" ]]; then
  echo "Runner not found or not executable: $RUNNER"
  exit 1
fi

cd "$ROOT_DIR"

echo "fbaux_only aux-weight sweep plan:"
echo "  w010 -> front_back_aux_weight=0.10"
echo "  w015 -> front_back_aux_weight=0.15"
echo "  w020 -> front_back_aux_weight=0.20"
echo "  fbaux_only -> front_back_aux_weight=0.30"
echo
echo "Recommended order: w010 -> w015 -> w020 -> fbaux_only(0.30)"
echo "Each run uses --train.num_workers 4 for server-side speed."
echo

for config in "${CONFIGS[@]}"; do
  echo "Launching: $config"
  "$RUNNER" "$config" --train.num_workers 4
done

echo
echo "All sweep jobs have been submitted."
