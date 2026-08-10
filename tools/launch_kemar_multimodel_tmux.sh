#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/disk2/bywang/DOA-net"
RUNNER="$ROOT_DIR/tools/run_training_tmux.sh"

TRAIN_ROOT="/disk2/bywang/DOA-net/data/kemar_sofamyroom/train_20h_minimal/train"
VAL_ROOT="/disk2/bywang/DOA-net/outputs/kemar_sofamyroom_dataset_val_4h/val"
TEST_ROOT="/disk2/bywang/DOA-net/outputs/kemar_sofamyroom_dataset_test_grid/test"

if [[ ! -x "$RUNNER" ]]; then
  echo "Runner not found or not executable: $RUNNER"
  exit 1
fi

COMMON_ARGS=(
  --dataset.train_root "$TRAIN_ROOT"
  --dataset.val_root "$VAL_ROOT"
  --dataset.test_root "$TEST_ROOT"
  --train.num_workers 4
)

launch() {
  local session_name="$1"
  local gpu_id="$2"
  local config_path="$3"
  shift 3

  "$RUNNER" "$session_name" "$config_path" \
    --train.device "cuda:${gpu_id}" \
    "${COMMON_ARGS[@]}" \
    "$@"
}

launch \
  kemar_v7dual_g0 \
  0 \
  configs/train_librispeech_multisubject_static_hybridbrir_gate2_50h_v1_v7_dualcue.yaml

launch \
  kemar_v7lite_g1 \
  1 \
  configs/train_librispeech_multisubject_static_hybridbrir_gate2_50h_v1_v7_litecueenc_concat_all_cf80_gru80.yaml

launch \
  kemar_biear_g2 \
  2 \
  configs/train_librispeech_multisubject_static_hybridbrir_gate2_50h_v1_biear_doa_cls.yaml

launch \
  kemar_bil_g3 \
  3 \
  configs/train_librispeech_multisubject_static_hybridbrir_gate2_50h_v1_bilstyle_gccphat_crn72.yaml

echo
echo "Launched 4 KEMAR training sessions."
echo "Check sessions: tmux ls"
echo "Attach example: tmux attach -t kemar_v7dual_g0"
