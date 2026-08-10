#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/disk2/bywang/DOA-net"
PYTHON_BIN="${PYTHON_BIN:-/home/bywang/miniconda3/envs/doa/bin/python}"
VAL_ROOT="/disk2/bywang/DOA-net/data/kemar_sofamyroom/val_4h_diffusefg_officialsplit/val"
TEST_ROOT="/disk2/bywang/DOA-net/data/kemar_sofamyroom_diffusefg_static_v1_test_officialsplit/test"
QUEUE="${1:-gpu4}"

cd "${ROOT_DIR}"

run_train() {
  local name="$1"
  local config="$2"
  local device="$3"
  local save_dir="$4"
  local log_dir="$5"

  mkdir -p "${log_dir}" "${save_dir}" outputs/logs_officialsplit_baseline_train_queue
  echo "[train] ${name} device=${device}"
  "${PYTHON_BIN}" -u train.py \
    --config "${config}" \
    --dataset.val_root "${VAL_ROOT}" \
    --dataset.test_root "${TEST_ROOT}" \
    --train.device "${device}" \
    --output.save_dir "${save_dir}" \
    --output.log_dir "${log_dir}" \
    > "outputs/logs_officialsplit_baseline_train_queue/${name}.launch.log" 2>&1
}

wait_session_done() {
  local session="$1"
  echo "[wait] ${session}"
  while tmux has-session -t "${session}" 2>/dev/null; do
    sleep 120
  done
}

case "${QUEUE}" in
  gpu4)
    wait_session_done train_sdel_official_seed44_g4
    run_train dprtf_official_seed44_g4 configs/train_kemar_dprtf_doa_cls_diffusefg_seed44_g4.yaml cuda:4 \
      outputs/checkpoints_kemar_dprtf_doa_cls_diffusefg_officialsplit_seed44_g4 \
      outputs/logs_kemar_dprtf_doa_cls_diffusefg_officialsplit_seed44_g4
    run_train bil_official_seed42_g4 configs/train_kemar_bilstyle_gccphat_crn72_diffusefg_g7.yaml cuda:4 \
      outputs/checkpoints_kemar_bilstyle_gccphat_crn72_diffusefg_officialsplit_seed42_g4 \
      outputs/logs_kemar_bilstyle_gccphat_crn72_diffusefg_officialsplit_seed42_g4
    run_train fnssl_official_seed43_g4 configs/train_kemar_fnssl_diffusefg_seed43_g4.yaml cuda:4 \
      outputs/checkpoints_kemar_fnssl_diffusefg_officialsplit_seed43_g4 \
      outputs/logs_kemar_fnssl_diffusefg_officialsplit_seed43_g4
    ;;
  gpu5)
    wait_session_done train_sdel_official_seed43_g5_wait
    run_train dprtf_official_seed42_g5 configs/train_kemar_dprtf_doa_cls_diffusefg_g5.yaml cuda:5 \
      outputs/checkpoints_kemar_dprtf_doa_cls_diffusefg_officialsplit_seed42_g5 \
      outputs/logs_kemar_dprtf_doa_cls_diffusefg_officialsplit_seed42_g5
    run_train bil_official_seed44_g5 configs/train_kemar_bilstyle_gccphat_crn72_diffusefg_seed44_g5.yaml cuda:5 \
      outputs/checkpoints_kemar_bilstyle_gccphat_crn72_diffusefg_officialsplit_seed44_g5 \
      outputs/logs_kemar_bilstyle_gccphat_crn72_diffusefg_officialsplit_seed44_g5
    run_train fnssl_official_seed44_g5 configs/train_kemar_fnssl_diffusefg_nofb_seed44_g5.yaml cuda:5 \
      outputs/checkpoints_kemar_fnssl_diffusefg_officialsplit_nofb_seed44_g5 \
      outputs/logs_kemar_fnssl_diffusefg_officialsplit_nofb_seed44_g5
    ;;
  gpu6)
    wait_session_done train_sdel_official_seed45_g6
    run_train dprtf_official_seed43_g6 configs/train_kemar_dprtf_doa_cls_diffusefg_seed43_g5.yaml cuda:6 \
      outputs/checkpoints_kemar_dprtf_doa_cls_diffusefg_officialsplit_seed43_g6 \
      outputs/logs_kemar_dprtf_doa_cls_diffusefg_officialsplit_seed43_g6
    run_train bil_official_seed43_g6 configs/train_kemar_bilstyle_gccphat_crn72_diffusefg_seed43_g6.yaml cuda:6 \
      outputs/checkpoints_kemar_bilstyle_gccphat_crn72_diffusefg_officialsplit_seed43_g6 \
      outputs/logs_kemar_bilstyle_gccphat_crn72_diffusefg_officialsplit_seed43_g6
    run_train fnssl_official_seed42_g6 configs/train_kemar_fnssl_diffusefg_g5_stable.yaml cuda:6 \
      outputs/checkpoints_kemar_fnssl_diffusefg_officialsplit_seed42_g6 \
      outputs/logs_kemar_fnssl_diffusefg_officialsplit_seed42_g6
    ;;
  *)
    echo "Usage: $0 {gpu4|gpu5|gpu6}" >&2
    exit 2
    ;;
esac
