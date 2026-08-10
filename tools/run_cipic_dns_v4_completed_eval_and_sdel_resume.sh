#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/disk2/bywang/DOA-net"
PYTHON_BIN="/home/bywang/miniconda3/envs/doa/bin/python"
TEST_ROOT="${ROOT_DIR}/data/librispeech_cipic_roomsim25_directional_dns_v4/test"
QUEUE_DIR="${ROOT_DIR}/outputs/logs_cipic_roomsim25_directional_dns_v4_eval_queue"
mkdir -p "${QUEUE_DIR}"
cd "${ROOT_DIR}"

run_eval() {
  local gpu="$1" name="$2" tag="$3" batch_size="$4"
  local log_dir="outputs/logs_${tag}"
  local checkpoint_dir="outputs/checkpoints_${tag}"
  local output_dir="${log_dir}/grouped_test_best_mae"
  mkdir -p "${output_dir}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] START ${name} GPU=${gpu}" | tee -a "${QUEUE_DIR}/queue.log"
  env CUDA_DEVICE_ORDER=PCI_BUS_ID "CUDA_VISIBLE_DEVICES=${gpu}" \
    "${PYTHON_BIN}" -u tools/evaluate_kemar_grouped.py \
      --config "${log_dir}/resolved_config.yaml" \
      --checkpoint "${checkpoint_dir}/best_mae.pth" \
      --output_dir "${output_dir}" \
      --test_root "${TEST_ROOT}" \
      --batch_size "${batch_size}" \
      --num_workers 8 \
      --device cuda:0 \
      --log_interval 100 \
      2>&1 | tee "${output_dir}/eval.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] DONE ${name}" | tee -a "${QUEUE_DIR}/queue.log"
}

queue_gpu2() {
  run_eval 2 LocalTF32 \
    cipic_roomsim25_directional_dns_v4_v7_localtf32_contextonly_bestmae_seed42 64
  run_eval 2 CPSD5-CueOnly \
    cipic_roomsim25_directional_dns_v4_v7_localtf32_contextonly_cpsd5_cue_bestmae_seed42 64
  run_eval 2 CPSD5-All \
    cipic_roomsim25_directional_dns_v4_v7_localtf32_contextonly_cpsd5_all_bestmae_seed42 64
}

queue_gpu3() {
  run_eval 3 DP-RTF \
    cipic_roomsim25_directional_dns_v4_dprtf_trainmean_bestmae_seed42 32
  run_eval 3 BIL-GCCPHAT-CRN25 \
    cipic_roomsim25_directional_dns_v4_bilstyle_gccphat_crn25_bestmae_seed42 64

  local tag="cipic_roomsim25_directional_dns_v4_sdel_bestmae_seed42"
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] RESUME SDEL GPU=3" | tee -a "${QUEUE_DIR}/queue.log"
  env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
    "${PYTHON_BIN}" -u train.py \
      --config configs/train_cipic_roomsim25_directional_dns_v4_sdel_bestmae_seed42.yaml \
      --resume "outputs/checkpoints_${tag}/latest.pth" \
      2>&1 | tee -a "outputs/logs_${tag}/resume_gpu3_stdout.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] DONE SDEL GPU=3" | tee -a "${QUEUE_DIR}/queue.log"
}

queue_gpu2 & pid2=$!
queue_gpu3 & pid3=$!
wait "${pid2}"
wait "${pid3}"
echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] ALL DONE" | tee -a "${QUEUE_DIR}/queue.log"
