#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/disk2/bywang/DOA-net"
PYTHON_BIN="/home/bywang/miniconda3/envs/doa/bin/python"
TEST_ROOT="${ROOT_DIR}/data/librispeech_cipic_roomsim25_anf_nonstationary_v3/test"
QUEUE_LOG_DIR="${ROOT_DIR}/outputs/logs_cipic_roomsim25_anf_nonstationary_v3_completed_eval_queue"

mkdir -p "${QUEUE_LOG_DIR}"
cd "${ROOT_DIR}"

run_eval() {
  local gpu="$1"
  local name="$2"
  local config="$3"
  local checkpoint="$4"
  local output_dir="$5"
  local batch_size="$6"

  mkdir -p "${output_dir}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] Starting ${name} on GPU ${gpu}" \
    | tee -a "${QUEUE_LOG_DIR}/queue.log"
  env CUDA_DEVICE_ORDER=PCI_BUS_ID "CUDA_VISIBLE_DEVICES=${gpu}" \
    "${PYTHON_BIN}" -u tools/evaluate_kemar_grouped.py \
      --config "${config}" \
      --checkpoint "${checkpoint}" \
      --output_dir "${output_dir}" \
      --test_root "${TEST_ROOT}" \
      --batch_size "${batch_size}" \
      --num_workers 8 \
      --device cuda:0 \
      --log_interval 100 \
      2>&1 | tee "${output_dir}/eval.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] Completed ${name}" \
    | tee -a "${QUEUE_LOG_DIR}/queue.log"
}

queue_gpu0() {
  run_eval 0 LocalTF32 \
    outputs/logs_cipic_roomsim25_anf_nonstationary_v3_v7_localtf32_contextonly_bestmae_seed42/resolved_config.yaml \
    outputs/checkpoints_cipic_roomsim25_anf_nonstationary_v3_v7_localtf32_contextonly_bestmae_seed42/best_mae.pth \
    outputs/logs_cipic_roomsim25_anf_nonstationary_v3_v7_localtf32_contextonly_bestmae_seed42/grouped_test_best_mae \
    64
  run_eval 0 GRU128 \
    outputs/logs_cipic_roomsim25_anf_nonstationary_v3_v7_localtf32_contextonly_gru128_bestmae_seed42/resolved_config.yaml \
    outputs/checkpoints_cipic_roomsim25_anf_nonstationary_v3_v7_localtf32_contextonly_gru128_bestmae_seed42/best_mae.pth \
    outputs/logs_cipic_roomsim25_anf_nonstationary_v3_v7_localtf32_contextonly_gru128_bestmae_seed42/grouped_test_best_mae \
    64
}

queue_gpu2() {
  run_eval 2 CPSD5-All \
    outputs/logs_cipic_roomsim25_anf_nonstationary_v3_v7_localtf32_contextonly_cpsd5_all_bestmae_seed42/resolved_config.yaml \
    outputs/checkpoints_cipic_roomsim25_anf_nonstationary_v3_v7_localtf32_contextonly_cpsd5_all_bestmae_seed42/best_mae.pth \
    outputs/logs_cipic_roomsim25_anf_nonstationary_v3_v7_localtf32_contextonly_cpsd5_all_bestmae_seed42/grouped_test_best_mae \
    64
  run_eval 2 CPSD5-CueOnly \
    outputs/logs_cipic_roomsim25_anf_nonstationary_v3_v7_localtf32_contextonly_cpsd5_cue_bestmae_seed42/resolved_config.yaml \
    outputs/checkpoints_cipic_roomsim25_anf_nonstationary_v3_v7_localtf32_contextonly_cpsd5_cue_bestmae_seed42/best_mae.pth \
    outputs/logs_cipic_roomsim25_anf_nonstationary_v3_v7_localtf32_contextonly_cpsd5_cue_bestmae_seed42/grouped_test_best_mae \
    64
}

queue_gpu3() {
  run_eval 3 DP-RTF \
    outputs/logs_cipic_roomsim25_anf_nonstationary_v3_dprtf_trainmean_bestmae_seed42/resolved_config.yaml \
    outputs/checkpoints_cipic_roomsim25_anf_nonstationary_v3_dprtf_trainmean_bestmae_seed42/best_mae.pth \
    outputs/logs_cipic_roomsim25_anf_nonstationary_v3_dprtf_trainmean_bestmae_seed42/grouped_test_best_mae \
    32
}

queue_gpu0 & pid0=$!
queue_gpu2 & pid2=$!
queue_gpu3 & pid3=$!

wait "${pid0}"
wait "${pid2}"
wait "${pid3}"

echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] All completed-model evaluations finished." \
  | tee -a "${QUEUE_LOG_DIR}/queue.log"
