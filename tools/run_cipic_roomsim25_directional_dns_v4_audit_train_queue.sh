#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/disk2/bywang/DOA-net"
PYTHON_BIN="/home/bywang/miniconda3/envs/doa/bin/python"
DATA_ROOT="${ROOT_DIR}/data/librispeech_cipic_roomsim25_directional_dns_v4"
NOISE_INVENTORY="${ROOT_DIR}/data/dns3_directional_v4_inventory/dns3_noise_inventory.csv"
QUEUE_LOG_DIR="${ROOT_DIR}/outputs/logs_cipic_roomsim25_directional_dns_v4_seven_model_queue"
QUEUE_LOG="${QUEUE_LOG_DIR}/queue.log"
STATUS_FILE="${QUEUE_LOG_DIR}/job_status.tsv"
AUDIT_REPORT="${QUEUE_LOG_DIR}/dataset_audit.json"
PREFLIGHT_REPORT="${QUEUE_LOG_DIR}/model_preflight.json"
GENERATOR="tools/generate_cipic_roomsim25_directional_dns_v4.py"

LOCALTF_CONFIG="configs/train_cipic_roomsim25_v1_v7_localtf32_contextonly_bestmae_seed42.yaml"
CPSD_CUE_CONFIG="configs/train_cipic_roomsim25_v1_v7_localtf32_contextonly_cpsd5_cue_bestmae_seed42.yaml"
CPSD_ALL_CONFIG="configs/train_cipic_roomsim25_v1_v7_localtf32_contextonly_cpsd5_all_bestmae_seed42.yaml"
SDEL_CONFIG="configs/train_cipic_roomsim25_directional_dns_v4_sdel_bestmae_seed42.yaml"
DPRTF_CONFIG="configs/train_cipic_roomsim25_v1_dprtf_trainmean_bestmae_seed42_phys3.yaml"
RAWCONCAT_CONFIG="configs/train_cipic_roomsim25_v1_v7_dualcue_liteenc_v1_rawconcat_bestmae_seed42_phys2.yaml"
BIL_CONFIG="configs/train_cipic_roomsim25_directional_dns_v4_bilstyle_gccphat_crn25_bestmae_seed42.yaml"

mkdir -p "${QUEUE_LOG_DIR}"
cd "${ROOT_DIR}"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

record_status() {
  local model="$1"
  local gpu="$2"
  local state="$3"
  local detail="${4:-}"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(timestamp)" "${model}" "${gpu}" "${state}" "${detail}" >> "${STATUS_FILE}"
}

generator_running() {
  pgrep -f "[g]enerate_cipic_roomsim25_directional_dns_v4.py.*--output_root ${DATA_ROOT}" >/dev/null 2>&1 \
    || pgrep -f "[r]un_cipic_roomsim25_directional_dns_v4_pipeline.sh" >/dev/null 2>&1
}

resume_generation_once() {
  echo "[$(timestamp)] Dataset is incomplete and no generator is active; attempting one deterministic resume."
  "${PYTHON_BIN}" -u "${GENERATOR}" \
    --noise_inventory "${NOISE_INVENTORY}" \
    --output_root "${DATA_ROOT}" \
    --mode full \
    --workers 6 \
    --parallel_splits \
    --resume
}

echo "[$(timestamp)] Waiting for the directional DNS v4 release manifest."
resume_attempted=0
while [[ ! -s "${DATA_ROOT}/manifest.json" || ! -s "${DATA_ROOT}/quality_report.json" ]]; do
  if generator_running; then
    progress="$(tr -d '\n' < "${DATA_ROOT}/train/progress.json" 2>/dev/null || true)"
    echo "[$(timestamp)] Dataset generation active. train_progress=${progress:-unavailable}"
    sleep 60
    continue
  fi
  if (( resume_attempted == 0 )); then
    resume_attempted=1
    resume_generation_once
    continue
  fi
  echo "[$(timestamp)] Dataset remains incomplete after resume; training gate is closed."
  record_status "DATASET" "-" "FAILED" "missing manifest/quality report"
  exit 1
done

echo "[$(timestamp)] Manifest exists; running generator regression tests."
"${PYTHON_BIN}" -m pytest -q \
  tests/test_build_dns3_noise_inventory.py \
  tests/test_generate_cipic_roomsim25_directional_dns_v4.py

echo "[$(timestamp)] Running full dataset release audit."
set +e
"${PYTHON_BIN}" -u tools/audit_cipic_roomsim25_directional_dns_v4.py \
  --dataset-root "${DATA_ROOT}" \
  --noise-inventory "${NOISE_INVENTORY}" \
  --report "${AUDIT_REPORT}" \
  --workers 16
audit_status=$?
set -e
if (( audit_status != 0 )); then
  record_status "DATASET" "-" "FAILED_AUDIT" "${AUDIT_REPORT}"
  echo "[$(timestamp)] Full audit failed; no training job will be started."
  exit "${audit_status}"
fi
record_status "DATASET" "-" "PASSED_AUDIT" "${AUDIT_REPORT}"

configs=(
  "${LOCALTF_CONFIG}"
  "${CPSD_CUE_CONFIG}"
  "${CPSD_ALL_CONFIG}"
  "${SDEL_CONFIG}"
  "${DPRTF_CONFIG}"
  "${RAWCONCAT_CONFIG}"
  "${BIL_CONFIG}"
)

echo "[$(timestamp)] Running real-sample CPU forward preflight for all seven models."
set +e
"${PYTHON_BIN}" -u tools/preflight_cipic_roomsim25_directional_dns_v4_models.py \
  --dataset-root "${DATA_ROOT}" \
  --report "${PREFLIGHT_REPORT}" \
  "${configs[@]}"
preflight_status=$?
set -e
if (( preflight_status != 0 )); then
  record_status "MODELS" "-" "FAILED_PREFLIGHT" "${PREFLIGHT_REPORT}"
  echo "[$(timestamp)] Model preflight failed; no GPU training job will be started."
  exit "${preflight_status}"
fi
record_status "MODELS" "-" "PASSED_PREFLIGHT" "${PREFLIGHT_REPORT}"

available_kb="$(df --output=avail -k /disk2 | tail -n 1 | tr -d ' ')"
if [[ ! "${available_kb}" =~ ^[0-9]+$ ]] || (( available_kb < 40 * 1024 * 1024 )); then
  record_status "STORAGE" "-" "FAILED" "available_kb=${available_kb:-unknown}"
  echo "[$(timestamp)] Less than 40 GiB remains on /disk2; refusing to start seven trainings."
  exit 1
fi

wait_for_gpu() {
  local gpu="$1"
  local attempts=0
  while (( attempts < 120 )); do
    local line=""
    line="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free,memory.total,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -n "${line}" ]]; then
      local free_mb total_mb util
      IFS=',' read -r free_mb total_mb util <<< "${line}"
      free_mb="${free_mb//[[:space:]]/}"
      total_mb="${total_mb//[[:space:]]/}"
      util="${util//[[:space:]]/}"
      if [[ "${free_mb}" =~ ^[0-9]+$ && "${total_mb}" =~ ^[0-9]+$ && "${util}" =~ ^[0-9]+$ ]] \
          && (( free_mb * 100 >= total_mb * 80 && util <= 10 )); then
        echo "[$(timestamp)] GPU ${gpu} ready: ${free_mb}/${total_mb} MiB free, util=${util}%."
        return 0
      fi
      echo "[$(timestamp)] GPU ${gpu} busy: ${free_mb:-?}/${total_mb:-?} MiB, util=${util:-?}%; waiting."
    else
      echo "[$(timestamp)] GPU ${gpu} is not queryable; waiting."
    fi
    attempts=$((attempts + 1))
    sleep 60
  done
  record_status "GPU" "${gpu}" "FAILED" "not ready after 120 minutes"
  return 1
}

run_job() {
  local gpu="$1"
  local model="$2"
  local config="$3"
  local output_tag="$4"
  local save_dir="outputs/checkpoints_${output_tag}"
  local log_dir="outputs/logs_${output_tag}"
  local stdout_log="${log_dir}/queue_stdout.log"
  local command_file="${log_dir}/queue_launch_command.sh"

  wait_for_gpu "${gpu}"
  mkdir -p "${save_dir}" "${log_dir}"
  local resume_args=()
  if [[ -f "${save_dir}/latest.pth" ]]; then
    resume_args=(--resume "${save_dir}/latest.pth")
  fi
  local command=(
    env CUDA_DEVICE_ORDER=PCI_BUS_ID "CUDA_VISIBLE_DEVICES=${gpu}"
    "${PYTHON_BIN}" -u train.py
    --config "${config}"
    --dataset.root_dir "${DATA_ROOT}/train"
    --dataset.train_root "${DATA_ROOT}/train"
    --dataset.val_root "${DATA_ROOT}/val"
    --dataset.test_root "${DATA_ROOT}/test"
    --output.save_dir "${save_dir}"
    --output.log_dir "${log_dir}"
    --train.num_workers 8
    "${resume_args[@]}"
  )
  printf '%q ' "${command[@]}" > "${command_file}"
  printf '\n' >> "${command_file}"
  chmod +x "${command_file}"

  record_status "${model}" "${gpu}" "STARTED" "${config}"
  echo "[$(timestamp)] Starting ${model} on physical GPU ${gpu}. Log: ${log_dir}/train.log"
  set +e
  "${command[@]}" 2>&1 | tee -a "${stdout_log}"
  local exit_code="${PIPESTATUS[0]}"
  set -e
  if (( exit_code == 0 )); then
    record_status "${model}" "${gpu}" "COMPLETED" "${save_dir}"
    echo "[$(timestamp)] Completed ${model} on GPU ${gpu}."
  else
    record_status "${model}" "${gpu}" "FAILED" "exit_code=${exit_code}"
    echo "[$(timestamp)] Failed ${model} on GPU ${gpu}, exit_code=${exit_code}."
    return "${exit_code}"
  fi
}

queue_gpu0() {
  local failed=0
  run_job 0 "LocalTF32-ContextOnly" "${LOCALTF_CONFIG}" \
    "cipic_roomsim25_directional_dns_v4_v7_localtf32_contextonly_bestmae_seed42" || failed=1
  run_job 0 "LiteV1-RawConcat" "${RAWCONCAT_CONFIG}" \
    "cipic_roomsim25_directional_dns_v4_v7_dualcue_liteenc_v1_rawconcat_bestmae_seed42" || failed=1
  return "${failed}"
}

queue_gpu2() {
  local failed=0
  run_job 2 "CPSD5-CueOnly" "${CPSD_CUE_CONFIG}" \
    "cipic_roomsim25_directional_dns_v4_v7_localtf32_contextonly_cpsd5_cue_bestmae_seed42" || failed=1
  run_job 2 "CPSD5-All" "${CPSD_ALL_CONFIG}" \
    "cipic_roomsim25_directional_dns_v4_v7_localtf32_contextonly_cpsd5_all_bestmae_seed42" || failed=1
  return "${failed}"
}

queue_gpu3() {
  local failed=0
  run_job 3 "DP-RTF" "${DPRTF_CONFIG}" \
    "cipic_roomsim25_directional_dns_v4_dprtf_trainmean_bestmae_seed42" || failed=1
  run_job 3 "BIL-GCCPHAT-CRN25" "${BIL_CONFIG}" \
    "cipic_roomsim25_directional_dns_v4_bilstyle_gccphat_crn25_bestmae_seed42" || failed=1
  # SDEL's two-layer bidirectional GRU is abnormally slow on physical GPU 7
  # on this host. Keep it on the same known-good GPU used by diffuse-v2.
  run_job 3 "SDEL" "${SDEL_CONFIG}" \
    "cipic_roomsim25_directional_dns_v4_sdel_bestmae_seed42" || failed=1
  return "${failed}"
}

echo "[$(timestamp)] All gates passed; launching three physical-GPU queues."
set +e
queue_gpu0 & pid0=$!
queue_gpu2 & pid2=$!
queue_gpu3 & pid3=$!
wait "${pid0}"; status0=$?
wait "${pid2}"; status2=$?
wait "${pid3}"; status3=$?
set -e

echo "[$(timestamp)] Queue exit codes: gpu0=${status0}, gpu2=${status2}, gpu3=${status3}."
if (( status0 != 0 || status2 != 0 || status3 != 0 )); then
  exit 1
fi
echo "[$(timestamp)] All seven model training jobs completed successfully."
