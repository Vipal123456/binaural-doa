#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/disk2/bywang/DOA-net"
PYTHON_BIN="/home/bywang/miniconda3/envs/doa/bin/python"
DATA_ROOT="${ROOT_DIR}/data/librispeech_cipic_roomsim25_anf_nonstationary_v3"
GENERATOR_PATTERN="tools/generate_cipic_roomsim25_anf_nonstationary_v3.py --output_root ${DATA_ROOT}"
QUEUE_LOG_DIR="${ROOT_DIR}/outputs/logs_cipic_roomsim25_anf_nonstationary_v3_six_model_queue"
QUEUE_LOG="${QUEUE_LOG_DIR}/queue.log"
STATUS_FILE="${QUEUE_LOG_DIR}/job_status.tsv"

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

echo "[$(timestamp)] Waiting for the ANF nonstationary v3 dataset."
while pgrep -f "${GENERATOR_PATTERN}" >/dev/null 2>&1; do
  train_progress="$(tr -d '\n' < "${DATA_ROOT}/train/progress.json" 2>/dev/null || true)"
  echo "[$(timestamp)] Dataset generation is active. train_progress=${train_progress:-unavailable}"
  sleep 60
done

echo "[$(timestamp)] Generator process ended; validating the completed dataset."
"${PYTHON_BIN}" - "${DATA_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {"train": 101250, "val": 9000, "test": 40500}
manifest_path = root / "manifest.json"
quality_path = root / "quality_report.json"
if not manifest_path.is_file() or not quality_path.is_file():
    raise SystemExit("Dataset generation ended without manifest.json and quality_report.json")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
quality = json.loads(quality_path.read_text(encoding="utf-8"))
if manifest.get("mode") != "full":
    raise SystemExit(f"Expected full dataset, got mode={manifest.get('mode')!r}")
if manifest.get("quality_passed") is not True or quality.get("passed") is not True:
    raise SystemExit("Dataset quality check did not pass")
if manifest.get("clip_counts") != expected:
    raise SystemExit(f"Unexpected manifest clip counts: {manifest.get('clip_counts')}")

for split, count in expected.items():
    metadata_path = root / split / "metadata.csv"
    if not metadata_path.is_file():
        raise SystemExit(f"Missing metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        rows = sum(1 for _ in csv.DictReader(handle))
    if rows != count:
        raise SystemExit(f"{split}: expected {count} metadata rows, got {rows}")

print("Dataset validation passed:", expected)
PY

echo "[$(timestamp)] Dataset validation passed; starting GPU queues."

wait_for_gpu() {
  local gpu="$1"
  while true; do
    if nvidia-smi -i "${gpu}" --query-gpu=memory.free,memory.total \
        --format=csv,noheader,nounits > "${QUEUE_LOG_DIR}/gpu${gpu}_memory.tmp" 2>/dev/null; then
      read -r free_mb total_mb < <(tr -d ',' < "${QUEUE_LOG_DIR}/gpu${gpu}_memory.tmp")
      if [[ "${free_mb}" =~ ^[0-9]+$ ]] && [[ "${total_mb}" =~ ^[0-9]+$ ]] \
          && (( free_mb * 100 >= total_mb * 80 )); then
        echo "[$(timestamp)] GPU ${gpu} ready: ${free_mb}/${total_mb} MiB free."
        return 0
      fi
      echo "[$(timestamp)] GPU ${gpu} busy: ${free_mb:-?}/${total_mb:-?} MiB free; waiting."
    else
      echo "[$(timestamp)] GPU ${gpu} is not queryable; waiting."
    fi
    sleep 60
  done
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
  if [[ "${exit_code}" -eq 0 ]]; then
    record_status "${model}" "${gpu}" "COMPLETED" "${save_dir}"
    echo "[$(timestamp)] Completed ${model} on GPU ${gpu}."
  else
    record_status "${model}" "${gpu}" "FAILED" "exit_code=${exit_code}"
    echo "[$(timestamp)] Failed ${model} on GPU ${gpu} with exit code ${exit_code}."
    return "${exit_code}"
  fi
}

queue_gpu0() {
  set -e
  run_job 0 "LocalTF32" \
    configs/train_cipic_roomsim25_v1_v7_localtf32_contextonly_bestmae_seed42.yaml \
    cipic_roomsim25_anf_nonstationary_v3_v7_localtf32_contextonly_bestmae_seed42
  run_job 0 "GRU128" \
    configs/train_cipic_roomsim25_v1_v7_localtf32_contextonly_gru128_bestmae_seed42.yaml \
    cipic_roomsim25_anf_nonstationary_v3_v7_localtf32_contextonly_gru128_bestmae_seed42
}

queue_gpu2() {
  set -e
  run_job 2 "CPSD5-All" \
    configs/train_cipic_roomsim25_v1_v7_localtf32_contextonly_cpsd5_all_bestmae_seed42.yaml \
    cipic_roomsim25_anf_nonstationary_v3_v7_localtf32_contextonly_cpsd5_all_bestmae_seed42
  run_job 2 "CPSD5-CueOnly" \
    configs/train_cipic_roomsim25_v1_v7_localtf32_contextonly_cpsd5_cue_bestmae_seed42.yaml \
    cipic_roomsim25_anf_nonstationary_v3_v7_localtf32_contextonly_cpsd5_cue_bestmae_seed42
}

queue_gpu3() {
  set -e
  run_job 3 "DP-RTF" \
    configs/train_cipic_roomsim25_v1_dprtf_trainmean_bestmae_seed42_phys3.yaml \
    cipic_roomsim25_anf_nonstationary_v3_dprtf_trainmean_bestmae_seed42
}

queue_gpu7() {
  set -e
  run_job 7 "SDEL" \
    configs/train_cipic_roomsim25_v1_sdel_bestmae_seed42_phys2.yaml \
    cipic_roomsim25_anf_nonstationary_v3_sdel_bestmae_seed42
}

set +e
queue_gpu0 & pid0=$!
queue_gpu2 & pid2=$!
queue_gpu3 & pid3=$!
queue_gpu7 & pid7=$!

wait "${pid0}"; status0=$?
wait "${pid2}"; status2=$?
wait "${pid3}"; status3=$?
wait "${pid7}"; status7=$?
set -e

echo "[$(timestamp)] GPU queue exit codes: gpu0=${status0} gpu2=${status2} gpu3=${status3} gpu7=${status7}"
if (( status0 != 0 || status2 != 0 || status3 != 0 || status7 != 0 )); then
  exit 1
fi
echo "[$(timestamp)] All six model training jobs completed successfully."
