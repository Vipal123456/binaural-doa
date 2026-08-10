#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/disk2/bywang/DOA-net"
PYTHON_BIN="/home/bywang/miniconda3/envs/doa/bin/python"

run_worker() {
  local physical_gpu="$1"
  shift

  cd "$ROOT_DIR"
  while [[ $# -ge 2 ]]; do
    local config_path="$1"
    local eval_output_dir="$2"
    shift 2

    mapfile -t config_values < <(
      "$PYTHON_BIN" - "$config_path" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
print(cfg["output"]["save_dir"])
print(cfg["output"]["log_dir"])
print(cfg["dataset"]["test_root"])
PY
    )
    local save_dir="${config_values[0]}"
    local log_dir="${config_values[1]}"
    local test_root="${config_values[2]}"
    local status_file="$log_dir/pipeline_status.txt"
    local resolved_config="$log_dir/resolved_config.yaml"
    local checkpoint="$save_dir/best_mae.pth"

    mkdir -p "$save_dir" "$log_dir" "$eval_output_dir"
    {
      echo "physical_gpu=$physical_gpu"
      echo "config=$config_path"
      echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
      echo "state=training"
    } > "$status_file"

    local resume_args=()
    if [[ -f "$save_dir/latest.pth" ]]; then
      resume_args=(--resume "$save_dir/latest.pth")
      echo "resume_from=$save_dir/latest.pth" >> "$status_file"
    fi

    set +e
    env CUDA_VISIBLE_DEVICES="$physical_gpu" "$PYTHON_BIN" -u train.py \
      --config "$config_path" \
      "${resume_args[@]}" \
      --train.device cuda:0 \
      --train.num_workers 8
    local train_exit=$?
    set -e
    echo "train_exit_code=$train_exit" >> "$status_file"
    if [[ "$train_exit" -ne 0 ]]; then
      echo "state=train_failed" >> "$status_file"
      return "$train_exit"
    fi

    echo "state=evaluating" >> "$status_file"
    set +e
    env CUDA_VISIBLE_DEVICES="$physical_gpu" "$PYTHON_BIN" -u tools/evaluate_kemar_grouped.py \
      --config "$resolved_config" \
      --checkpoint "$checkpoint" \
      --test_root "$test_root" \
      --output_dir "$eval_output_dir" \
      --batch_size 64 \
      --num_workers 8 \
      --device cuda:0 \
      --log_interval 20 \
      > "$eval_output_dir/eval.log" 2>&1
    local eval_exit=$?
    set -e
    {
      echo "eval_exit_code=$eval_exit"
      echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    } >> "$status_file"
    if [[ "$eval_exit" -ne 0 ]]; then
      echo "state=eval_failed" >> "$status_file"
      return "$eval_exit"
    fi
    echo "state=completed" >> "$status_file"
  done
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  if [[ $# -lt 3 || $((($# - 1) % 2)) -ne 0 ]]; then
    echo "Worker usage: $0 --worker <physical_gpu> <config> <eval_output> [...]"
    exit 1
  fi
  run_worker "$@"
  exit $?
fi

if [[ $# -lt 4 || $((($# - 2) % 2)) -ne 0 ]]; then
  echo "Usage: $0 <session> <physical_gpu> <config> <eval_output> [<config> <eval_output> ...]"
  exit 1
fi

SESSION_NAME="$1"
PHYSICAL_GPU="$2"
shift 2

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  exit 1
fi

printf -v WORKER_COMMAND '%q ' "$0" --worker "$PHYSICAL_GPU" "$@"
tmux new-session -d -s "$SESSION_NAME" "$WORKER_COMMAND"

echo "Started CIPIC train/eval queue."
echo "Session: $SESSION_NAME"
echo "Physical GPU: $PHYSICAL_GPU"
echo "Attach: tmux attach -t $SESSION_NAME"
