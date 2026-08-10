#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/disk2/bywang/DOA-net"
PYTHON_BIN="/home/bywang/miniconda3/envs/doa/bin/python"
TRAIN_LAUNCHER="$ROOT_DIR/tools/run_training_tmux.sh"

usage() {
  echo "Usage: $0 <wait_session> <train_session> <gpu_uuid> <config_path>"
}

if [[ "${1:-}" == "--worker" ]]; then
  if [[ $# -ne 6 ]]; then
    usage
    exit 1
  fi
  WAIT_SESSION="$2"
  TRAIN_SESSION="$3"
  GPU_UUID="$4"
  CONFIG_PATH="$5"
  WAIT_LOG="$6"

  cd "$ROOT_DIR"
  mkdir -p "$(dirname "$WAIT_LOG")"
  exec >>"$WAIT_LOG" 2>&1
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] waiting for CUDA device $GPU_UUID"

  ready_checks=0
  while true; do
    if CUDA_VISIBLE_DEVICES="$GPU_UUID" "$PYTHON_BIN" -c \
      'import sys, torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    sys.exit(1)
x = torch.ones((1024, 1024), device="cuda")
if x.sum().item() != 1048576:
    sys.exit(1)
torch.cuda.synchronize()' \
      >/dev/null 2>&1; then
      ready_checks=$((ready_checks + 1))
      echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] CUDA readiness check $ready_checks/3 passed"
      if [[ "$ready_checks" -ge 3 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] CUDA stable; launching $TRAIN_SESSION"
        DOA_CUDA_VISIBLE_DEVICES="$GPU_UUID" \
          "$TRAIN_LAUNCHER" "$TRAIN_SESSION" "$CONFIG_PATH"
        sleep 30
        if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
          echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] $TRAIN_SESSION remained alive for 30 seconds"
          exit 0
        fi
        echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] $TRAIN_SESSION exited during startup; resuming CUDA wait"
        ready_checks=0
      fi
      sleep 5
      continue
    fi
    ready_checks=0
    echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] CUDA unavailable; retrying in 60 seconds"
    sleep 60
  done
fi

if [[ $# -ne 4 ]]; then
  usage
  exit 1
fi

WAIT_SESSION="$1"
TRAIN_SESSION="$2"
GPU_UUID="$3"
CONFIG_PATH="$4"
WAIT_LOG="$ROOT_DIR/outputs/cuda_waiters/${WAIT_SESSION}.log"

cd "$ROOT_DIR"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH"
  exit 1
fi
if tmux has-session -t "$WAIT_SESSION" 2>/dev/null; then
  echo "tmux wait session already exists: $WAIT_SESSION"
  exit 0
fi

mkdir -p "$(dirname "$WAIT_LOG")"
tmux new-session -d -s "$WAIT_SESSION" \
  "$0" --worker "$WAIT_SESSION" "$TRAIN_SESSION" "$GPU_UUID" "$CONFIG_PATH" "$WAIT_LOG"

echo "Started CUDA wait session."
echo "Wait session: $WAIT_SESSION"
echo "Train session: $TRAIN_SESSION"
echo "GPU UUID: $GPU_UUID"
echo "Config: $CONFIG_PATH"
echo "Wait log: $WAIT_LOG"
