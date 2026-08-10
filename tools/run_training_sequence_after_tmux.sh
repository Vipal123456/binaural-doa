#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/disk2/bywang/DOA-net"
TRAIN_LAUNCHER="$ROOT_DIR/tools/run_training_tmux.sh"

usage() {
  echo "Usage: $0 <watcher_session> <wait_session> <gpu_uuid> <next_session> <config_path> [<next_session> <config_path> ...]"
}

if [[ "${1:-}" == "--worker" ]]; then
  if [[ $# -lt 6 ]] || (( ( $# - 4 ) % 2 != 0 )); then
    usage
    exit 1
  fi
  WAIT_SESSION="$2"
  GPU_UUID="$3"
  WATCH_LOG="$4"
  shift 4
  cd "$ROOT_DIR"
  mkdir -p "$(dirname "$WATCH_LOG")"
  exec >>"$WATCH_LOG" 2>&1
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] waiting for $WAIT_SESSION"
  while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
    sleep 60
  done

  while [[ $# -ge 2 ]]; do
    NEXT_SESSION="$1"
    CONFIG_PATH="$2"
    shift 2
    echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] launching $NEXT_SESSION on $GPU_UUID"
    DOA_CUDA_VISIBLE_DEVICES="$GPU_UUID" \
      "$TRAIN_LAUNCHER" "$NEXT_SESSION" "$CONFIG_PATH"
    while tmux has-session -t "$NEXT_SESSION" 2>/dev/null; do
      sleep 60
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] $NEXT_SESSION finished"
  done
  exit 0
fi

if [[ $# -lt 5 ]] || (( ( $# - 3 ) % 2 != 0 )); then
  usage
  exit 1
fi

WATCHER_SESSION="$1"
WAIT_SESSION="$2"
GPU_UUID="$3"
shift 3
WATCH_LOG="$ROOT_DIR/outputs/training_watchers/${WATCHER_SESSION}.log"

cd "$ROOT_DIR"
if ! tmux has-session -t "$WAIT_SESSION" 2>/dev/null; then
  echo "Session to wait for does not exist: $WAIT_SESSION"
  exit 1
fi
if tmux has-session -t "$WATCHER_SESSION" 2>/dev/null; then
  echo "Watcher session already exists: $WATCHER_SESSION"
  exit 0
fi

ARGS=("$@")
for ((index = 1; index < ${#ARGS[@]}; index += 2)); do
  if [[ ! -f "${ARGS[index]}" ]]; then
    echo "Config not found: ${ARGS[index]}"
    exit 1
  fi
done

mkdir -p "$(dirname "$WATCH_LOG")"
tmux new-session -d -s "$WATCHER_SESSION" \
  "$0" --worker "$WAIT_SESSION" "$GPU_UUID" "$WATCH_LOG" "$@"

echo "Started sequential training watcher."
echo "Watcher: $WATCHER_SESSION"
echo "Waiting for: $WAIT_SESSION"
echo "GPU UUID: $GPU_UUID"
echo "Log: $WATCH_LOG"
