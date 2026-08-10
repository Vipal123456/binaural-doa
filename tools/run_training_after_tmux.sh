#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/disk2/bywang/DOA-net"
TRAIN_LAUNCHER="$ROOT_DIR/tools/run_training_tmux.sh"

usage() {
  echo "Usage: $0 <watcher_session> <wait_session> <next_session> <gpu_uuid> <config_path>"
}

if [[ "${1:-}" == "--worker" ]]; then
  if [[ $# -ne 6 ]]; then
    usage
    exit 1
  fi
  WAIT_SESSION="$2"
  NEXT_SESSION="$3"
  GPU_UUID="$4"
  CONFIG_PATH="$5"
  WATCH_LOG="$6"
  cd "$ROOT_DIR"
  mkdir -p "$(dirname "$WATCH_LOG")"
  exec >>"$WATCH_LOG" 2>&1
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] waiting for $WAIT_SESSION"
  while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
    sleep 60
  done
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] launching $NEXT_SESSION on $GPU_UUID"
  DOA_CUDA_VISIBLE_DEVICES="$GPU_UUID" \
    "$TRAIN_LAUNCHER" "$NEXT_SESSION" "$CONFIG_PATH"
  exit $?
fi

if [[ $# -ne 5 ]]; then
  usage
  exit 1
fi

WATCHER_SESSION="$1"
WAIT_SESSION="$2"
NEXT_SESSION="$3"
GPU_UUID="$4"
CONFIG_PATH="$5"
WATCH_LOG="$ROOT_DIR/outputs/training_watchers/${WATCHER_SESSION}.log"

cd "$ROOT_DIR"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH"
  exit 1
fi
if ! tmux has-session -t "$WAIT_SESSION" 2>/dev/null; then
  echo "Session to wait for does not exist: $WAIT_SESSION"
  exit 1
fi
if tmux has-session -t "$WATCHER_SESSION" 2>/dev/null; then
  echo "Watcher session already exists: $WATCHER_SESSION"
  exit 0
fi

mkdir -p "$(dirname "$WATCH_LOG")"
tmux new-session -d -s "$WATCHER_SESSION" \
  "$0" --worker "$WAIT_SESSION" "$NEXT_SESSION" "$GPU_UUID" "$CONFIG_PATH" "$WATCH_LOG"

echo "Started deferred training watcher."
echo "Watcher: $WATCHER_SESSION"
echo "Waiting for: $WAIT_SESSION"
echo "Next session: $NEXT_SESSION"
echo "GPU UUID: $GPU_UUID"
echo "Log: $WATCH_LOG"
