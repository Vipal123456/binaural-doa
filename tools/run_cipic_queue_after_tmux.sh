#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/disk2/bywang/DOA-net"

if [[ "${1:-}" == "--worker" ]]; then
  shift
  WAIT_SESSION="$1"
  QUEUE_SESSION="$2"
  PHYSICAL_GPU="$3"
  shift 3
  cd "$ROOT_DIR"
  while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
    sleep 60
  done
  exec ./tools/run_cipic_train_eval_queue_tmux.sh \
    "$QUEUE_SESSION" "$PHYSICAL_GPU" "$@"
fi

if [[ $# -lt 6 || $((($# - 4) % 2)) -ne 0 ]]; then
  echo "Usage: $0 <watcher_session> <wait_session> <queue_session> <physical_gpu> <config> <eval_output> [...]"
  exit 1
fi

WATCHER_SESSION="$1"
WAIT_SESSION="$2"
QUEUE_SESSION="$3"
PHYSICAL_GPU="$4"
shift 4

if tmux has-session -t "$WATCHER_SESSION" 2>/dev/null; then
  echo "tmux watcher already exists: $WATCHER_SESSION"
  exit 1
fi

printf -v WORKER_COMMAND '%q ' \
  "$0" --worker "$WAIT_SESSION" "$QUEUE_SESSION" "$PHYSICAL_GPU" "$@"
tmux new-session -d -s "$WATCHER_SESSION" "$WORKER_COMMAND"

echo "Started deferred CIPIC queue watcher."
echo "Watcher: $WATCHER_SESSION"
echo "Waiting for: $WAIT_SESSION"
echo "Next queue: $QUEUE_SESSION"
echo "Physical GPU: $PHYSICAL_GPU"
