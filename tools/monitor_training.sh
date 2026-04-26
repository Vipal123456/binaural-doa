#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/disk2/bywang/DOA-net"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <log_dir>"
  exit 1
fi

LOG_DIR="$1"
cd "$ROOT_DIR"

PID_FILE="$LOG_DIR/train.pid"
STATUS_FILE="$LOG_DIR/train_status.txt"
TRAIN_LOG="$LOG_DIR/train.log"
STDOUT_LOG="$LOG_DIR/train_stdout.log"

if [[ -f "$STATUS_FILE" ]]; then
  echo "== Status =="
  cat "$STATUS_FILE"
  echo
fi

if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${PID}" ]] && kill -0 "$PID" 2>/dev/null; then
    echo "== Process =="
    ps -fp "$PID"
  else
    echo "== Process =="
    echo "PID file exists, but process is not running."
  fi
  echo
fi

if [[ -f "$TRAIN_LOG" ]]; then
  echo "== train.log (tail 20) =="
  tail -n 20 "$TRAIN_LOG"
  echo
fi

if [[ -f "$STDOUT_LOG" ]]; then
  echo "== train_stdout.log (tail 20) =="
  tail -n 20 "$STDOUT_LOG"
  echo
fi

LATEST_CKPT="$(find "$ROOT_DIR/outputs" -path '*/latest.pth' -newer "$STATUS_FILE" 2>/dev/null | head -n 1 || true)"
if [[ -n "$LATEST_CKPT" ]]; then
  echo "== Recent Checkpoint =="
  stat "$LATEST_CKPT"
fi
