#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/disk2/bywang/DOA-net"
PYTHON_BIN="/home/bywang/miniconda3/envs/doa/bin/python"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config_path> [extra train_moving.py args...]"
  exit 1
fi

CONFIG_PATH="$1"
shift || true

cd "$ROOT_DIR"

mapfile -t CFG_VALUES < <(
  "$PYTHON_BIN" - <<'PY' "$CONFIG_PATH"
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print(cfg["output"]["save_dir"])
print(cfg["output"]["log_dir"])
PY
)

SAVE_DIR="${CFG_VALUES[0]}"
LOG_DIR="${CFG_VALUES[1]}"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

PID_FILE="$LOG_DIR/train.pid"
STATUS_FILE="$LOG_DIR/train_status.txt"
STDOUT_LOG="$LOG_DIR/train_stdout.log"
CMD_FILE="$LOG_DIR/launch_command.sh"
RUNNER_FILE="$LOG_DIR/run_with_status.sh"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Moving training is already running with PID=$OLD_PID"
    echo "Log dir: $LOG_DIR"
    exit 0
  fi
fi

TRAIN_CMD=(
  "$PYTHON_BIN" -u train_moving.py
  --config "$CONFIG_PATH"
  "$@"
)

printf '%q ' "${TRAIN_CMD[@]}" > "$CMD_FILE"
printf '\n' >> "$CMD_FILE"
chmod +x "$CMD_FILE"

cat > "$RUNNER_FILE" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(printf '%q' "$ROOT_DIR")
{
  echo "runner_start_time=\$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "runner_pid=\$$"
} >> $(printf '%q' "$STATUS_FILE")
set +e
$(cat "$CMD_FILE")
EXIT_CODE=\$?
set -e
{
  echo "end_time=\$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "exit_code=\$EXIT_CODE"
} >> $(printf '%q' "$STATUS_FILE")
exit \$EXIT_CODE
EOF
chmod +x "$RUNNER_FILE"

{
  echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "config=$CONFIG_PATH"
  echo "save_dir=$SAVE_DIR"
  echo "log_dir=$LOG_DIR"
  echo "stdout_log=$STDOUT_LOG"
  echo "launch_command=$(cat "$CMD_FILE")"
} > "$STATUS_FILE"

nohup "$RUNNER_FILE" > "$STDOUT_LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PID_FILE"
echo "pid=$PID" >> "$STATUS_FILE"

echo "Started moving training in background."
echo "PID: $PID"
echo "Config: $CONFIG_PATH"
echo "Checkpoint dir: $SAVE_DIR"
echo "Log dir: $LOG_DIR"
echo "Stdout log: $STDOUT_LOG"
