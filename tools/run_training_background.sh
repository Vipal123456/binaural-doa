#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/disk2/bywang/DOA-net"
PYTHON_BIN="/home/bywang/miniconda3/envs/doa/bin/python"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config_path> [extra train.py args...]"
  exit 1
fi

CONFIG_PATH="$1"
shift || true

cd "$ROOT_DIR"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH"
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found: $PYTHON_BIN"
  exit 1
fi

mapfile -t CFG_VALUES < <(
  "$PYTHON_BIN" - <<'PY' "$CONFIG_PATH"
import sys
import yaml

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

save_dir = cfg["output"]["save_dir"]
log_dir = cfg["output"]["log_dir"]
print(save_dir)
print(log_dir)
PY
)

SAVE_DIR="${CFG_VALUES[0]}"
LOG_DIR="${CFG_VALUES[1]}"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

PID_FILE="$LOG_DIR/train.pid"
STATUS_FILE="$LOG_DIR/train_status.txt"
CMD_FILE="$LOG_DIR/launch_command.sh"
RUNNER_FILE="$LOG_DIR/run_with_status.sh"
STDOUT_LOG="$LOG_DIR/train_stdout.log"
STDOUT_LOG_TS="$LOG_DIR/train_stdout_$(date +%Y%m%d_%H%M%S).log"
LATEST_CKPT="$SAVE_DIR/latest.pth"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Training is already running with PID=$OLD_PID"
    echo "Log dir: $LOG_DIR"
    exit 0
  fi
fi

RESUME_ARGS=()
if [[ -f "$LATEST_CKPT" ]]; then
  RESUME_ARGS=(--resume "$LATEST_CKPT")
fi

HAS_NUM_WORKERS_OVERRIDE=0
for arg in "$@"; do
  if [[ "$arg" == "--train.num_workers" ]] || [[ "$arg" == --train.num_workers=* ]]; then
    HAS_NUM_WORKERS_OVERRIDE=1
    break
  fi
done

SAFE_ARGS=()
if [[ "$HAS_NUM_WORKERS_OVERRIDE" -eq 0 ]]; then
  # Detached/background training is more stable with a single-process DataLoader.
  SAFE_ARGS=(--train.num_workers 0)
fi

TRAIN_CMD=(
  "$PYTHON_BIN" -u train.py
  --config "$CONFIG_PATH"
  "${RESUME_ARGS[@]}"
  "${SAFE_ARGS[@]}"
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
  echo "stdout_log_snapshot=$STDOUT_LOG_TS"
  if [[ ${#RESUME_ARGS[@]} -gt 0 ]]; then
    echo "resume_from=$LATEST_CKPT"
  else
    echo "resume_from="
  fi
} > "$STATUS_FILE"

nohup "$RUNNER_FILE" > "$STDOUT_LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PID_FILE"
cp "$STDOUT_LOG" "$STDOUT_LOG_TS" 2>/dev/null || true

{
  echo "pid=$PID"
  echo "runner_file=$RUNNER_FILE"
  echo "launch_command=$(cat "$CMD_FILE")"
} >> "$STATUS_FILE"

echo "Started training in background."
echo "PID: $PID"
echo "Config: $CONFIG_PATH"
echo "Checkpoint dir: $SAVE_DIR"
echo "Log dir: $LOG_DIR"
echo "Stdout log: $STDOUT_LOG"
