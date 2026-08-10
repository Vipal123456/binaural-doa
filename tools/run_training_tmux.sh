#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/disk2/bywang/DOA-net"
PYTHON_BIN="/home/bywang/miniconda3/envs/doa/bin/python"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <session_name> <config_path> [extra train.py args...]"
  exit 1
fi

SESSION_NAME="$1"
CONFIG_PATH="$2"
shift 2 || true

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
  "$PYTHON_BIN" - "$CONFIG_PATH" "$@" <<'PY'
import sys

from utils.config import load_config

cfg = load_config("configs/default.yaml", ["--config", sys.argv[1], *sys.argv[2:]])
print(cfg.output.save_dir)
print(cfg.output.log_dir)
PY
)

SAVE_DIR="${CFG_VALUES[0]}"
LOG_DIR="${CFG_VALUES[1]}"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

STATUS_FILE="$LOG_DIR/train_status_tmux.txt"
CONSOLE_FILE="$LOG_DIR/train_console_tmux.log"
CMD_FILE="$LOG_DIR/launch_command_tmux.sh"
RUNNER_FILE="$LOG_DIR/run_in_tmux.sh"
LATEST_CKPT="$SAVE_DIR/latest.pth"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  exit 0
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
  SAFE_ARGS=(--train.num_workers 8)
fi

TRAIN_PREFIX=()
if [[ -n "${DOA_CUDA_VISIBLE_DEVICES:-}" ]]; then
  TRAIN_PREFIX=(
    env
    "CUDA_DEVICE_ORDER=PCI_BUS_ID"
    "CUDA_VISIBLE_DEVICES=${DOA_CUDA_VISIBLE_DEVICES}"
  )
fi

TRAIN_CMD=(
  "${TRAIN_PREFIX[@]}"
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
  echo "session_name=$(printf '%q' "$SESSION_NAME")"
  echo "start_time=\$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "config=$(printf '%q' "$CONFIG_PATH")"
  echo "save_dir=$(printf '%q' "$SAVE_DIR")"
  echo "log_dir=$(printf '%q' "$LOG_DIR")"
  echo "console_log=$(printf '%q' "$CONSOLE_FILE")"
  echo "resume_from=$(printf '%q' "${LATEST_CKPT}")"
  echo "launch_command=$(cat "$CMD_FILE")"
} > $(printf '%q' "$STATUS_FILE")
set +e
$(cat "$CMD_FILE") >> $(printf '%q' "$CONSOLE_FILE") 2>&1
EXIT_CODE=\$?
set -e
{
  echo "end_time=\$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "exit_code=\$EXIT_CODE"
} >> $(printf '%q' "$STATUS_FILE")
exit \$EXIT_CODE
EOF
chmod +x "$RUNNER_FILE"

tmux new-session -d -s "$SESSION_NAME" "$RUNNER_FILE"

echo "Started tmux training session."
echo "Session: $SESSION_NAME"
echo "Config: $CONFIG_PATH"
echo "Checkpoint dir: $SAVE_DIR"
echo "Log dir: $LOG_DIR"
echo "Status file: $STATUS_FILE"
echo "Attach: tmux attach -t $SESSION_NAME"
