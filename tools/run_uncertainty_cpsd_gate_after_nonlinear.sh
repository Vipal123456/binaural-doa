#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk2/bywang/DOA-net"
PYTHON="/home/bywang/miniconda3/envs/doa/bin/python"
GPU_UUID="GPU-31131b39-0a35-b861-c7fb-a57ebf88a337"
LOG="$ROOT/outputs/training_watchers/doa_uncertainty_after_nonlinear.log"

NONLINEAR_EVAL_SESSION="doa_e1_cuefactor_nonlinear_eval"
NONLINEAR_RESULT="$ROOT/outputs/logs_cipic_roomsim25_directional_dns_v4_v7_localtf32_cuefactor_cpsd5_nonlinear_bestmae_seed42/grouped_test_best_mae/overall.json"
B2_TEST_MAE="3.449074074074074"
REQUIRED_IMPROVEMENT="0.10"
B2_VAL_MAE="3.1433333333333335"
U1_VAL_TOLERANCE="0.10"

U1_SESSION="doa_u1_cuefactor_precision"
U1_CONFIG="configs/train_cipic_roomsim25_directional_dns_v4_v7_localtf32_cuefactor_cpsd5_precision_bestmae_seed42.yaml"
U1_LOG="$ROOT/outputs/logs_cipic_roomsim25_directional_dns_v4_v7_localtf32_cuefactor_cpsd5_precision_bestmae_seed42"
U1_CHECKPOINT="$ROOT/outputs/checkpoints_cipic_roomsim25_directional_dns_v4_v7_localtf32_cuefactor_cpsd5_precision_bestmae_seed42/best_mae.pth"

U2_SESSION="doa_u2_cuefactor_precision_calibrated"
U2_CONFIG="configs/train_cipic_roomsim25_directional_dns_v4_v7_localtf32_cuefactor_cpsd5_precision_calibrated_bestmae_seed42.yaml"
U2_LOG="$ROOT/outputs/logs_cipic_roomsim25_directional_dns_v4_v7_localtf32_cuefactor_cpsd5_precision_calibrated_bestmae_seed42"
U2_CHECKPOINT="$ROOT/outputs/checkpoints_cipic_roomsim25_directional_dns_v4_v7_localtf32_cuefactor_cpsd5_precision_calibrated_bestmae_seed42/best_mae.pth"

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
cd "$ROOT"

wait_for_session() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    sleep 60
  done
}

run_eval() {
  local status_file="$1"
  local config="$2"
  local checkpoint="$3"
  local standard_output="$4"
  local compound_output="$5"
  local eval_log="$6"
  env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_UUID" \
    "$ROOT/tools/run_cpsd_eval_after_training.sh" \
      "$status_file" "$config" "$checkpoint" \
      "$standard_output" "$compound_output" "$eval_log"
}

echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] waiting for nonlinear B2 evaluation"
wait_for_session "$NONLINEAR_EVAL_SESSION"
if [[ ! -f "$NONLINEAR_RESULT" ]]; then
  echo "nonlinear result missing; conditional queue stopped: $NONLINEAR_RESULT"
  exit 1
fi

if "$PYTHON" - "$NONLINEAR_RESULT" "$B2_TEST_MAE" "$REQUIRED_IMPROVEMENT" <<'PY'
import json
import sys

path, baseline, required = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
with open(path) as handle:
    mae = float(json.load(handle)["mae"])
threshold = baseline - required
print(f"nonlinear_mae={mae:.6f} required_threshold={threshold:.6f}")
raise SystemExit(0 if mae <= threshold else 1)
PY
then
  echo "nonlinear B2 met the 0.10-degree gate; U1/U2 skipped"
  exit 0
fi

echo "nonlinear B2 missed the gate; launching U1 precision weighting"
DOA_CUDA_VISIBLE_DEVICES="$GPU_UUID" \
  "$ROOT/tools/run_training_tmux.sh" "$U1_SESSION" "$U1_CONFIG"
wait_for_session "$U1_SESSION"
run_eval \
  "$U1_LOG/train_status_tmux.txt" \
  "$U1_LOG/resolved_config.yaml" \
  "$U1_CHECKPOINT" \
  "$U1_LOG/grouped_test_best_mae" \
  "$ROOT/outputs/compound_demand_v1_eval/u1_cuefactor_precision" \
  "$U1_LOG/auto_eval.log"

if [[ ! -f "$U1_CHECKPOINT" ]]; then
  echo "U1 checkpoint missing; U2 skipped"
  exit 1
fi

if ! "$PYTHON" - "$U1_CHECKPOINT" "$B2_VAL_MAE" "$U1_VAL_TOLERANCE" <<'PY'
import sys
import torch

path, baseline, tolerance = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
mae = float(checkpoint["best_mae"])
threshold = baseline + tolerance
print(f"u1_best_val_mae={mae:.6f} continuation_threshold={threshold:.6f}")
raise SystemExit(0 if mae <= threshold else 1)
PY
then
  echo "U1 validation degraded by more than 0.10 degree; U2 skipped"
  exit 0
fi

echo "U1 passed the validation gate; launching U2 aggregate calibration"
DOA_CUDA_VISIBLE_DEVICES="$GPU_UUID" \
  "$ROOT/tools/run_training_tmux.sh" "$U2_SESSION" "$U2_CONFIG"
wait_for_session "$U2_SESSION"
run_eval \
  "$U2_LOG/train_status_tmux.txt" \
  "$U2_LOG/resolved_config.yaml" \
  "$U2_CHECKPOINT" \
  "$U2_LOG/grouped_test_best_mae" \
  "$ROOT/outputs/compound_demand_v1_eval/u2_cuefactor_precision_calibrated" \
  "$U2_LOG/auto_eval.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] uncertainty CPSD queue complete"
