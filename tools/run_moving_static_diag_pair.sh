#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk2/bywang/DOA-net"
PYTHON="/home/bywang/miniconda3/envs/doa/bin/python"
cd "$ROOT"

"$PYTHON" tools/diagnostics/analyze_moving_static_subset.py \
  --checkpoint outputs/checkpoints_moving_hybridbrir_gate2_50h_v1_v7_dualcue_seq/best.pth \
  --config configs/train_moving_hybridbrir_gate2_50h_v1_v7_dualcue_seq.yaml \
  --output_dir outputs/analysis_moving_hybridbrir_gate2_50h_v1_v7_dualcue_seq_staticdiag \
  --num_workers 0

"$PYTHON" tools/diagnostics/analyze_moving_static_subset.py \
  --checkpoint outputs/checkpoints_moving_hybridbrir_gate2_50h_v1_v7_dualcue_seq_preenc_static_gate2/best.pth \
  --config configs/train_moving_hybridbrir_gate2_50h_v1_v7_dualcue_seq_preenc_static_gate2.yaml \
  --output_dir outputs/analysis_moving_hybridbrir_gate2_50h_v1_v7_dualcue_seq_preenc_static_gate2_staticdiag \
  --num_workers 0
