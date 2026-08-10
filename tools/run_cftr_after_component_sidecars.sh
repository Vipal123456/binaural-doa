#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/disk2/bywang/DOA-net"
DATA_ROOT="$ROOT_DIR/data/librispeech_cipic_roomsim25_directional_dns_v4"
LOG_DIR="$ROOT_DIR/outputs/cftr_cpsd_oracle_val"
LOG_FILE="$LOG_DIR/training_launcher.log"

mkdir -p "$LOG_DIR"
exec >>"$LOG_FILE" 2>&1
cd "$ROOT_DIR"

count_files() {
  find "$1" -maxdepth 1 -type f -name '*.wav' | wc -l
}

while true; do
  train_target=$(count_files "$DATA_ROOT/train/components/target")
  train_interferer=$(count_files "$DATA_ROOT/train/components/interferer")
  val_target=$(count_files "$DATA_ROOT/val/components/target")
  val_interferer=$(count_files "$DATA_ROOT/val/components/interferer")
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] train=$train_target/$train_interferer val=$val_target/$val_interferer"
  if [[ "$train_target" -eq 120000 && "$train_interferer" -eq 120000 \
        && "$val_target" -eq 12000 && "$val_interferer" -eq 12000 ]]; then
    break
  fi
  sleep 60
done

DOA_CUDA_VISIBLE_DEVICES=2 ./tools/run_training_tmux.sh \
  doa_cftr_b1_targetrw_g2 \
  configs/train_cipic_roomsim25_directional_dns_v4_v7_localtf32_targetrw_cpsd5_mask_bestmae_seed42.yaml

DOA_CUDA_VISIBLE_DEVICES=3 ./tools/run_training_tmux.sh \
  doa_cftr_b3_maskcov_g3 \
  configs/train_cipic_roomsim25_directional_dns_v4_v7_localtf32_cftr_cpsd5_maskcov_bestmae_seed42.yaml

./tools/run_training_after_tmux.sh \
  doa_cftr_b4_watcher \
  doa_cftr_b1_targetrw_g2 \
  doa_cftr_b4_maskonly_g2 \
  2 \
  configs/train_cipic_roomsim25_directional_dns_v4_v7_localtf32_cftr_cpsd5_maskonly_bestmae_seed42.yaml

echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] CFTR training jobs launched"
