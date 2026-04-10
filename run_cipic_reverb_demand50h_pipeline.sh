#!/usr/bin/env bash
set -euo pipefail

cd /disk2/bywang/DOA-net

mkdir -p outputs/logs_librispeech_subject003_cipic_reverb_demand50h

# 1) 在已完成的50h clean（CIPIC空间化）基础上，加房间混响+DEMAND噪声。
#    仅使用 reverb_plus_noise 模式；SNR 在 [-10, 10] dB 随机采样。
/home/bywang/miniconda3/envs/doa/bin/python prepare_demand_mixed_data.py \
  --clean_root /disk2/bywang/DOA-net/data/librispeech_cipic_subject003_50h_clean \
  --demand_root /disk2/bywang/data/demand \
  --output_root /disk2/bywang/DOA-net/data/librispeech_cipic_subject003_reverb_demand50h \
  --scenes OOFFICE PCAFETER TMETRO TBUS SPSQUARE NPARK \
  --snr_min_db -10 \
  --snr_max_db 10 \
  --rt60_min 0.2 \
  --rt60_max 0.8 \
  --room_profiles small medium large \
  --clean_prob 0.0 \
  --reverb_only_prob 0.0 \
  --reverb_noise_prob 1.0 \
  --sample_rate 16000 \
  --seed 42 \
  --overwrite

# 2) 烟测训练 + 烟测评估（快速确认流程与指标可产出）。
/home/bywang/miniconda3/envs/doa/bin/python train.py \
  --config configs/train_librispeech_subject003_cipic_reverb_demand50h.yaml \
  --train.epochs 1 \
  --train.batch_size 16 \
  --train.num_workers 2

/home/bywang/miniconda3/envs/doa/bin/python evaluate.py \
  --checkpoint outputs/checkpoints_librispeech_subject003_cipic_reverb_demand50h/best.pth \
  --config configs/train_librispeech_subject003_cipic_reverb_demand50h.yaml \
  --output.log_dir outputs/logs_librispeech_subject003_cipic_reverb_demand50h_test_smoke

# 3) 完整训练（后台运行）。
nohup /home/bywang/miniconda3/envs/doa/bin/python -u train.py \
  --config configs/train_librispeech_subject003_cipic_reverb_demand50h.yaml \
  > outputs/logs_librispeech_subject003_cipic_reverb_demand50h/train_full.log 2>&1 &

echo "Pipeline done (full training started in background)."
