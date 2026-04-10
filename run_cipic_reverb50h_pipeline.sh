#!/usr/bin/env bash
set -euo pipefail

cd /disk2/bywang/DOA-net

mkdir -p outputs/logs_librispeech_subject003_cipic_reverb50h

# 1) Generate ~50h clean spatialized dataset with CIPIC HRTF.
/home/bywang/miniconda3/envs/doa/bin/python synthesize_librispeech_cipic.py \
  --librispeech_root /disk2/bywang/data/LibriSpeech/train-clean-100 \
  --sofa_path /disk2/bywang/data/HRTF/subject_003.sofa \
  --output_root /disk2/bywang/DOA-net/data/librispeech_cipic_subject003_50h_clean \
  --num_recordings 18000 \
  --sample_rate 16000 \
  --duration_sec 10 \
  --seed 42

# 2) Add room reverb only (no additive noise).
/home/bywang/miniconda3/envs/doa/bin/python prepare_demand_mixed_data.py \
  --clean_root /disk2/bywang/DOA-net/data/librispeech_cipic_subject003_50h_clean \
  --demand_root /disk2/bywang/data/demand \
  --output_root /disk2/bywang/DOA-net/data/librispeech_cipic_subject003_reverb50h \
  --scenes OOFFICE PCAFETER TMETRO TBUS SPSQUARE NPARK \
  --rt60_min 0.2 \
  --rt60_max 0.8 \
  --room_profiles small medium large \
  --clean_prob 0.0 \
  --reverb_only_prob 1.0 \
  --reverb_noise_prob 0.0 \
  --sample_rate 16000 \
  --seed 42 \
  --overwrite

# 3) Smoke train + smoke test.
/home/bywang/miniconda3/envs/doa/bin/python train.py \
  --config configs/train_librispeech_subject003_cipic_reverb50h.yaml \
  --train.epochs 1 \
  --train.batch_size 16 \
  --train.num_workers 2

/home/bywang/miniconda3/envs/doa/bin/python evaluate.py \
  --checkpoint outputs/checkpoints_librispeech_subject003_cipic_reverb50h/best.pth \
  --config configs/train_librispeech_subject003_cipic_reverb50h.yaml \
  --output.log_dir outputs/logs_librispeech_subject003_cipic_reverb50h_test_smoke

# 4) Full train.
nohup /home/bywang/miniconda3/envs/doa/bin/python -u train.py \
  --config configs/train_librispeech_subject003_cipic_reverb50h.yaml \
  > outputs/logs_librispeech_subject003_cipic_reverb50h/train_full.log 2>&1 &

echo "Pipeline done (full training started in background)."
