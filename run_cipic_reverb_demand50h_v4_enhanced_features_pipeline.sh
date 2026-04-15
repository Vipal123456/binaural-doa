#!/usr/bin/env bash
set -euo pipefail

cd /disk2/bywang/DOA-net

mkdir -p outputs/logs_librispeech_subject003_cipic_reverb_demand50h_v4_enhanced_features

# 1) 烟测训练
/home/bywang/miniconda3/envs/doa/bin/python train.py \
  --config configs/train_librispeech_subject003_cipic_reverb_demand50h_v4_enhanced_features.yaml \
  --train.epochs 1 \
  --train.batch_size 16 \
  --train.num_workers 2

# 2) 烟测评估
/home/bywang/miniconda3/envs/doa/bin/python evaluate.py \
  --checkpoint outputs/checkpoints_librispeech_subject003_cipic_reverb_demand50h_v4_enhanced_features/best.pth \
  --config configs/train_librispeech_subject003_cipic_reverb_demand50h_v4_enhanced_features.yaml \
  --output.log_dir outputs/logs_librispeech_subject003_cipic_reverb_demand50h_v4_enhanced_features_test_smoke

# 3) 完整训练（后台）
nohup /home/bywang/miniconda3/envs/doa/bin/python -u train.py \
  --config configs/train_librispeech_subject003_cipic_reverb_demand50h_v4_enhanced_features.yaml \
  > outputs/logs_librispeech_subject003_cipic_reverb_demand50h_v4_enhanced_features/train_full.log 2>&1 &

echo "Pipeline done (full training started in background)."
