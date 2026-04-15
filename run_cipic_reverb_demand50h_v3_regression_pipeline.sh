#!/usr/bin/env bash
set -euo pipefail

cd /disk2/bywang/DOA-net

mkdir -p outputs/logs_librispeech_subject003_cipic_reverb_demand50h_v3_regression

# 1) 烟测训练：快速验证模型改动链路可跑通。
/home/bywang/miniconda3/envs/doa/bin/python train.py \
  --config configs/train_librispeech_subject003_cipic_reverb_demand50h_v3_regression.yaml \
  --train.epochs 1 \
  --train.batch_size 16 \
  --train.num_workers 2

# 2) 烟测评估。
/home/bywang/miniconda3/envs/doa/bin/python evaluate.py \
  --checkpoint outputs/checkpoints_librispeech_subject003_cipic_reverb_demand50h_v3_regression/best.pth \
  --config configs/train_librispeech_subject003_cipic_reverb_demand50h_v3_regression.yaml \
  --output.log_dir outputs/logs_librispeech_subject003_cipic_reverb_demand50h_v3_regression_test_smoke

# 3) 完整训练（后台运行）。
nohup /home/bywang/miniconda3/envs/doa/bin/python -u train.py \
  --config configs/train_librispeech_subject003_cipic_reverb_demand50h_v3_regression.yaml \
  > outputs/logs_librispeech_subject003_cipic_reverb_demand50h_v3_regression/train_full.log 2>&1 &

echo "Pipeline done (full training started in background)."
