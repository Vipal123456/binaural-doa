#!/usr/bin/env bash
set -euo pipefail

cd /disk2/bywang/DOA-net

# Wait until pyroom dataset generation reaches target size.
while true; do
  n=$(find data/librispeech_subject003_pyroom_reverb50h/binaural_dev -type f 2>/dev/null | wc -l)
  if [[ "$n" -ge 18000 ]]; then
    break
  fi
  sleep 60
done

/home/bywang/miniconda3/envs/doa/bin/python train.py \
  --config configs/train_librispeech_subject003_pyroom_reverb50h.yaml \
  --train.epochs 1 \
  --train.batch_size 16 \
  --train.num_workers 2

/home/bywang/miniconda3/envs/doa/bin/python evaluate.py \
  --checkpoint outputs/checkpoints_librispeech_subject003_pyroom_reverb50h/best.pth \
  --config configs/train_librispeech_subject003_pyroom_reverb50h.yaml \
  --output.log_dir outputs/logs_librispeech_subject003_pyroom_reverb50h_test_smoke

nohup /home/bywang/miniconda3/envs/doa/bin/python -u train.py \
  --config configs/train_librispeech_subject003_pyroom_reverb50h.yaml \
  > outputs/logs_librispeech_subject003_pyroom_reverb50h/train_full.log 2>&1 &
