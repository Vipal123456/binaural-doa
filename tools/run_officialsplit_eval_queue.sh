#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/bywang/miniconda3/envs/doa/bin/python}"
TEST_ROOT="${TEST_ROOT:-/disk2/bywang/DOA-net/data/kemar_sofamyroom_diffusefg_static_v1_test_officialsplit/test}"
OUT_ROOT="${OUT_ROOT:-/disk2/bywang/DOA-net/outputs/grouped_eval_runs_officialsplit}"
DEVICE="${DEVICE:-cuda:4}"
QUEUE="${1:-main}"

cd "${ROOT_DIR}"
mkdir -p "${OUT_ROOT}"

run_eval() {
  local name="$1"
  local cfg="$2"
  local ckpt="$3"
  local batch_size="$4"
  local out="${OUT_ROOT}/${name}"
  local log="${OUT_ROOT}/${name}.log"

  if [[ -f "${out}/overall.json" ]]; then
    echo "[skip] ${name}"
    return 0
  fi
  if [[ ! -f "${cfg}" ]]; then
    echo "[missing-config] ${name}: ${cfg}" >&2
    return 1
  fi
  if [[ ! -f "${ckpt}" ]]; then
    echo "[missing-checkpoint] ${name}: ${ckpt}" >&2
    return 1
  fi

  echo "[eval] ${name} device=${DEVICE} batch=${batch_size}"
  "${PYTHON_BIN}" tools/evaluate_kemar_grouped.py \
    --config "${cfg}" \
    --checkpoint "${ckpt}" \
    --test_root "${TEST_ROOT}" \
    --output_dir "${out}" \
    --batch_size "${batch_size}" \
    --num_workers 4 \
    --device "${DEVICE}" \
    --log_interval 20 \
    > "${log}" 2>&1
}

case "${QUEUE}" in
  main)
    run_eval main_seed42 configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_metricfix_seed42_g5.yaml outputs/checkpoints_kemar_v7_dualcue_liteenc_v1_diffusefg_metricfix_seed42_g5/best.pth 128
    run_eval main_seed43 configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_seed43_g6.yaml outputs/checkpoints_kemar_v7_dualcue_liteenc_v1_diffusefg_metricfix_seed43_g6/best.pth 128
    run_eval main_seed44 configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_seed44_g4.yaml outputs/checkpoints_kemar_v7_dualcue_liteenc_v1_diffusefg_metricfix_seed44_g4/best.pth 128
    run_eval norel_seed42 configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_norel_g4.yaml outputs/checkpoints_kemar_v7_dualcue_liteenc_v1_diffusefg_norel_metricfix_seed42/best.pth 128
    run_eval norel_seed43 configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_norel_seed43_g5.yaml outputs/checkpoints_kemar_v7_dualcue_liteenc_v1_diffusefg_norel_metricfix_seed43/best.pth 128
    run_eval norel_seed44 configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_norel_seed44_g5.yaml outputs/checkpoints_kemar_v7_dualcue_liteenc_v1_diffusefg_norel_metricfix_seed44/best.pth 128
    ;;
  baselines)
    run_eval sdel_seed43 configs/train_kemar_sdel_doa_cls_diffusefg_nofb_seed43_g5.yaml outputs/checkpoints_kemar_sdel_doa_cls_diffusefg_nofb_metricfix_seed43_g5/best.pth 128
    run_eval sdel_seed44 configs/train_kemar_sdel_doa_cls_diffusefg_nofb_seed44_g4.yaml outputs/checkpoints_kemar_sdel_doa_cls_diffusefg_nofb_metricfix_seed44_g4/best.pth 128
    run_eval sdel_seed45 configs/train_kemar_sdel_doa_cls_diffusefg_nofb_seed45_g6.yaml outputs/checkpoints_kemar_sdel_doa_cls_diffusefg_nofb_metricfix_seed45_g6/best.pth 128
    run_eval dprtf_seed42 configs/train_kemar_dprtf_doa_cls_diffusefg_g5.yaml outputs/checkpoints_kemar_dprtf_doa_cls_diffusefg_metricfix_g5/best.pth 128
    run_eval dprtf_seed43 configs/train_kemar_dprtf_doa_cls_diffusefg_seed43_g5.yaml outputs/checkpoints_kemar_dprtf_doa_cls_diffusefg_metricfix_seed43_g5/best.pth 128
    run_eval dprtf_seed44 configs/train_kemar_dprtf_doa_cls_diffusefg_seed44_g4.yaml outputs/checkpoints_kemar_dprtf_doa_cls_diffusefg_metricfix_seed44_g4/best.pth 128
    run_eval bil_seed42 configs/train_kemar_bilstyle_gccphat_crn72_diffusefg_g7.yaml outputs/checkpoints_kemar_bilstyle_gccphat_crn72_diffusefg_metricfix_g7/best.pth 128
    run_eval bil_seed43 configs/train_kemar_bilstyle_gccphat_crn72_diffusefg_seed43_g6.yaml outputs/checkpoints_kemar_bilstyle_gccphat_crn72_diffusefg_metricfix_seed43_g6/best.pth 128
    run_eval bil_seed44 configs/train_kemar_bilstyle_gccphat_crn72_diffusefg_seed44_g5.yaml outputs/checkpoints_kemar_bilstyle_gccphat_crn72_diffusefg_metricfix_seed44_g5/best.pth 128
    ;;
  ablations)
    run_eval nocontent_seed42 configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_nocontent_g5.yaml outputs/checkpoints_kemar_v7_dualcue_liteenc_v1_diffusefg_nocontent_metricfix_seed42/best.pth 128
    run_eval nocontent_seed43 configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_nocontent_seed43_g6.yaml outputs/checkpoints_kemar_v7_dualcue_liteenc_v1_diffusefg_nocontent_metricfix_seed43/best.pth 128
    run_eval nocontent_seed44 configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_nocontent_seed44_g6.yaml outputs/checkpoints_kemar_v7_dualcue_liteenc_v1_diffusefg_nocontent_metricfix_seed44/best.pth 128
    run_eval mergedcue_seed42 configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_mergedcue_g6.yaml outputs/checkpoints_kemar_v7_dualcue_liteenc_v1_diffusefg_mergedcue_metricfix_seed42/best.pth 128
    run_eval mergedcue_seed43 configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_mergedcue_seed43_g4.yaml outputs/checkpoints_kemar_v7_dualcue_liteenc_v1_diffusefg_mergedcue_metricfix_seed43/best.pth 128
    run_eval mergedcue_seed44 configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_mergedcue_seed44_g4.yaml outputs/checkpoints_kemar_v7_dualcue_liteenc_v1_diffusefg_mergedcue_metricfix_seed44/best.pth 128
    ;;
  fnssl)
    run_eval fnssl_seed42 configs/train_kemar_fnssl_diffusefg_g5_stable.yaml outputs/checkpoints_kemar_fnssl_diffusefg_metricfix_g5_stable/best.pth 8
    run_eval fnssl_seed43 configs/train_kemar_fnssl_diffusefg_seed43_g4.yaml outputs/checkpoints_kemar_fnssl_diffusefg_metricfix_seed43_g3/best.pth 8
    ;;
  *)
    echo "Usage: $0 {main|baselines|ablations|fnssl}" >&2
    exit 2
    ;;
esac
