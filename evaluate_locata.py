#!/usr/bin/env python3
"""Zero-shot evaluation of static KEMAR classifiers on LOCATA Task 1."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.locata_dataset import LocataTask1Dataset
from models import build_model
from utils.angle import angular_error, bins_to_angles
from utils.checkpoint import load_checkpoint
from utils.config import load_config


def _setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("evaluate_locata")


def _load_model_state_compat(model: torch.nn.Module, state_dict: Dict, logger: logging.Logger) -> None:
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError as error:
        message = str(error)
        needs_temporal_rename = (
            "temporal_head.temporal_encoder" in message
            and ("temporal_head.gru" in message or "temporal_head.lstm" in message)
        )
        if not needs_temporal_rename:
            raise

    renamed = {}
    for key, value in state_dict.items():
        if key.startswith("temporal_head.gru."):
            key = key.replace("temporal_head.gru.", "temporal_head.temporal_encoder.", 1)
        elif key.startswith("temporal_head.lstm."):
            key = key.replace("temporal_head.lstm.", "temporal_head.temporal_encoder.", 1)
        renamed[key] = value
    logger.info("Detected historical temporal-head names; applying checkpoint compatibility mapping.")
    model.load_state_dict(renamed)


def _fold_front_back(angles_deg: np.ndarray) -> np.ndarray:
    return np.degrees(np.arcsin(np.sin(np.radians(angles_deg))))


def _mirror_pair_groups(
    num_classes: int,
    azimuth_range: tuple[float, float],
) -> tuple[np.ndarray, List[np.ndarray]]:
    """Return folded lateral angles and their front/back class groups."""
    centers = bins_to_angles(
        np.arange(num_classes, dtype=np.int64),
        num_classes,
        azimuth_range,
    )
    folded = np.round(_fold_front_back(centers), decimals=6)
    lateral_angles = np.unique(folded)
    groups = [np.flatnonzero(folded == angle) for angle in lateral_angles]
    return lateral_angles, groups


def _decode_mirror_pairs(
    probabilities: np.ndarray,
    num_classes: int,
    azimuth_range: tuple[float, float],
    front_back_predictions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode lateral mirror pairs, optionally choosing the half-plane externally.

    Without ``front_back_predictions``, the higher-probability member of the
    winning mirror pair is selected. With it, 0 selects the front member and 1
    selects the back member. At +/-90 degrees only one member exists.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != num_classes:
        raise ValueError(
            f"Expected probabilities [N, {num_classes}], got {probabilities.shape}"
        )
    if front_back_predictions is not None:
        front_back_predictions = np.asarray(front_back_predictions, dtype=np.int64)
        if front_back_predictions.shape != (probabilities.shape[0],):
            raise ValueError("front_back_predictions must have shape [N]")

    centers = bins_to_angles(
        np.arange(num_classes, dtype=np.int64),
        num_classes,
        azimuth_range,
    )
    _, groups = _mirror_pair_groups(num_classes, azimuth_range)
    pair_scores = np.stack(
        [probabilities[:, group].sum(axis=1) for group in groups],
        axis=1,
    )
    winning_groups = np.argmax(pair_scores, axis=1)

    pred_bins = np.empty(probabilities.shape[0], dtype=np.int64)
    for row_index, group_index in enumerate(winning_groups):
        group = groups[int(group_index)]
        if group.size == 1 or front_back_predictions is None:
            local_index = np.argmax(probabilities[row_index, group])
            pred_bins[row_index] = int(group[local_index])
            continue

        desired_back = bool(front_back_predictions[row_index])
        group_is_back = np.abs(centers[group]) > 90.0
        candidates = group[group_is_back == desired_back]
        if candidates.size == 0:
            candidates = group
        local_index = np.argmax(probabilities[row_index, candidates])
        pred_bins[row_index] = int(candidates[local_index])

    return pred_bins, centers[pred_bins]


def _metric_summary(
    errors: np.ndarray,
    pred_degrees: np.ndarray | None = None,
    true_degrees: np.ndarray | None = None,
) -> Dict[str, float]:
    if errors.size == 0:
        raise ValueError("Cannot summarize an empty error array")
    within_30 = errors <= 30.0
    metrics = {
        "mae_deg": float(np.mean(errors)),
        "median_ae_deg": float(np.median(errors)),
        "std_ae_deg": float(np.std(errors)),
        "acc_at_5deg": float(np.mean(errors <= 5.0)),
        "acc_at_10deg": float(np.mean(errors <= 10.0)),
        "acc_at_20deg": float(np.mean(errors <= 20.0)),
        "coverage_at_30deg": float(np.mean(within_30)),
        "gross_error_rate_gt_30deg": float(np.mean(errors > 30.0)),
        "conditional_mae_at_30deg": (
            float(np.mean(errors[within_30])) if np.any(within_30) else float("nan")
        ),
        "max_ae_deg": float(np.max(errors)),
    }
    if pred_degrees is not None and true_degrees is not None:
        pred_degrees = np.asarray(pred_degrees, dtype=np.float64)
        true_degrees = np.asarray(true_degrees, dtype=np.float64)
        pred_back = np.abs(pred_degrees) > 90.0
        true_back = np.abs(true_degrees) > 90.0
        lateral_errors = np.abs(
            _fold_front_back(pred_degrees) - _fold_front_back(true_degrees)
        )
        metrics["front_back_error_rate"] = float(np.mean(pred_back != true_back))
        metrics["folded_lateral_mae_deg"] = float(np.mean(lateral_errors))
        metrics["folded_lateral_median_ae_deg"] = float(np.median(lateral_errors))
    return metrics


def _recording_summary(
    rows: List[Dict],
    pred_column: str = "pred_azimuth_deg",
    error_column: str = "angular_error_deg",
) -> tuple[List[Dict], Dict[str, float]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["recording_id"])].append(float(row[error_column]))

    recording_rows = []
    for recording_id in sorted(grouped, key=lambda value: int(value.replace("recording", ""))):
        group_rows = [row for row in rows if row["recording_id"] == recording_id]
        errors = np.asarray(grouped[recording_id], dtype=np.float64)
        pred_degrees = np.asarray(
            [float(row[pred_column]) for row in group_rows], dtype=np.float64
        )
        true_degrees = np.asarray(
            [float(row["true_azimuth_deg"]) for row in group_rows], dtype=np.float64
        )
        metrics = _metric_summary(errors, pred_degrees, true_degrees)
        recording_rows.append(
            {
                "recording_id": recording_id,
                "num_windows": int(errors.size),
                **metrics,
            }
        )

    macro_keys = [
        "mae_deg",
        "median_ae_deg",
        "acc_at_5deg",
        "acc_at_10deg",
        "acc_at_20deg",
        "coverage_at_30deg",
        "gross_error_rate_gt_30deg",
        "conditional_mae_at_30deg",
        "front_back_error_rate",
        "folded_lateral_mae_deg",
        "folded_lateral_median_ae_deg",
    ]
    macro = {
        f"recording_macro_{key}": float(
            np.nanmean([float(row[key]) for row in recording_rows])
        )
        for key in macro_keys
    }
    return recording_rows, macro


def _write_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--locata-root", default="/disk2/bywang/data/zenodo_data")
    parser.add_argument("--split", choices=("dev", "eval"), default="eval")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    parser.add_argument("--min-vad-ratio", type=float, default=0.5)
    parser.add_argument("--left-channel", type=int, default=1, help="LOCATA one-based channel")
    parser.add_argument("--right-channel", type=int, default=3, help="LOCATA one-based channel")
    parser.add_argument("--no-peak-normalize", action="store_true")
    parser.add_argument(
        "--feature-cache-dir",
        default="outputs/locata_task1_dummy_eval/feature_cache",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    logger = _setup_logger()
    cfg = load_config(args.config, [])
    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    dataset = LocataTask1Dataset(
        root_dir=args.locata_root,
        split=args.split,
        sample_rate=int(cfg.dataset.sample_rate),
        segment_seconds=args.segment_seconds,
        hop_seconds=args.hop_seconds,
        min_vad_ratio=args.min_vad_ratio,
        channels=(args.left_channel - 1, args.right_channel - 1),
        peak_normalize=not args.no_peak_normalize,
        n_fft=int(cfg.feature.n_fft),
        hop_length=int(cfg.feature.hop_length),
        win_length=int(cfg.feature.win_length),
        window=str(cfg.feature.window),
        num_classes=int(cfg.model.num_classes),
        azimuth_range=tuple(cfg.model.azimuth_range),
        feature_cache_dir=args.feature_cache_dir,
    )
    logger.info(
        "LOCATA protocol: %d recordings, %d windows, channels=%s, VAD>=%.2f",
        len(dataset.recordings),
        len(dataset),
        dataset.channels,
        dataset.min_vad_ratio,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_model(cfg).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=str(device))
    _load_model_state_compat(model, checkpoint["model"], logger)
    model.eval()

    prediction_rows: List[Dict] = []
    output_batches: List[Dict[str, np.ndarray]] = []
    with torch.no_grad():
        for batch in loader:
            recording_ids = list(batch["recording_id"])
            starts = batch["start_sec"].numpy()
            vad_ratios = batch["vad_ratio"].numpy()
            true_degrees = batch["azimuth_deg"].numpy()
            elevations = batch["elevation_deg"].numpy()
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            model_output = model(batch)
            logits_tensor = model_output["logits"].float().cpu()
            logits = logits_tensor.numpy()
            probabilities = F.softmax(logits_tensor, dim=-1).numpy()
            pred_bins = np.argmax(logits, axis=-1)
            pred_degrees = bins_to_angles(
                pred_bins,
                int(cfg.model.num_classes),
                tuple(cfg.model.azimuth_range),
            )
            errors = angular_error(pred_degrees, true_degrees)

            centers = bins_to_angles(
                np.arange(int(cfg.model.num_classes), dtype=np.int64),
                int(cfg.model.num_classes),
                tuple(cfg.model.azimuth_range),
            )
            class_back_probability = probabilities[:, np.abs(centers) > 90.0].sum(axis=1)
            class_fb_predictions = (class_back_probability > 0.5).astype(np.int64)
            pair_local_bins, pair_local_degrees = _decode_mirror_pairs(
                probabilities,
                int(cfg.model.num_classes),
                tuple(cfg.model.azimuth_range),
            )
            pair_class_bins, pair_class_degrees = _decode_mirror_pairs(
                probabilities,
                int(cfg.model.num_classes),
                tuple(cfg.model.azimuth_range),
                class_fb_predictions,
            )

            aux_probabilities = None
            aux_fb_predictions = None
            pair_aux_bins = None
            pair_aux_degrees = None
            if "front_back_logits" in model_output:
                aux_probabilities = F.softmax(
                    model_output["front_back_logits"].float().cpu(), dim=-1
                ).numpy()
                aux_fb_predictions = np.argmax(aux_probabilities, axis=-1)
                pair_aux_bins, pair_aux_degrees = _decode_mirror_pairs(
                    probabilities,
                    int(cfg.model.num_classes),
                    tuple(cfg.model.azimuth_range),
                    aux_fb_predictions,
                )

            pair_local_errors = angular_error(pair_local_degrees, true_degrees)
            pair_class_errors = angular_error(pair_class_degrees, true_degrees)
            pair_aux_errors = (
                angular_error(pair_aux_degrees, true_degrees)
                if pair_aux_degrees is not None
                else None
            )
            for index, recording_id in enumerate(recording_ids):
                row = {
                        "recording_id": recording_id,
                        "start_sec": float(starts[index]),
                        "vad_ratio": float(vad_ratios[index]),
                        "true_azimuth_deg": float(true_degrees[index]),
                        "true_elevation_deg": float(elevations[index]),
                        "pred_bin": int(pred_bins[index]),
                        "pred_azimuth_deg": float(pred_degrees[index]),
                        "angular_error_deg": float(errors[index]),
                        "class_back_probability": float(class_back_probability[index]),
                        "class_front_back_pred": int(class_fb_predictions[index]),
                        "pair_local_pred_bin": int(pair_local_bins[index]),
                        "pair_local_pred_azimuth_deg": float(pair_local_degrees[index]),
                        "pair_local_angular_error_deg": float(pair_local_errors[index]),
                        "pair_class_fb_pred_bin": int(pair_class_bins[index]),
                        "pair_class_fb_pred_azimuth_deg": float(pair_class_degrees[index]),
                        "pair_class_fb_angular_error_deg": float(pair_class_errors[index]),
                    }
                if aux_fb_predictions is not None:
                    row.update(
                        {
                            "aux_back_probability": float(aux_probabilities[index, 1]),
                            "aux_front_back_pred": int(aux_fb_predictions[index]),
                            "pair_aux_pred_bin": int(pair_aux_bins[index]),
                            "pair_aux_pred_azimuth_deg": float(pair_aux_degrees[index]),
                            "pair_aux_angular_error_deg": float(pair_aux_errors[index]),
                        }
                    )
                prediction_rows.append(row)

            output_batch = {
                "true_degrees": np.asarray(true_degrees),
                "class_fb_predictions": class_fb_predictions,
            }
            if aux_fb_predictions is not None:
                output_batch["aux_fb_predictions"] = aux_fb_predictions
            output_batches.append(output_batch)

    errors = np.asarray(
        [float(row["angular_error_deg"]) for row in prediction_rows], dtype=np.float64
    )
    all_pred_degrees = np.asarray(
        [float(row["pred_azimuth_deg"]) for row in prediction_rows], dtype=np.float64
    )
    all_true_degrees = np.asarray(
        [float(row["true_azimuth_deg"]) for row in prediction_rows], dtype=np.float64
    )
    window_metrics = _metric_summary(errors, all_pred_degrees, all_true_degrees)
    recording_rows, recording_macro = _recording_summary(prediction_rows)

    decoder_columns = {
        "argmax": ("pred_azimuth_deg", "angular_error_deg"),
        "pair_local": (
            "pair_local_pred_azimuth_deg",
            "pair_local_angular_error_deg",
        ),
        "pair_class_fb": (
            "pair_class_fb_pred_azimuth_deg",
            "pair_class_fb_angular_error_deg",
        ),
    }
    if prediction_rows and "pair_aux_pred_azimuth_deg" in prediction_rows[0]:
        decoder_columns["pair_aux"] = (
            "pair_aux_pred_azimuth_deg",
            "pair_aux_angular_error_deg",
        )

    decoder_results: Dict[str, Dict] = {}
    decoder_recording_rows: Dict[str, List[Dict]] = {}
    for decoder_name, (pred_column, error_column) in decoder_columns.items():
        pred_values = np.asarray(
            [float(row[pred_column]) for row in prediction_rows], dtype=np.float64
        )
        error_values = np.asarray(
            [float(row[error_column]) for row in prediction_rows], dtype=np.float64
        )
        decoder_window = _metric_summary(error_values, pred_values, all_true_degrees)
        per_recording, decoder_macro = _recording_summary(
            prediction_rows,
            pred_column=pred_column,
            error_column=error_column,
        )
        decoder_results[decoder_name] = {
            "window_micro": decoder_window,
            "recording_macro": decoder_macro,
        }
        decoder_recording_rows[decoder_name] = per_recording

    true_fb = (np.abs(all_true_degrees) > 90.0).astype(np.int64)
    class_fb = np.concatenate(
        [batch_output["class_fb_predictions"] for batch_output in output_batches]
    )
    front_back_heads = {
        "class_probability_mass": {
            "accuracy": float(np.mean(class_fb == true_fb)),
            "num_correct": int(np.sum(class_fb == true_fb)),
            "num_windows": int(true_fb.size),
        }
    }
    if output_batches and "aux_fb_predictions" in output_batches[0]:
        aux_fb = np.concatenate(
            [batch_output["aux_fb_predictions"] for batch_output in output_batches]
        )
        front_back_heads["auxiliary_head"] = {
            "accuracy": float(np.mean(aux_fb == true_fb)),
            "num_correct": int(np.sum(aux_fb == true_fb)),
            "num_windows": int(true_fb.size),
        }
    result = {
        "verification_status": "executed",
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_mae": checkpoint.get("best_mae"),
        "model_type": str(cfg.model.type),
        "decode": (
            "argmax plus untuned mirror-pair decoders; pair score is the summed "
            "probability of front/back mirror classes"
        ),
        "protocol": dataset.protocol_summary(),
        "window_micro": window_metrics,
        "recording_macro": recording_macro,
        "front_back_heads": front_back_heads,
        "decoders": decoder_results,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "predictions.csv", prediction_rows)
    _write_csv(output_dir / "per_recording.csv", recording_rows)
    for decoder_name, rows in decoder_recording_rows.items():
        _write_csv(output_dir / f"per_recording_{decoder_name}.csv", rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(result), handle, ensure_ascii=False, indent=2)

    logger.info("Window-micro MAE: %.3f deg", window_metrics["mae_deg"])
    logger.info("Window-micro Median AE: %.3f deg", window_metrics["median_ae_deg"])
    logger.info("Window-micro Acc@5/10: %.2f%% / %.2f%%",
                100.0 * window_metrics["acc_at_5deg"],
                100.0 * window_metrics["acc_at_10deg"])
    logger.info("Recording-macro MAE: %.3f deg", recording_macro["recording_macro_mae_deg"])
    for head_name, head_metrics in front_back_heads.items():
        logger.info(
            "Front/back %s accuracy: %.2f%% (%d/%d)",
            head_name,
            100.0 * head_metrics["accuracy"],
            head_metrics["num_correct"],
            head_metrics["num_windows"],
        )
    for decoder_name, decoder_payload in decoder_results.items():
        metrics = decoder_payload["window_micro"]
        logger.info(
            "Decoder %-13s MAE=%.3f deg Acc@5/10=%.2f%%/%.2f%% FBErr=%.2f%%",
            decoder_name,
            metrics["mae_deg"],
            100.0 * metrics["acc_at_5deg"],
            100.0 * metrics["acc_at_10deg"],
            100.0 * metrics["front_back_error_rate"],
        )
    logger.info("Saved LOCATA results to %s", output_dir)


if __name__ == "__main__":
    main()
