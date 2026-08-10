#!/usr/bin/env python3
"""Evaluate static DOA models on Neidhardt measured BRIR test set."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.binaural_doa_net import build_model
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.logger import setup_logger
from dataset.feature_extractor import FeatureExtractor
from utils.angle import bins_to_angles


class NeidhardtTestDataset(Dataset):
    def __init__(self, root: Path, feature_extractor: FeatureExtractor):
        self.meta_dir = root / "metadata_dev"
        self.wav_dir = root / "binaural_dev"
        self.metas = sorted(self.meta_dir.glob("*.json"))
        self.fe = feature_extractor

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        m = json.loads(self.metas[idx].read_text())
        fid = m["file_id"]
        stereo, sr = sf.read(str(self.wav_dir / f"binaural{fid:06d}.wav"), dtype="float32")
        if stereo.ndim == 1:
            stereo = np.stack([stereo, stereo], axis=1)
        # FeatureExtractor expects [2, N] (channels first)
        stereo_2n = np.ascontiguousarray(stereo.T)
        feats = self.fe.extract(torch.from_numpy(stereo_2n).float())
        feats["doa_label"] = int(m["doa_class"])
        feats["azimuth_deg"] = float(m["azimuth_deg"])
        feats["snr_label"] = str(m.get("snr_label", "clean"))
        feats["listener_position"] = int(m.get("listener_position", 0))
        feats["sofa_file"] = str(m.get("sofa_file", ""))
        feats["speaker_orientation"] = str(m.get("speaker_orientation", "unknown"))
        return feats


def wrap_deg(a):
    a = np.asarray(a, dtype=np.float64)
    return ((a + 180) % 360) - 180


def circular_error(pred_deg, true_deg):
    diff = np.abs(wrap_deg(pred_deg - true_deg))
    return np.minimum(diff, 360 - diff)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--test_dir", type=str,
                        default="data/librispeech_neidhardt_measured_brir_test_v2/test_all")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, [])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    fe = FeatureExtractor(
        n_fft=cfg.feature.n_fft,
        hop_length=cfg.feature.hop_length,
        win_length=cfg.feature.win_length,
        window=cfg.feature.window,
    )

    ds = NeidhardtTestDataset(Path(args.test_dir), fe)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"Test set: {len(ds)} segments")

    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(args.checkpoint, map_location=str(device))
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Accumulators
    all_preds = []
    all_labels = []
    all_angles = []
    all_snrs = []
    all_positions = []
    all_orientations = []
    snr_groups: Dict[str, list] = defaultdict(list)
    pos_groups: Dict[int, list] = defaultdict(list)
    bin_groups: Dict[int, list] = defaultdict(list)

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(batch)
        logits = out["logits"].float().cpu().numpy()
        preds = logits.argmax(axis=-1)
        pred_angles = bins_to_angles(preds)

        true_labels = batch["doa_label"].cpu().numpy()
        true_angles = batch["azimuth_deg"].float().cpu().numpy()
        e = circular_error(pred_angles, true_angles)

        all_preds.append(preds)
        all_labels.append(true_labels)
        all_angles.append(e)
        all_snrs.extend(str(snr) for snr in batch["snr_label"])
        all_positions.extend(int(pos) for pos in batch["listener_position"])
        all_orientations.extend(str(value) for value in batch["speaker_orientation"])

        for i, snr in enumerate(batch["snr_label"]):
            snr_groups[str(snr)].append(e[i])
        for i, pos in enumerate(batch["listener_position"]):
            pos_groups[int(pos)].append(e[i])
        for i, tl in enumerate(true_labels):
            bin_groups[int(tl)].append(e[i])

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_errors = np.concatenate(all_angles)
    all_snrs = np.asarray(all_snrs)
    all_positions = np.asarray(all_positions)
    all_orientations = np.asarray(all_orientations)
    pred_angles = bins_to_angles(all_preds)
    true_angles = bins_to_angles(all_labels)

    coordinate_targets = {
        "as_is": true_angles,
        "target_plus_180": wrap_deg(true_angles + 180.0),
        "target_negated": wrap_deg(-true_angles),
        "target_negated_plus_180": wrap_deg(-true_angles + 180.0),
    }
    coordinate_diagnostics = {}
    for name, diagnostic_targets in coordinate_targets.items():
        diagnostic_errors = circular_error(pred_angles, diagnostic_targets)
        coordinate_diagnostics[name] = {
            "mae_deg": float(diagnostic_errors.mean()),
            "median_ae_deg": float(np.median(diagnostic_errors)),
            "acc_at_5deg": float((diagnostic_errors <= 5).mean()),
            "acc_at_10deg": float((diagnostic_errors <= 10).mean()),
        }

    # Overall metrics
    acc = float((all_preds == all_labels).mean())
    mae = float(all_errors.mean())
    acc5 = float((all_errors <= 5).mean())
    acc10 = float((all_errors <= 10).mean())
    pred_fb = np.abs(wrap_deg(pred_angles)) > 90
    true_fb = np.abs(wrap_deg(true_angles)) > 90
    fb = float((pred_fb != true_fb).mean())
    large = float((all_errors > 90).mean())
    opposite = float((all_errors > 150).mean())

    def summarize(mask):
        errors = all_errors[mask]
        correct = all_preds[mask] == all_labels[mask]
        return {
            "n": int(len(errors)),
            "accuracy": float(correct.mean()),
            "mae_deg": float(errors.mean()),
            "median_ae_deg": float(np.median(errors)),
            "acc_at_5deg": float((errors <= 5).mean()),
            "acc_at_10deg": float((errors <= 10).mean()),
            "large_err_rate": float((errors > 90).mean()),
            "opposite_err_rate": float((errors > 150).mean()),
        }

    print(f"\n{'='*60}")
    print(f"  Model: {args.checkpoint}")
    print(f"{'='*60}")
    print(f"  Overall:  Acc={acc:.2%}  MAE={mae:.2f} deg  Acc@5={acc5:.2%}  Acc@10={acc10:.2%}")
    print(f"            FB_err={fb:.2%}  Large_err={large:.2%}  Opp_err={opposite:.2%}")
    print(f"            median_AE={np.median(all_errors):.2f} deg  N={len(all_errors)}")
    print("\n  --- Fixed coordinate diagnostics (not primary metrics) ---")
    for name, values in coordinate_diagnostics.items():
        print(
            f"    {name:>24s}: MAE={values['mae_deg']:.2f} deg  "
            f"Acc@10={values['acc_at_10deg']:.2%}"
        )

    # Per-SNR
    print(f"\n  --- Per SNR ---")
    for snr in ["clean", "10", "5", "0", "-5", "-10"]:
        mask = all_snrs == snr
        snr_errors = all_errors[mask]
        snr_acc = all_preds[mask] == all_labels[mask]
        if len(snr_errors) == 0:
            print(f"    SNR={snr:>6s}: NO DATA")
            continue
        print(f"    SNR={snr:>6s}: Acc={snr_acc.mean():.2%}  MAE={snr_errors.mean():.2f} deg  Acc@5={(snr_errors<=5).mean():.2%}  Acc@10={(snr_errors<=10).mean():.2%}  N={len(snr_errors)}")

    # Per-position
    print(f"\n  --- Per Position ---")
    for pos in sorted(pos_groups):
        p = np.array(pos_groups[pos])
        print(f"    Pos {pos}: MAE={p.mean():.2f} deg  N={len(p)}")

    # Per-bin summary
    bin_maes = {}
    for b in sorted(bin_groups):
        bin_maes[b] = float(np.mean(bin_groups[b]))
    def region_mean(bins):
        values = [bin_maes[b] for b in bins if b in bin_maes]
        return float(np.mean(values)) if values else float("nan")

    front_mae = region_mean(range(27, 45))
    back_mae = region_mean(list(range(0, 9)) + list(range(63, 72)))
    side_mae = region_mean(list(range(9, 27)) + list(range(45, 63)))
    print(f"\n    Front (±45 deg): {front_mae:.2f} deg")
    print(f"    Back (±45 deg from 180): {back_mae:.2f} deg")
    print(f"    Side: {side_mae:.2f} deg")
    print(f"    Worst 3 bins: {sorted(bin_maes.items(), key=lambda x: -x[1])[:3]}")
    print(f"    Best 3 bins:  {sorted(bin_maes.items(), key=lambda x: x[1])[:3]}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "config": str(Path(args.config).resolve()),
            "test_dir": str(Path(args.test_dir).resolve()),
            "decode": "argmax class; angle = -180 + 5 * class",
            "coordinate_diagnostics_note": "Fixed transforms are diagnostic only and must not replace the preregistered primary mapping after observing test results.",
            "coordinate_diagnostics": coordinate_diagnostics,
            "overall": {
                **summarize(np.ones(len(all_errors), dtype=bool)),
                "front_back_err_rate": fb,
            },
            "by_snr": {
                snr: summarize(all_snrs == snr)
                for snr in ["clean", "10", "5", "0", "-5", "-10"]
            },
            "by_position": {
                str(pos): summarize(all_positions == pos)
                for pos in sorted(np.unique(all_positions))
            },
            "by_speaker_orientation": {
                orientation: summarize(all_orientations == orientation)
                for orientation in sorted(np.unique(all_orientations))
            },
            "per_bin_mae_deg": {str(key): value for key, value in bin_maes.items()},
        }
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"    Saved JSON: {output_path}")

    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
