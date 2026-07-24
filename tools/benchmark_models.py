#!/usr/bin/env python3
"""Benchmark all main models: params, FLOPs, inference speed."""

from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.config import Config
from models.binaural_doa_net import build_model


# ── Benchmark models and their configs ──────────────────────────────
# (label, config_path)
BENCHMARKS = [
    # ─── v7 mainlines ───
    ("v7 DualCue VR",        "configs/train_librispeech_multisubject_robust50h_v7_dualcue_vr_cf80_gru80_nocsl_fbaux_cohfix.yaml"),
    ("v7 LiteCue All",       "configs/train_librispeech_multisubject_robust50h_v7_litecueenc_concat_all_cf80_gru80_nocsl_fbaux_cohfix.yaml"),
    ("v7 cf80_cue24_gru80",  "configs/train_librispeech_multisubject_robust50h_v7_litecueenc_concat_all_cf80_cue24_gru80_nocsl_fbaux_cohfix.yaml"),
    ("v7 EncV2Balanced",     "configs/train_librispeech_multisubject_robust50h_v7_native_lite_encoderv2_balanced_nocsl_fbaux_cohfix.yaml"),
    ("v7 ContentOnly",       "configs/train_librispeech_multisubject_robust50h_v7_contentonly_cf80_gru80_nocsl_fbaux_cohfix.yaml"),
    ("v7 EarlyFusion",       "configs/train_librispeech_multisubject_robust50h_v7_earlyfusion_all_cf80_gru80_nocsl_fbaux_cohfix.yaml"),

    # ─── External baselines ───
    ("BiL GCC-PHAT CRN72",   "configs/train_librispeech_multisubject_robust50h_bilstyle_gccphat_crn72_nocsl.yaml"),
    ("FAViT ILD/IPD",        "configs/train_librispeech_multisubject_robust50h_favitstyle_ildipd_nocsl_fbaux_cohfix.yaml"),
    ("SDEL-DOA-Cls",         "configs/train_librispeech_multisubject_robust50h_sdel_doa_cls_baseline.yaml"),
    ("SDEL-DOA-Reg",         "configs/train_librispeech_multisubject_robust50h_sdel_doa_reg_baseline.yaml"),

    # ─── Legacy heavy ───
    ("v5 Heavy (gate+attn)", "configs/train_librispeech_multisubject_robust50h_v5_bias_gating_attnpool_csl_enhanced_fbaux_only_cohfix.yaml"),

    # ─── AMViT external ───
    ("AMViT mul",            "configs/train_librispeech_multisubject_static_hybridbrir_gate2_50h_v1_amvitstyle_mul.yaml"),
]


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def measure_flops_thop(model, dummy_input, device):
    """Try thop first, fall back to fvcore."""
    try:
        from thop import profile
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        return macs * 2  # thop returns MACs, we want FLOPs
    except ImportError:
        pass
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, dummy_input)
        return flops.total()
    except ImportError:
        pass
    return None


def benchmark_inference(model, dummy_input, device, warmup=30, repeats=100):
    """Measure inference latency and throughput."""
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Latency (batch_size=1)
    single_input = {k: v[:1].to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in dummy_input.items()}
    with torch.no_grad():
        if device == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = model(single_input)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    latency_ms = np.mean(times) * 1000

    # Throughput (batch_size=64 or as configured)
    bs = dummy_input[list(dummy_input.keys())[0]].shape[0]
    with torch.no_grad():
        if device == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(repeats // 2):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    throughput = bs / np.mean(times)  # samples/sec

    return {"latency_ms": round(latency_ms, 3), "throughput_sps": round(throughput, 1)}


def build_dummy_input(cfg, batch_size=64, device="cuda"):
    """Build a dummy input dict matching the model's expected input."""
    T = 200  # time frames
    F = 257  # freq bins
    dtype = torch.float32
    # Build rich dummy input covering all possible model requirements
    dummy = {
        "log_mag_L":   torch.randn(batch_size, T, F, dtype=dtype),
        "log_mag_R":   torch.randn(batch_size, T, F, dtype=dtype),
        "spec_real_L": torch.randn(batch_size, T, F, dtype=dtype),
        "spec_imag_L": torch.randn(batch_size, T, F, dtype=dtype),
        "spec_real_R": torch.randn(batch_size, T, F, dtype=dtype),
        "spec_imag_R": torch.randn(batch_size, T, F, dtype=dtype),
        "ipd":         torch.randn(batch_size, T, F, dtype=dtype),
        "ipd_sin":     torch.randn(batch_size, T, F, dtype=dtype),
        "ipd_cos":     torch.randn(batch_size, T, F, dtype=dtype),
        "ild":         torch.randn(batch_size, T, F, dtype=dtype),
        "coherence":   torch.randn(batch_size, T, F, dtype=dtype),
    }
    return {k: v.to(device) for k, v in dummy.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, Batch size: {args.batch_size}")
    print()

    results = []

    for label, config_path in BENCHMARKS:
        config_full = ROOT / config_path
        if not config_full.exists():
            print(f"[SKIP] {label} — config not found: {config_path}")
            continue

        print(f"[{label}]", end=" ", flush=True)

        try:
            cfg = Config.from_yaml(str(config_full))
            # Override device
            cfg.train.device = device

            model = build_model(cfg).to(device)
            dummy = build_dummy_input(cfg, batch_size=args.batch_size, device=device)

            # 1. Params
            params = count_parameters(model)

            # 2. FLOPs
            flops = measure_flops_thop(model, dummy, device)

            # 3. Inference speed
            speed = benchmark_inference(model, dummy, device)

            # 4. Model size (MB)
            import io
            buf = io.BytesIO()
            torch.save(model.state_dict(), buf)
            ckpt_mb = len(buf.getvalue()) / (1024 * 1024)

            row = {
                "model": label,
                "params": params["total"],
                "trainable": params["trainable"],
                "flops_g": round(flops / 1e9, 2) if flops else None,
                "latency_ms": speed["latency_ms"],
                "throughput": speed["throughput_sps"],
                "ckpt_mb": round(ckpt_mb, 1),
            }
            results.append(row)
            print(f"→ {params['total']:,} params, "
                  f"{row['flops_g']}G FLOPs" if row['flops_g'] else "FLOPs=N/A",
                  f", {speed['latency_ms']:.2f}ms, {speed['throughput_sps']:.0f} samples/s")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ── Print table ──────────────────────────────────────────────────
    print()
    print("=" * 120)
    print(f"{'Model':<28} {'Params':>10} {'FLOPs':>8} {'Latency':>8} {'Throughput':>12} {'Ckpt':>7}")
    print(f"{'':28} {'(M)':>10} {'(G)':>8} {'(ms)':>8} {'(samples/s)':>12} {'(MB)':>7}")
    print("-" * 120)
    for r in results:
        flops_str = f"{r['flops_g']:.2f}" if r['flops_g'] is not None else "N/A"
        print(f"{r['model']:<28} {r['params']/1e6:>8.2f}M {flops_str:>8} "
              f"{r['latency_ms']:>7.2f} {r['throughput']:>10.0f} {r['ckpt_mb']:>7.1f}")
    print("=" * 120)

    # Also save as CSV
    csv_path = ROOT / "outputs" / "model_benchmark.csv"
    with open(csv_path, "w") as f:
        f.write("model,params,flops_g,latency_ms,throughput_sps,ckpt_mb\n")
        for r in results:
            flops_str = f"{r['flops_g']:.2f}" if r['flops_g'] is not None else ""
            f.write(f"{r['model']},{r['params']},{flops_str},{r['latency_ms']:.3f},{r['throughput']:.0f},{r['ckpt_mb']:.1f}\n")
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
