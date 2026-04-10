#!/usr/bin/env python3
"""推理脚本 -- 对单个双耳 WAV 文件预测 DOA。

用法:
    python infer.py --checkpoint outputs/checkpoints/best.pth --wav test.wav
    python infer.py --checkpoint best.pth --wav long_file.wav --segment_seconds 2.0
"""

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from utils.config import load_config
from utils.angle import bin_to_angle
from dataset.feature_extractor import FeatureExtractor

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import librosa
except ImportError:
    librosa = None


def load_audio(wav_path: str, target_sr: int) -> np.ndarray:
    """加载立体声 WAV 文件，返回采样率为 *target_sr* 的 ``[2, N]`` float32 numpy 数组。"""
    if sf is not None:
        data, sr = sf.read(wav_path, dtype="float32")  # [N, C]
        if data.ndim == 1:
            data = np.stack([data, data], axis=-1)
        data = data.T  # [C, N]
        if sr != target_sr and librosa is not None:
            resampled = []
            for ch in range(data.shape[0]):
                resampled.append(librosa.resample(data[ch], orig_sr=sr, target_sr=target_sr))
            data = np.stack(resampled, axis=0)
    elif librosa is not None:
        data, _ = librosa.load(wav_path, sr=target_sr, mono=False)
        if data.ndim == 1:
            data = np.stack([data, data], axis=0)
    else:
        raise RuntimeError("请安装 soundfile 或 librosa 以加载音频文件。")

    return data[:2].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="双耳 DOA 推理")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--wav", type=str, required=True, help="立体声 WAV 文件路径")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--segment_seconds", type=float, default=None,
                        help="片段长度（默认值: 从配置文件读取）")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None,
                        help="可选的 JSON 结果输出路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seg_sec = args.segment_seconds or cfg.dataset.segment_seconds
    sr = cfg.dataset.sample_rate
    device = torch.device(args.device or (cfg.train.device if torch.cuda.is_available() else "cpu"))
    num_classes = cfg.model.num_classes
    azimuth_range = tuple(cfg.model.azimuth_range)

    # ---- 加载模型 ----
    from models.binaural_doa_net import build_model
    from utils.checkpoint import load_checkpoint

    model = build_model(cfg)
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # ---- 特征提取器 ----
    fe = FeatureExtractor(
        n_fft=cfg.feature.n_fft,
        hop_length=cfg.feature.hop_length,
        win_length=cfg.feature.win_length,
    )

    # ---- 加载音频 ----
    audio_np = load_audio(args.wav, sr)  # [2, N]
    total_samples = audio_np.shape[1]
    seg_samples = int(sr * seg_sec)

    # ---- 滑动窗口推理 ----
    results = []
    num_segments = max(1, total_samples // seg_samples)

    for i in range(num_segments):
        start = i * seg_samples
        end = start + seg_samples
        segment = audio_np[:, start:end]

        # 若片段长度不足则进行填充
        if segment.shape[1] < seg_samples:
            pad_len = seg_samples - segment.shape[1]
            segment = np.pad(segment, ((0, 0), (0, pad_len)))

        audio_tensor = torch.from_numpy(segment).float()
        feats = fe.extract(audio_tensor)

        # 构建批次字典 [1, T, F]
        batch = {
            k: v.unsqueeze(0).to(device) for k, v in feats.items()
        }

        with torch.no_grad():
            out = model(batch)

        logits = out["logits"][0]  # [C]
        probs = F.softmax(logits, dim=-1)
        pred_bin = probs.argmax().item()
        pred_angle = bin_to_angle(pred_bin, num_classes, azimuth_range)

        seg_result = {
            "segment": i,
            "start_sec": start / sr,
            "end_sec": min(end, total_samples) / sr,
            "predicted_bin": pred_bin,
            "predicted_azimuth_deg": round(pred_angle, 2),
            "confidence": round(probs[pred_bin].item(), 4),
        }
        results.append(seg_result)

        print(f"片段 {i}: 方位角 = {pred_angle:+.1f}° "
              f"(分箱 {pred_bin}, 置信度 {probs[pred_bin]:.3f})")

    # ---- 平均预测 ----
    avg_angle = np.mean([r["predicted_azimuth_deg"] for r in results])
    print(f"\n平均预测方位角: {avg_angle:+.1f}°")

    # ---- 保存 JSON ----
    if args.output_json:
        output = {
            "file": args.wav,
            "segments": results,
            "average_azimuth_deg": round(float(avg_angle), 2),
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"结果已保存至 {args.output_json}")


if __name__ == "__main__":
    main()
