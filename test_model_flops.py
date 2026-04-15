#!/usr/bin/env python3
"""计算各版本模型的参数量和 FLOPs"""

import torch
import torch.nn as nn
from thop import profile, clever_format
from utils.config import load_config
from models.binaural_doa_net import build_model

def analyze_version(version_name, config_path):
    """分析单个版本"""
    print(f"\n【版本：{version_name}】")
    print(f"配置：{config_path}")
    
    try:
        # 加载配置
        cfg = load_config(config_path, [])
        
        # 构建模型
        model = build_model(cfg)
        model.eval()
        
        # ---- 参数量统计 ----
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # ---- FLOPs 计算 ----
        # 输入：(batch=1, channels=4, frames=197, freq_bins=257)
        # 197 frames = 2.0s @ 16kHz with hop=160
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        dummy_input = torch.randn(1, 4, 197, 257).to(device)
        
        try:
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            flops_g = flops / 1e9
            flops_str = f"{flops_g:.2f}G"
        except Exception as e:
            print(f"  ⚠ FLOPs 计算出错: {e}")
            flops = None
            flops_str = "计算失败"
        
        # ---- 打印结果 ----
        print(f"  总参数量：{total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  可训练参数：{trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"  FLOPs（单样本）：{flops_str}")
        print(f"  模型大小(FP32)：{(total_params*4)/(1024**2):.2f} MB")
        
        return {
            'name': version_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'flops': flops,
            'config': cfg
        }
        
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        return None

def main():
    print("="*70)
    print("DOA-Net v2/v3/v4/v5 模型复杂度对比分析")
    print("="*70)
    
    versions = [
        ("v2", "configs/train_librispeech_subject003_cipic_reverb_demand50h_v2.yaml"),
        ("v3", "configs/train_librispeech_subject003_cipic_reverb_demand50h_v3_regression.yaml"),
        ("v4", "configs/train_librispeech_subject003_cipic_reverb_demand50h_v4_enhanced_features.yaml"),
        ("v5", "configs/train_librispeech_subject003_cipic_reverb_demand50h_v5_bias_gating_attnpool_csl.yaml"),
    ]
    
    results = []
    for version_name, config_path in versions:
        res = analyze_version(version_name, config_path)
        if res:
            results.append(res)
    
    # ---- 对比总结 ----
    if results:
        print("\n" + "="*70)
        print("版本对比总结表")
        print("="*70)
        
        print("\n| 版本 | 参数量(M) | FLOPs | 相对v2(参数) | 相对v2(FLOPs) |")
        print("|------|----------|-------|-------------|---------------|")
        
        v2_info = next((r for r in results if r['name'] == 'v2'), None)
        v2_params = v2_info['total_params'] if v2_info else None
        v2_flops = v2_info['flops'] if v2_info and v2_info['flops'] else None
        
        for res in results:
            version = res['name']
            params_m = res['total_params'] / 1e6
            flops_g = res['flops'] / 1e9 if res['flops'] else 0
            flops_str = f"{flops_g:.2f}G"
            
            if v2_params and version != 'v2':
                param_diff = (res['total_params'] - v2_params) / v2_params * 100
                param_str = f"+{param_diff:.1f}%" if param_diff > 0 else f"{param_diff:.1f}%"
            else:
                param_str = "baseline" if version == 'v2' else "-"
            
            if v2_flops and version != 'v2' and res['flops']:
                flops_diff = (res['flops'] - v2_flops) / v2_flops * 100
                flops_str_diff = f"+{flops_diff:.1f}%" if flops_diff > 0 else f"{flops_diff:.1f}%"
            else:
                flops_str_diff = "baseline" if version == 'v2' else "-"
            
            print(f"| {version} | {params_m:.2f} | {flops_str} | {param_str} | {flops_str_diff} |")
    
    # ---- GPU推理时间估计 ----
    print("\n" + "="*70)
    print("推理时间估计（单样本，batch_size=1）")
    print("="*70)
    
    gpu_specs = [
        ("V100", 100.0),
        ("A100", 312.0),
        ("RTX 3090", 35.6),
        ("RTX 4090", 82.6),
    ]
    
    for res in results:
        if not res['flops']:
            continue
        
        version = res['name']
        print(f"\n{version} (FLOPs: {res['flops']/1e9:.2f}G):")
        
        for gpu_name, peak_tflops in gpu_specs:
            inference_ms = (res['flops'] / 1e12) / peak_tflops * 1000
            throughput = 1000 / inference_ms
            print(f"  {gpu_name:12s}: {inference_ms:.2f} ms/sample, {throughput:7.1f} samples/s")
    
    print("\n" + "="*70)
    print("✓ 分析完成")
    print("="*70)

if __name__ == "__main__":
    main()
