#!/usr/bin/env python3
"""手工计算模型的 FLOPs"""

import torch
import torch.nn as nn
from utils.config import load_config
from models.binaural_doa_net import build_model

def calculate_flops_manual(model, input_shape):
    """手工计算主要层的 FLOPs"""
    
    total_flops = 0
    
    # 输入形状：(batch, channels, freq_bins, frames)
    # 我们的格式实际是：(batch, channels, frames, freq_bins)
    batch_size = input_shape[0]
    
    print("=== 主要模块 FLOPs 计算 ===\n")
    
    # 遍历所有模块
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Conv2d FLOPs = out_h * out_w * kernel_h * kernel_w * in_c * out_c
            out_c = module.out_channels
            in_c = module.in_channels
            kh, kw = module.kernel_size
            stride = module.stride[0]
            
            # 粗估输出大小（假设不改变输入大小）
            out_h, out_w = input_shape[2], input_shape[3]
            flops = batch_size * out_h * out_w * kh * kw * in_c * out_c
            total_flops += flops
            if 'encoder' in name or 'conv' in name.lower():
                print(f"{name}: {flops/1e6:.1f}M FLOPs")
        
        elif isinstance(module, nn.Linear):
            # Linear FLOPs = in_features * out_features * batch_size
            in_f = module.in_features
            out_f = module.out_features
            flops = batch_size * in_f * out_f
            total_flops += flops
            if 'head' in name or 'fc' in name:
                print(f"{name}: {flops/1e6:.1f}M FLOPs")
        
        elif isinstance(module, nn.GRU):
            # GRU FLOPs ≈ 3 * seq_len * batch_size * (hidden_size^2 + input_size * hidden_size)
            # 简化：seq_len * batch_size * hidden_size^2 * 9 (3 gates * 3 matrix ops)
            hidden_size = module.hidden_size
            input_size = module.input_size
            seq_len = 256  # 粗估序列长度
            gru_flops = seq_len * batch_size * (hidden_size ** 2 * 9 + input_size * hidden_size * 3)
            total_flops += gru_flops
            print(f"{name}: {gru_flops/1e6:.1f}M FLOPs")
        
        elif isinstance(module, nn.MultiheadAttention):
            # Attention FLOPs = 4 * seq_len^2 * d_model (Q@K, softmax, @V, output)
            # 简化估计
            d_model = module.embed_dim
            seq_len = 256  # 粗估
            attn_flops = batch_size * 4 * seq_len ** 2 * d_model
            total_flops += attn_flops
            print(f"{name}: {attn_flops/1e6:.1f}M FLOPs")
    
    return total_flops

def main():
    print("="*70)
    print("DOA-Net 模型参数量对比")
    print("="*70)
    
    versions = [
        ("v2", "configs/train_librispeech_subject003_cipic_reverb_demand50h_v2.yaml"),
        ("v3", "configs/train_librispeech_subject003_cipic_reverb_demand50h_v3_regression.yaml"),
        ("v4", "configs/train_librispeech_subject003_cipic_reverb_demand50h_v4_enhanced_features.yaml"),
        ("v5", "configs/train_librispeech_subject003_cipic_reverb_demand50h_v5_bias_gating_attnpool_csl.yaml"),
    ]
    
    results = []
    
    for version_name, config_path in versions:
        print(f"\n【版本：{version_name}】")
        
        try:
            cfg = load_config(config_path, [])
            model = build_model(cfg).eval()
            
            # 参数统计
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # 分层参数统计
            print("\n  分层参数统计：")
            layer_params = {}
            for name, module in model.named_modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    params = module.weight.numel()
                    if hasattr(module, 'bias') and module.bias is not None:
                        params += module.bias.numel()
                    
                    # 归类到顶层模块
                    top_level = name.split('.')[0] if '.' in name else name
                    if top_level not in layer_params:
                        layer_params[top_level] = 0
                    layer_params[top_level] += params
            
            # 排序输出前5个最大的模块
            sorted_layers = sorted(layer_params.items(), key=lambda x: x[1], reverse=True)
            for layer_name, layer_param in sorted_layers[:5]:
                pct = (layer_param / total_params) * 100
                print(f"    {layer_name:30s}: {layer_param:10,} ({pct:5.1f}%)")
            
            # 输出总参数
            print(f"\n  总参数量：{total_params:,} ({total_params/1e6:.2f}M)")
            print(f"  模型大小(FP32)：{(total_params*4)/(1024**2):.2f} MB")
            print(f"  模型大小(FP16)：{(total_params*2)/(1024**2):.2f} MB")
            
            results.append({
                'name': version_name,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'model_size_fp32_mb': (total_params*4)/(1024**2),
                'layer_params': layer_params
            })
            
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
    
    # 对比表
    if results:
        print("\n" + "="*70)
        print("版本对比总结表")
        print("="*70)
        
        print("\n| 版本 | 参数量(M) | FP32大小(MB) | 相对v2增长 |")
        print("|------|----------|------------|-----------|")
        
        v2_info = next((r for r in results if r['name'] == 'v2'), None)
        v2_params = v2_info['total_params'] if v2_info else None
        
        for res in results:
            version = res['name']
            params_m = res['total_params'] / 1e6
            size_mb = res['model_size_fp32_mb']
            
            if v2_params and version != 'v2':
                param_diff = (res['total_params'] - v2_params) / v2_params * 100
                param_str = f"+{param_diff:.1f}%" if param_diff > 0 else f"{param_diff:.1f}%"
            else:
                param_str = "baseline" if version == 'v2' else "-"
            
            print(f"| {version:4s} | {params_m:8.2f} | {size_mb:10.2f} | {param_str:9s} |")
    
    # 内存与推理估计
    print("\n" + "="*70)
    print("推理成本估计（单样本推理）")
    print("="*70)
    
    # 基于标准深度学习推理时间估计
    # 对于这样的轻量模型：~1-5ms on CPU, <1ms on GPU
    
    print("\n推理时间粗估（单样本，batch_size=1）：")
    print("  CPU (Intel i7/i9):        5-20 ms")
    print("  CPU (AMD Ryzen):          5-15 ms")
    print("  GPU (RTX 3090/4090):      <1 ms")
    print("  GPU (V100/A100):          <1 ms")
    print("  Edge Device (ARM CPU):    50-200 ms")
    
    print("\n内存占用估计（推理）：")
    for res in results:
        # 推理时：模型参数 + 中间激活 (粗估为参数的 2-3 倍)
        total_mem_mb = res['model_size_fp32_mb'] * 3  # 参数 + activations
        print(f"  {res['name']}: ~{total_mem_mb:.1f} MB (含激活值)")
    
    print("\n✓ 分析完成")

if __name__ == "__main__":
    main()
