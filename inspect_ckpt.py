import torch
import sys
import os

ckpt_path = "checkpoints/BiFlowNet_0453500.pt"
try:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"Keys: {ckpt.keys()}")
    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
        
    if 'ema' in ckpt:
        print("Found EMA, checking that too")
        
    # Check init_conv
    # Keys might have 'module.' prefix
    keys = list(state_dict.keys())
    init_conv_weight = next((k for k in keys if 'init_conv.weight' in k), None)
    
    if init_conv_weight:
        w = state_dict[init_conv_weight]
        print(f"init_conv shape: {w.shape}")
        # (Out, In, K, K, K) -> Out is model_dim
        
    # Check downs to infer dim_mults
    # downs.0.0.block1.proj.weight
    # downs.1.0.block1.proj.weight ...
    
    for i in range(10):
        k = next((k for k in keys if f'downs.{i}.0.block1.proj.weight' in k), None)
        if k:
            w = state_dict[k]
            print(f"downs[{i}] block1 weight: {w.shape}")
        else:
            break
            
except Exception as e:
    print(f"Error: {e}")

