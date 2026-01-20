#!/bin/bash
source .venv/bin/activate

# Training ControlNet on skullstrip data (MRTIBrain task)
# Using 8x AE and 8x Base Model (BiFlowNet_0453500.pt)
# Resolution 24x24x24 (Latent) -> 192x192x192 (Image)
# Patch size 1 for 8x model

torchrun --nproc_per_node=4 train/train_ControlNet.py \
  --data-path data/skullstrip/index.json \
  --results-dir results/controlnet_train_8x \
  --pretrained-base-ckpt checkpoints/BiFlowNet_0453500.pt \
  --AE-ckpt checkpoints/PatchVolume_8x_s2.ckpt \
  --resolution 24 24 24 \
  --patch-size 1 \
  --batch-size 1 \
  --num-workers 4 \
  --downsample-factor 8 \
  --epochs 1000 \
  --log-every 10 \
  --ckpt-every 1000 \
  --model-dim 72 \
  --dim-mults 1 1 2 4 8 \
  --use-attn 0 0 0 1 1 \
  --volume-channels 8
