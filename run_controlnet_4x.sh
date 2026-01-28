#!/bin/bash
source .venv/bin/activate

# Training ControlNet on skullstrip data (MRTIBrain task)
# Using 4x AE and 4x Base Model
# Latents are pre-computed in data/skullstrip/latents_4x
# Resolution 48x48x48 (Latent) -> 192x192x192 (Image)

torchrun --nproc_per_node=4 train/train_ControlNet.py \
  --data-path data/skullstrip/index.json \
  --results-dir results/controlnet_train_4x \
  --pretrained-base-ckpt checkpoints/BiFlowNet_4x.pt \
  --AE-ckpt checkpoints/PatchVolume4x_s2.ckpt \
  --resolution 48 48 48 \
  --patch-size 2 \
  --batch-size 4 \
  --num-workers 4 \
  --epochs 100 \
  --log-every 10 \
  --ckpt-every 1000 \
  --latent-root data/skullstrip/latents_4x \
  --downsample-factor 4 \
  --model-dim 72 \
  --dim-mults 1 1 2 4 8 \
  --use-attn 0 0 0 1 1 \
  --volume-channels 8

