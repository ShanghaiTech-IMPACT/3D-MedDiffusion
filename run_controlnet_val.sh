#!/bin/bash
source .venv/bin/activate

# Training ControlNet on skullstrip data (MRTIBrain task)
# Using 8x AE and 8x Base Model (BiFlowNet_0453500.pt)
# Resolution 24x24x24 (Latent) -> 192x192x192 (Image)
# Patch size 1 for 8x model

python inference_ControlNet.py \
  --base-ckpt checkpoints/BiFlowNet_0453500.pt \
  --control-ckpt results/controlnet_train_8x/007-ControlNet/checkpoints/0010000.pt \
  --ae-ckpt checkpoints/PatchVolume_8x_s2.ckpt \
  --output-dir results/inference_controlnet_8x \
  --modality T1 \
  --age 0.8 \
  --sex 0.0 \
  --resolution 24 24 24 \
  --timesteps 1000