# 3D MedDiffusion ControlNet Implementation

This document details the implementation of ControlNet for 3D MedDiffusion, enabling conditional generation based on Age and Sex for MRBrain T1/T2 modalities.

## 1. Implementation Overview

The implementation follows the ControlNet paradigm (Zhang et al.) adapted for the specific `BiFlowNet` architecture used in 3D MedDiffusion.

### Architecture (`ddpm/ControlNet.py`)

*   **ControlNet Module**: A trainable copy of the `BiFlowNet` encoder.
    *   **Input**: Takes the noisy latent volume ($x_t$) concatenated with the control volume ($c$) along the channel dimension.
    *   **Encoder Copy**: Includes `init_conv`, `IntraPatchFlow_input` (Transformer blocks), `downs` (ResNet blocks), and `mid_block`.
    *   **Zero Convolutions**: 1x1x1 convolutions initialized to zero are added after every block in the encoder to project features back to the base model's space.
        *   `zero_convs_intra`: For Transformer patch features.
        *   `zero_convs_downs`: For CNN feature maps at multiple resolutions.
        *   `zero_convs_mid`: For the bottleneck features.

*   **ControlledBiFlowNet**: A wrapper class that:
    1.  Freezes the base `BiFlowNet`.
    2.  Runs `ControlNet` to extract feature residuals.
    3.  Injects these residuals into the base `BiFlowNet` decoder (`ups`, `IntraPatchFlow_output`) and middle block.

### BiFlowNet Modifications (`ddpm/BiFlowNet.py`)

*   Modified `forward` method to accept an optional `control_states` dictionary.
*   Added logic to add `control_states` to the skip connections and feature maps in the decoder.

### Conditioning (`dataset/Control_dataset.py`)

*   **Controls**: Age (float 0-1) and Sex (float 0/1).
*   **Spatial Conditioning**: These scalar values are broadcasted to full spatial resolution tensors $(B, 2, D, H, W)$ to match the latent dimensions, preserving spatial correspondence capabilities for future extensions (e.g., segmentation masks).

## 2. Usage Instructions

### Environment
Ensure you are using the project's virtual environment:
```bash
source .venv/bin/activate  # or use .venv/bin/python directly
```

### A. Training (`train/train_ControlNet.py`)

To fine-tune ControlNet on your data:

```bash
python train/train_ControlNet.py \
  --data-path /path/to/MRBrain_data \
  --results-dir results/controlnet_finetune \
  --pretrained-base-ckpt checkpoints/BiFlowNet_0453500.pt \
  --AE-ckpt checkpoints/PatchVolume_8x_s2.ckpt \
  --batch-size 4 \
  --resolution 32 32 32 \
  --num-workers 8
```

*   **Note**: `resolution` refers to the **latent** resolution. If your input images are 256x256x256 and you use the 8x AE, the latent resolution is 32x32x32.

### B. Inference (`inference_ControlNet.py`)

To generate samples with specific Age/Sex conditions:

```bash
python inference_ControlNet.py \
  --modality T1 \
  --age 0.7 \
  --sex 1.0 \
  --base-ckpt checkpoints/BiFlowNet_0453500.pt \
  --ae-ckpt checkpoints/PatchVolume_8x_s2.ckpt \
  --output-dir results/inference \
  --control-scale 1.0
```

*   **Modality**: `T1` (Class 3) or `T2` (Class 4).
*   **Controls**: `age` (0.0 to 1.0), `sex` (0.0 or 1.0).

## 3. Important Details & Notes

1.  **Memory Management**:
    *   3D volumes are memory-intensive. During inference, the script explicitly deletes the diffusion model and clears CUDA cache before running the AutoEncoder decoder to avoid OOM errors on 16GB GPUs.
    *   If OOM persists, the decoding step falls back to CPU automatically.

2.  **Architecture Specifics**:
    *   **Transformer vs CNN**: BiFlowNet is a hybrid. The ControlNet implementation respects this by handling both patch-based (Transformer) and voxel-based (CNN) features.
    *   **Patch Unfolding**: The `ControlNet` replicates the `unfold`/`rearrange` logic of the base model to ensure patch alignment.

3.  **Dataset**:
    *   The current `Control_dataset.py` uses random values for Age/Sex. **Action Required**: Modify `dataset/Control_dataset.py` to load real metadata (e.g., from a CSV file matching filenames) for meaningful training.

4.  **Weights**:
    *   `BiFlowNet` weights must be loaded into the base model.
    *   `ControlNet` weights are initialized from the base model weights (encoder only) at the start of training.
    *   The saved checkpoints in `results/` contain **only** the ControlNet parameters to save space.

## 4. File Structure

*   `ddpm/ControlNet.py`: Core logic for ControlNet and the wrapper.
*   `train/train_ControlNet.py`: Training loop with freezing logic.
*   `inference_ControlNet.py`: Verification and generation script.
*   `dataset/Control_dataset.py`: Dataset loader with Age/Sex channel generation.

