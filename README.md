# 3D MedDiffusion: A 3D Medical Diffusion Model for Controllable and High-quality Medical Image Generation
[![Static Badge](https://img.shields.io/badge/Project-page-blue)](https://shanghaitech-impact.github.io/3D-MedDiffusion/)
[![arXiv](https://img.shields.io/badge/arXiv-2402.19043-b31b1b.svg)](https://arxiv.org/abs/2412.13059)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model%20Page-orange)](https://huggingface.co/MMorss/3D-MedDiffusion)

This is the official PyTorch implementation of the paper **3D MedDiffusion: A 3D Medical Diffusion Model for Controllable and High-quality Medical Image Generation** 

![SMD in Action](assets/gif_github.gif)


## Paper Abstract
The generation of medical images presents significant challenges due to their high-resolution and three-dimensional nature. Existing methods often yield suboptimal performance in generating high-quality 3D medical images, and there is currently no universal generative framework for medical imaging.
In this paper, we introduce the 3D Medical Diffusion (3D MedDiffusion) model for controllable, high-quality 3D medical image generation. 
3D MedDiffusion incorporates a novel, highly efficient Patch-Volume Autoencoder that compresses medical images into latent space through patch-wise encoding and recovers back into image space through volume-wise decoding.
Additionally, we design a new noise estimator to capture both local details and global structure information during diffusion denoising process.
3D MedDiffusion can generate fine-detailed, high-resolution images (up to 512x512x512) and effectively adapt to various downstream tasks as it is trained on large-scale datasets covering CT and MRI modalities and different anatomical regions (from head to leg).
Experimental results demonstrate that 3D MedDiffusion surpasses state-of-the-art methods in generative quality and exhibits strong generalizability across tasks such as sparse-view CT reconstruction, fast MRI reconstruction, and data augmentation.

## ✅ ToDo



- ~~📦 Training code for single-resolution release~~  
- ~~🧠 Pre-trained weights (8x downsampling) release~~  
- ~~🌐 Inference code release~~  
- ~~📄 Pre-trained weights (4x downsampling) release~~
- 📝 Training code for multi-resolution release  


## Installation
```
## Clone this repo
git clone https://github.com/ShanghaiTech-IMPACT/3D-MedDiffusion.git


# Setup the environment
conda create -n 3DMedDiffusion python=3.11.11

conda activate 3DMedDiffusion 

pip install -r requirements.txt

```

## Training 
### PatchVolume Autoencoder — Stage 1


```
## 4x compression
python train/train_PatchVolume.py --config config/PatchVolume_4x.yaml

## 8x compression
python train/train_PatchVolume.py --config config/PatchVolume_8x.yaml
```
**Note:**  
1. All training images should be normalized to `[-1, 1]`.  
2. Update the `default_root_dir`and `root_dir` fileds in `config/PatchVolume_4x.yaml` / `config/PatchVolume_8x.yaml` to match your local paths.
3. Provide a `data.json` following the format shown in the `config/PatchVolume_data.json` example.


### PatchVolume Autoencoder — Stage 2

```
## 4x compression
python train/train_PatchVolume_stage2.py --config config/PatchVolume_4x_s2.yaml

## 8x compression
python train/train_PatchVolume_stage2.py --config config/PatchVolume_8x_s2.yaml
```

**Note:** Set the `resume_from_checkpoint` in `PatchVolume_4x.yaml` / `PatchVolume_8x.yaml` to the checkpoint path from Stage 1 training.


### Encode the Images to latents 
```
python train/generate_training_latent.py --data-path config/Singleres_dataset.json --AE-ckpt checkpoints/trained_AE.ckpt --batch-size 4
```

### BiFlowNet
```
torchrun --nnodes=1 --nproc_per_node=8 --master_port 29513 train/train_BiFlowNet_SingleRes.py --data-path config/Singleres_dataset.json --results-dir  /input/your/results/dir --num-classes 2  --AE-ckpt input/your/AE/checkpoint/path  --resolution 32 32 32  --batch-size 48 --num-workers 48 
```

## Inference
Generation using 8x downsampling
```
python evaluation/class_conditional_generation.py --AE-ckpt checkpoints/PatchVolume_8x_s2.ckpt --model-ckpt checkpoints/BiFlowNet_0453500.pt --output-dir input/your/save/dir
```
Generation using 4x downsampling
```
python evaluation/class_conditional_generation_4x.py --AE-ckpt checkpoints/PatchVolume_4x_s2.ckpt --model-ckpt checkpoints/BiFlowNet_4x.pt --output-dir input/your/save/dir
```

**Note:**  Make sure your GPU has at least 40 GB of memory available to run inference at all supported resolutions.


## Pretrained Models
The pretrained checkpoint is provided [here](https://drive.google.com/drive/folders/1h1Ina5iUkjfSAyvM5rUs4n1iqg33zB-J?usp=drive_link):

Please download the checkpoints and put it to ./checkpoints.

## Acknowledgements
This repository builds upon the following excellent open-source projects: [LDMs](https://github.com/CompVis/latent-diffusion) and [medicaldiffusion](https://github.com/firasgit/medicaldiffusion). 