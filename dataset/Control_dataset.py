from dataset.Singleres_dataset import Singleres_dataset
import torch
import numpy as np
import json
import os
import glob
import torchio as tio

class Control_dataset(Singleres_dataset):
    def __init__(self, root_dir=None, resolution=[32,32,32], generate_latents=False, downsample_factor=8, latent_root=None):
        # super().__init__(root_dir=None, resolution=resolution, generate_latents=generate_latents)
        
        self.resolution = resolution
        self.generate_latents = generate_latents
        self.metadata = {}
        self.all_files = []
        self.downsample_factor = downsample_factor
        self.latent_root = latent_root
        
        # Calculate target image size
        self.target_image_size = tuple([r * self.downsample_factor for r in resolution])
        self.transform = tio.CropOrPad(self.target_image_size)

        # Load index.json
        if root_dir.endswith('.json'):
            json_path = root_dir
            data_dir = os.path.dirname(json_path)
        else:
            json_path = os.path.join(root_dir, 'index.json')
            data_dir = root_dir
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        print(f"Loading data from {json_path}...")
        
        for entry in data:
            study_uid = entry.get('studyUID')
            age = float(entry.get('age', 0))
            sex_str = entry.get('sex')
            
            # Map sex
            if sex_str == '男':
                sex = 1.0
            elif sex_str == '女':
                sex = 0.0
            else:
                sex = 0.0 
                
            # Find file
            search_pattern = os.path.join(data_dir, f"{study_uid}*_mni.nii.gz")
            found_files = glob.glob(search_pattern)
            
            if found_files:
                file_path = found_files[0]
                # Use class 3 (T1) as default
                self.all_files.append({'3': file_path})
                self.metadata[file_path] = (age, sex)
                
        self.file_num = len(self.all_files)
        print(f"Total files found: {self.file_num}")

    def __getitem__(self, index):
        # latent: (C, D, H, W)
        # y: class index
        # res: resolution
        
        item = list(self.all_files[index].items())[0]
        cls_idx = int(item[0])
        path = item[1]
        
        age, sex = self.metadata[path]
        
        # Normalize age 0-1 (assuming max age 100)
        age = age / 100.0
        
        # Create control tensor (2, D, H, W) matching TARGET LATENT spatial dims
        # Use self.resolution which is the latent resolution
        # For 4x model, latent res is 48.
        # But wait, BiFlowNet 4x architecture might have DIFFERENT downsampling or patching?
        # BiFlowNet config for 4x:
        #   dim_mults = [1, 1, 2, 4, 8]
        #   sub_volume_size = (8,8,8) ? No, sub_volume_size depends on architecture.
        #   BiFlowNet defaults: sub_volume_size=(8,8,8), patch_size=2.
        #   
        # If input 'x' to BiFlowNet is (B, C, 48, 48, 48).
        # And BiFlowNet uses PatchEmbed_Voxel with patch_size=2.
        # Then it patches 48 -> 24.
        # 
        # The error: "Expected size 192 but got size 48".
        # 192 = 48 * 4.
        # This suggests that something expects the input to be 192 (Image resolution).
        # 
        # Is BiFlowNet expecting the RAW image and encoding it internally?
        # No, `train_ControlNet.py` loop:
        #   if AE is not None: z = AE.encode(z)
        #   diffusion.p_losses(model, z, ...)
        # So 'z' is the latent.
        #
        # If 'z' is 48, then 'model' (ControlledBiFlowNet) receives 48.
        # Inside ControlNet.forward(x, control):
        #   x is 48.
        #   control is 48 (from dataset).
        #   torch.cat works.
        #   x_in is 48.
        #   
        #   x_IntraPatch = x_in.clone()
        #   p = self.sub_volume_size[0] (default 8?)
        #   x_IntraPatch.unfold(2, p, p)...
        #   48 is divisible by 8 (6). So this works.
        #
        #   x = self.init_conv(x_in)
        #   ...
        #
        # Where does 192 come from?
        # Maybe `BiFlowNet_4x.pt` was trained on 192 resolution?
        # And maybe it has parameters or buffers that enforce this?
        # Or maybe `ControlNet` is copying `BiFlowNet` which has some hardcoded size?
        #
        # "RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 192 but got size 48 for tensor number 1 in the list."
        # This usually happens in torch.cat or similar.
        # Tensor 0: size 192 (implied by "Expected size 192")? Or Tensor 0 is correct and Tensor 1 is wrong?
        # "Expected size 192 but got size 48 for tensor number 1"
        # Usually means Tensor 0 has size 192 on that dimension.
        #
        # forward: x_in = torch.cat([x, control], dim=1)
        # x is Tensor 0. control is Tensor 1.
        # If x has spatial size 192, and control has 48.
        # Then x must be 192.
        #
        # But 'z' passed to diffusion.p_losses is 48 (latents).
        # How can 'x' in ControlNet be 192?
        #
        # Wait, `diffusion.p_losses` -> `q_sample` -> `denoise_fn(x_noisy, ...)`
        # x_noisy has same shape as x_start (z). So x_noisy is 48.
        #
        # Is it possible that `train_ControlNet.py` is NOT using AE encoding when `latent_root` is None?
        # We modified `train_ControlNet.py` to:
        #   if AE is not None: encode...
        #
        # But for 4x training, we are using `latent_root` -> `Control_dataset` returns PRE-COMPUTED latents (48).
        # So `z` from loader is 48.
        # And `AE` logic in train loop is skipped or `AE.encode` is not called on `z` (which is already latent).
        # Correct.
        #
        # So `z` is 48.
        # So `x` in ControlNet is 48.
        # So why does torch.cat expect 192?
        # That means `x` is 192?
        #
        # OR `control` is 192?
        # If `control` is 192, and `x` is 48.
        # "Expected size 192 but got size 48 for tensor number 1"
        # If dimension is not 1 (channel), but spatial (2,3,4).
        # torch.cat checks all non-cat dimensions match.
        # If `x` is (B, 8, 192, 192, 192) and `control` is (B, 2, 48, 48, 48).
        # Then error: "Sizes of tensors must match except in dimension 1. Expected size 192 but got size 48".
        # This implies `x` is 192.
        #
        # How can `x` be 192?
        # `z` comes from loader. Loader loads from `latent_root`.
        # Latent files are 48x48x48.
        #
        # CHECK: Did we actually point to the right latent directory in `run_controlnet_4x.sh`?
        #   --latent-root data/skullstrip/latents_4x
        #
        # CHECK: Are the files in `latents_4x` actually 48x48x48?
        # We verified one file: `torch.Size([8, 48, 48, 48])`.
        #
        # CHECK: Is it possible `Control_dataset` is NOT using `latent_root` path for some reason?
        # In `__init__`: `self.latent_root = latent_root`.
        # In `__getitem__`: `if self.latent_root is not None: load ... return`.
        #
        # CHECK: Arguments passing in `train_ControlNet.py`.
        #   dataset = Control_dataset(..., latent_root=args.latent_root)
        #
        # Let's verify `train_ControlNet.py` passes `latent_root`.
        # We need to check `train_ControlNet.py` again.
        
        D, H, W = self.resolution
        
        control = torch.zeros((2, D, H, W), dtype=torch.float32)
        control[0] = age
        control[1] = sex
        
        if self.latent_root is not None:
            # Load pre-computed latent
            original_name = os.path.basename(path)
            stem = original_name.replace('.nii.gz', '').replace('.nii', '')
            latent_path = os.path.join(self.latent_root, f"{stem}.pt")
            
            if not os.path.exists(latent_path):
                # Fallback or error? For now error to be safe
                raise FileNotFoundError(f"Latent file not found: {latent_path}")
                
            data = torch.load(latent_path)
            # data is (C, D, H, W). For 4x model (48, 48, 48)
            # Control is constructed using self.resolution which is (48, 48, 48)
            # So they should match.
            
            return data, torch.tensor(cls_idx), torch.tensor(self.resolution)/64.0, control
        
        # Load raw image
        img = tio.ScalarImage(path)
        img = self.transform(img)
        data = img.data.to(torch.float32)
        
        # Normalize to [-1, 1]
        d_min = data.min()
        d_max = data.max()
        if d_max > d_min:
            data = (data - d_min) / (d_max - d_min) * 2.0 - 1.0
        else:
            data = torch.zeros_like(data)
            
        # Transpose dimensions to match AE expectation (C, D, H, W)
        # Assuming input is (C, W, H, D) from tio
        data = data.transpose(1,3).transpose(2,3)
        
        
        return data, torch.tensor(cls_idx), torch.tensor(self.resolution)/64.0, control

