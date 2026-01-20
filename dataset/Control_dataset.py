from dataset.Singleres_dataset import Singleres_dataset
import torch
import numpy as np
import json
import os
import glob
import torchio as tio

class Control_dataset(Singleres_dataset):
    def __init__(self, root_dir=None, resolution=[32,32,32], generate_latents=False, downsample_factor=8):
        # super().__init__(root_dir=None, resolution=resolution, generate_latents=generate_latents)
        
        self.resolution = resolution
        self.generate_latents = generate_latents
        self.metadata = {}
        self.all_files = []
        self.downsample_factor = downsample_factor
        
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
        
        age, sex = self.metadata[path]
        
        # Normalize age 0-1 (assuming max age 100)
        age = age / 100.0
        
        # Create control tensor (2, D, H, W) matching TARGET LATENT spatial dims
        # Use self.resolution which is the latent resolution
        D, H, W = self.resolution
        
        control = torch.zeros((2, D, H, W), dtype=torch.float32)
        control[0] = age
        control[1] = sex
        
        return data, torch.tensor(cls_idx), torch.tensor(self.resolution)/64.0, control

