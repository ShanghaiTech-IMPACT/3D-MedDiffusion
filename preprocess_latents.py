import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "."))
sys.path.append(project_root)
import torch
import torch.distributed as dist
import argparse
import logging
from AutoEncoder.model.PatchVolume import patchvolumeAE
from dataset.Control_dataset import Control_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

def main(args):
    # Setup DDP
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, device={device}")

    # Create output directory
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    dist.barrier()

    # Load AE
    print(f"Rank {rank}: Loading AE...")
    AE = patchvolumeAE.load_from_checkpoint(args.AE_ckpt).to(device)
    AE.eval()

    # Setup Dataset and Loader
    # Note: resolution here is used for constructing the dataset logic but not for resizing if we load raw.
    # We want to load raw images and process them.
    # Control_dataset loads image, crops/pads to target size.
    # We need to set downsample_factor correctly so Control_dataset prepares the right input size.
    # For 4x model on 192 input: latent is 48. 48*4 = 192. So downsample_factor=4.
    # But if input images are 256?
    # skullstrip images are various sizes.
    # For 8x run we used resolution 24 -> target 192. (24*8)
    # For 4x run, if we want target 192, we use resolution 48. (48*4)
    dataset = Control_dataset(args.data_path, resolution=args.resolution, downsample_factor=args.downsample_factor)
    
    # sampler = DistributedSampler(dataset, shuffle=False)
    # loader = DataLoader(
    #     dataset=dataset,
    #     batch_size=1, # Process one by one to save memory
    #     num_workers=4,
    #     sampler=sampler,
    #     shuffle=False,
    #     pin_memory=False
    # )

    # print(f"Rank {rank}: Processing {len(loader)} files...")

    # for i, batch in enumerate(tqdm(loader, disable=(rank!=0))):
        # batch: z, y, res, control
        # z is the raw image tensor (B, C, D, H, W) from Control_dataset when latent_root is None
        # But wait, Control_dataset returns 'data'. 
        
        # Access original file path to save with same name
        # Control_dataset doesn't return file path in __getitem__.
        # We need to modify Control_dataset or access it via index.
        # But DistributedSampler shuffles indices.
        # We can't easily get the path unless we return it.
        # Let's rely on the fact that we process sequentially if shuffle=False?
        # No, DDP sampler subsamples.
        
        # We should probably modify Control_dataset to return path or add a new mode.
        # Or, we can just use the fact that we have the dataset object.
        # But sampler indices are internal.
        
        # Hack: dataset[index] returns path if we modify it?
        # Or, just iterate over dataset indices allocated to this rank.
        # pass

    # Better approach: Iterate indices assigned to this rank manually
    # dataset.all_files is list of dicts.
    
    total_files = len(dataset)
    indices = list(range(total_files))
    # Split indices among ranks
    my_indices = indices[rank::dist.get_world_size()]
    
    for idx in tqdm(my_indices, disable=(rank!=0), desc=f"Rank {rank}"):
        # Get item
        # We need raw data and path
        # Control_dataset.__getitem__ returns (data, cls_idx, res, control)
        # It doesn't return path.
        # We need to access path directly.
        
        item = list(dataset.all_files[idx].items())[0]
        # cls_idx = int(item[0])
        path = item[1]
        
        original_name = os.path.basename(path)
        stem = original_name.replace('.nii.gz', '').replace('.nii', '')
        save_path = os.path.join(args.output_dir, f"{stem}.pt")
        
        if os.path.exists(save_path):
            # Check if file is valid
            try:
                torch.load(save_path)
                continue
            except:
                print(f"Corrupt file {save_path}, reprocessing...")
            
        # Load and process using dataset logic manually or via __getitem__?
        # __getitem__ does crop/pad and normalization. We should use it.
        # But we need to ensure we are calling the right method.
        # dataset[idx] calls __getitem__(idx).
        
        try:
            data, _, _, _ = dataset[idx]
            # data is (C, D, H, W)
            
            # Move to GPU
            x = data.unsqueeze(0).to(device) # (1, C, D, H, W)
            
            # Encode
            with torch.no_grad():
                # encode
                # patchvolumeAE.encode returns (quantized, diff) or similar?
                # Let's check patchvolumeAE.encode signature or usage in train_ControlNet.py
                # embeddings, _ = AE.encode(z, include_embeddings=True, quantize=True)
                
                # Use mixed precision if possible
                with torch.cuda.amp.autocast():
                    embeddings, _ = AE.encode(x, include_embeddings=True, quantize=True)
                
                # Normalize
                min_val = AE.codebook.embeddings.min()
                max_val = AE.codebook.embeddings.max()
                z = (embeddings - min_val) / (max_val - min_val) * 2.0 - 1.0
                
                # Save
                torch.save(z.squeeze(0).cpu(), save_path)
                
                # Clear cache
                del x
                del embeddings
                del z
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing {path}: {e}")
            torch.cuda.empty_cache()

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--AE-ckpt", type=str, required=True)
    parser.add_argument('--resolution', nargs='+', type=int, default=[48, 48, 48])
    parser.add_argument("--downsample-factor", type=int, default=4)
    
    args = parser.parse_args()
    main(args)

