import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from collections import OrderedDict
from glob import glob
from time import time
import argparse
import logging
import os
from ddpm.BiFlowNet import  GaussianDiffusion
from ddpm.BiFlowNet import BiFlowNet
from ddpm.ControlNet import ControlNet, ControlledBiFlowNet
from AutoEncoder.model.PatchVolume import patchvolumeAE
import torchio as tio
import copy
from torch.cuda.amp import autocast, GradScaler
import random
from torch.optim.lr_scheduler import StepLR
from dataset.Control_dataset import Control_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    dist.destroy_process_group()

def create_logger(logging_dir):
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def _ddp_dict(_dict):
    new_dict = {}
    for k in _dict:
        new_dict['module.' + k] = _dict[k]
    return new_dict

def main(args):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    start_epoch = 0
    train_steps = 0
    log_steps = 0
    running_loss = 0

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        if args.ckpt == None:
            experiment_index = len(glob(f"{args.results_dir}/*"))
            model_string_name = "ControlNet"
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        else:
            experiment_dir = os.path.dirname(os.path.dirname(args.ckpt))
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        samples_dir = f"{experiment_dir}/samples" 
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok= True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # 1. Create Base Model (BiFlowNet)
    base_model = BiFlowNet(
            dim=args.model_dim,
            dim_mults=args.dim_mults,
            channels=args.volume_channels,
            init_kernel_size=3,
            cond_classes=args.num_classes,
            learn_sigma=False,
            use_sparse_linear_attn=args.use_attn,
            vq_size=args.vq_size,
            num_mid_DiT = args.num_dit,
            patch_size = args.patch_size
        ).to(device)

    # 2. Load Pretrained Weights into Base Model
    if args.pretrained_base_ckpt:
        logger.info(f"Loading pretrained base model from {args.pretrained_base_ckpt}")
        checkpoint = torch.load(args.pretrained_base_ckpt, map_location='cpu')
        # Handle key mismatch if needed (remove 'module.' or similar)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # Check if DDP
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '') 
            new_state_dict[name] = v
            
        base_model.load_state_dict(new_state_dict, strict=True)
    else:
        logger.warning("No pretrained base model provided! ControlNet will not work as expected.")

    # 3. Create ControlNet and ControlledBiFlowNet
    controlnet = ControlNet(base_model, control_channels=2).to(device)
    model = ControlledBiFlowNet(base_model, controlnet).to(device)

    # 4. Freeze Base Model
    requires_grad(base_model, False)
    requires_grad(controlnet, True)

    diffusion = GaussianDiffusion(
        channels=args.volume_channels,
        timesteps=args.timesteps,
        loss_type=args.loss_type,
    ).to(device)

    # We only optimize ControlNet
    opt = torch.optim.Adam(controlnet.parameters(), lr=1e-5)
    
    # DDP Wrapper
    # Only wrap model? ControlledBiFlowNet contains both.
    # Since base_model is frozen, DDP will synchronize gradients for controlnet only.
    model = DDP(model, device_ids=[rank], find_unused_parameters=False) # find_unused because base_model is frozen

    amp = args.enable_amp
    scaler = GradScaler(enabled=amp)
    
    if args.AE_ckpt:
        AE = patchvolumeAE.load_from_checkpoint(args.AE_ckpt).to(device)
        AE.eval()
    else:
        AE = None

    logger.info(f"ControlNet Parameters: {sum(p.numel() for p in controlnet.parameters()):,}")

    dataset = Control_dataset(args.data_path, resolution=args.resolution, downsample_factor=args.downsample_factor)
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(
        dataset=dataset,
        batch_size = args.batch_size, 
        num_workers=args.num_workers,
        sampler=sampler,
        shuffle=False,
        pin_memory=False,
        drop_last=True
    )

    model.train()
    
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        loader.sampler.set_epoch(epoch)
        for z, y, res, control in loader:
            b = z.shape[0]
            z = z.to(device)
            y = y.to(device)
            res = res.to(device)
            control = control.to(device)
            
            # Encode raw images to latents if AE is available
            if AE is not None:
                with torch.no_grad():
                    embeddings, _ = AE.encode(z, include_embeddings=True, quantize=True)
                    # Normalize embeddings to [-1, 1]
                    min_val = AE.codebook.embeddings.min()
                    max_val = AE.codebook.embeddings.max()
                    z = (embeddings - min_val) / (max_val - min_val) * 2.0 - 1.0
            
            with autocast(enabled=amp):
                t = torch.randint(0, diffusion.num_timesteps, (b,), device=device)
                # Pass control as hint
                loss = diffusion.p_losses(model, z, t, y=y, res=res, hint=control)
                
                scaler.scale(loss).backward()

            scaler.step(opt)
            scaler.update()
            opt.zero_grad()          

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}")
                running_loss = 0
                log_steps = 0

            # Save Checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "controlnet": model.module.controlnet.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "epoch": epoch
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                    # Validation Sampling
                    if AE is not None and device == 1:
                        with torch.no_grad():
                            milestone = train_steps // args.ckpt_every
                            cls_num = np.random.choice(list(range(0, args.num_classes)))
                            volume_size = args.resolution
                            z_sample = torch.randn(1, args.volume_channels, volume_size[0], volume_size[1], volume_size[2], device=device)
                            y_sample = torch.tensor([cls_num], device=device)
                            res_sample = torch.tensor(volume_size, device=device)/64.0
                            
                            # Random Control (Age/Sex)
                            control_sample = torch.zeros((1, 2, volume_size[0], volume_size[1], volume_size[2]), device=device)
                            control_sample[:, 0] = 0.5 # Age 0.5
                            control_sample[:, 1] = 0.0 # Sex 0
                            
                            samples = diffusion.p_sample_loop(
                                model, z_sample, y=y_sample, res=res_sample, hint=control_sample
                            )
                            
                            samples = (((samples + 1.0) / 2.0) * (AE.codebook.embeddings.max() - AE.codebook.embeddings.min())) + AE.codebook.embeddings.min()
                            volume = AE.decode(samples, quantize=True)
                            
                            volume_path = os.path.join(samples_dir, str(f'{milestone}_{str(cls_num)}.nii.gz'))
                            volume = volume.detach().squeeze(0).cpu()
                            volume = volume.transpose(1,3).transpose(1,2)
                            tio.ScalarImage(tensor=volume).save(volume_path)
                
                dist.barrier()

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--loss-type", type=str, default='l1')
    parser.add_argument("--volume-channels", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--model-dim", type=int, default=72)
    parser.add_argument("--dim-mults", nargs='+', type=int, default=[1,1,2,4,8])
    parser.add_argument("--use-attn", nargs='+', type=int, default=[0,0,0,1,1])
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--num-dit", type=int, default=1)
    parser.add_argument("--enable_amp", action='store_true') # Changed from type=bool
    parser.add_argument("--model", type=str,default="BiFlowNet")
    parser.add_argument("--AE-ckpt", type=str, required=True)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8) 
    parser.add_argument("--batch-size", type=int, default=1) # Reduced for ControlNet training/debugging
    parser.add_argument('--resolution', nargs='+', type=int, default=[32, 32, 32])
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--pretrained-base-ckpt", type=str, required=True)
    parser.add_argument("--vq-size", type=int, default=64)
    parser.add_argument("--downsample-factor", type=int, default=4)
    args = parser.parse_args()
    main(args)

