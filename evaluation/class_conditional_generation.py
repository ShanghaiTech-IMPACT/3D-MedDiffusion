import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import os
from ddpm.BiFlowNet import  GaussianDiffusion
from ddpm import BiFlowNet
from AutoEncoder.model.PatchVolume import patchvolumeAE
import torchio as tio
import numpy as np


class_res_mapping={0:[(32,32,32),(16,32,32)],1:[(32,32,32),(64,64,64)],2:[(32,32,32),(64,32,32)],3:[(24,24,24)],4:[(24,24,24)],5:[(16,32,32)],6:[(8,40,40)]}
spacing_mapping={0:[(1,1,1),(1,1,1)],1:[(1.25,1.25,1.25),(0.7,0.7,1)],2:[(1.25,1.25,1.25),(1.25,1.25,1.25)],3:[(1,1,1)],4:[(1,1,1)],5:[(1.2,1.2,2)],6:[(1,1,1)]}
name_mapping = {0:'CTHeadNeck',1:'CTChestAbdomen',2:'CTLegs',3:'MRTIBrain',4:'MRT2Brain',5:'MRAbdomen',6:'MRKnee'}
def main(args):



    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    args.output_dir = os.path.join(args.output_dir, f'seed_{args.seed}')
    os.makedirs(args.output_dir, exist_ok=True)

    model = BiFlowNet(
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
        ).cuda()
    diffusion= GaussianDiffusion(
        channels=args.volume_channels,
        timesteps=args.timesteps,
        loss_type=args.loss_type,
    ).cuda()
    model_ckpt = torch.load(args.model_ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(model_ckpt['ema'], strict = True)
    model = model.cuda()
    model.train()
    AE = patchvolumeAE.load_from_checkpoint(args.AE_ckpt).cuda()
    AE.eval()
    device = torch.device("cuda")
    output_dir = args.output_dir
    os.makedirs(output_dir , exist_ok=True)


    for key, value in class_res_mapping.items():
        class_idx = key
        anatomy_name = name_mapping[key]
        for idx,res in enumerate(value):
            with torch.no_grad():
                spacing = tuple(spacing_mapping[key][idx])
                affine = np.diag(spacing + (1,))
                output_name = str(f'{anatomy_name}_{res[0]*8}x{res[1]*8}x{res[2]*8}.nii.gz')
                z = torch.randn(1, args.volume_channels, res[0], res[1],res[2], device=device)
                y = torch.tensor([class_idx], device=device)
                res_emb = torch.tensor(res,device=device)/64.0
                samples = diffusion.sample(
                    model, z, y = y, res=res_emb, strategy = args.sampling_strategy
                )
                samples = (((samples + 1.0) / 2.0) * (AE.codebook.embeddings.max() -
                                                    AE.codebook.embeddings.min())) + AE.codebook.embeddings.min()
                if res[0]*res[1]*res[2] <= 32*32*32:
                    volume = AE.decode(samples, quantize=True)
                else:
                    volume = AE.decode_sliding(samples, quantize=True)
                volume_path = os.path.join(output_dir,output_name) 

                volume = volume.detach().squeeze(0).cpu()
                volume = volume.transpose(1,3).transpose(1,2)            
                tio.ScalarImage(tensor = volume,affine = affine).save(volume_path)
            torch.cuda.empty_cache()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--AE-ckpt", type=str, required=True,help="Path to Autoencoder Checkpoint ") 
    parser.add_argument("--model-ckpt", type=str, required=True, help="Path to Diffusion Model Checkpoint")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save the generated images")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--model-dim", type=int, default=72)
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--num-dit", type=int, default=1)
    parser.add_argument("--dim-mults", type=list, default=[1,1,2,4,8])
    parser.add_argument("--use-attn", type=list, default=[0,0,0,1,1])
    parser.add_argument("--sampling-strategy", type=str, default='ddpm',help='The sampling strategy')   
    parser.add_argument("--ddim_steps", type=int, default=100,help='DDIM sampling steps') 
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--loss-type", type=str, default='l1')
    parser.add_argument("--volume-channels", type=int, default=8)
    parser.add_argument("--vq-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=64)
    args = parser.parse_args()
    main(args)

