import torch
from torch import nn
from ddpm.BiFlowNet import BiFlowNet, DiTBlock, FinalLayer, Mlp, PatchEmbed_Voxel, ResnetBlock, AttentionBlock, Downsample, PreNorm, Residual, Upsample, Block
import copy
from einops import rearrange

def exists(x):
    return x is not None

class ZeroConv3d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 1, padding=0)
        nn.init.constant_(self.conv.weight, 0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)

class ControlNet(nn.Module):
    def __init__(self, biflownet_model, control_channels=2):
        super().__init__()
        # We don't store biflownet_model as a member to avoid saving it twice when saving ControlNet
        # But we use it for init.
        
        # Copy encoder parts
        self.init_conv = copy.deepcopy(biflownet_model.init_conv)
        self.channels = biflownet_model.channels
        self.dim = biflownet_model.dim
        self.sub_volume_size = biflownet_model.sub_volume_size
        self.patch_size = biflownet_model.patch_size
        self.vq_size = biflownet_model.vq_size
        
        # Adapt init_conv
        old_weights = self.init_conv.weight.data
        new_conv = nn.Conv3d(
            self.channels + control_channels,
            self.dim,
            kernel_size=self.init_conv.kernel_size,
            padding=self.init_conv.padding
        )
        
        new_conv.weight.data[:, :self.channels] = old_weights
        nn.init.zeros_(new_conv.weight.data[:, self.channels:])
        new_conv.bias.data = self.init_conv.bias.data
        self.init_conv = new_conv

        self.time_mlp = copy.deepcopy(biflownet_model.time_mlp)
        
        self.cond_classes = biflownet_model.cond_classes
        if self.cond_classes is not None:
            self.cond_emb = copy.deepcopy(biflownet_model.cond_emb)
            
        self.res_condition = biflownet_model.res_condition
        if self.res_condition:
            self.res_mlp = copy.deepcopy(biflownet_model.res_mlp)

        self.x_embedder = copy.deepcopy(biflownet_model.x_embedder)
        
        # Adapt x_embedder (PatchEmbed_Voxel) for control channels
        # x_embedder.proj is the Conv3d layer
        old_embedder_conv = self.x_embedder.proj
        new_embedder_conv = nn.Conv3d(
            self.channels + control_channels,
            self.dim,
            kernel_size=old_embedder_conv.kernel_size,
            stride=old_embedder_conv.stride,
            padding=old_embedder_conv.padding,
            bias=old_embedder_conv.bias is not None
        )
        
        # Copy weights
        # old_embedder_conv weight shape: (dim, in_chans, k, k, k)
        new_embedder_conv.weight.data[:, :self.channels] = old_embedder_conv.weight.data
        nn.init.zeros_(new_embedder_conv.weight.data[:, self.channels:])
        if old_embedder_conv.bias is not None:
             new_embedder_conv.bias.data = old_embedder_conv.bias.data
             
        self.x_embedder.proj = new_embedder_conv

        self.pos_embed = copy.deepcopy(biflownet_model.pos_embed)
        
        self.IntraPatchFlow_input = copy.deepcopy(biflownet_model.IntraPatchFlow_input)
        
        # Need unpatchify_voxels helper? 
        # It depends on self.sub_volume_size etc.
        # We can implement it here or copy it.
        
        self.downs = copy.deepcopy(biflownet_model.downs)
        self.feature_fusion = biflownet_model.feature_fusion
        
        self.mid_block1 = copy.deepcopy(biflownet_model.mid_block1)
        self.mid_spatial_attn = copy.deepcopy(biflownet_model.mid_spatial_attn)
        self.mid_block2 = copy.deepcopy(biflownet_model.mid_block2)

        self.zero_convs_intra = nn.ModuleList()
        for i in range(len(self.IntraPatchFlow_input)):
            self.zero_convs_intra.append(ZeroConv3d(self.dim))

        self.zero_convs_downs = nn.ModuleList()
        # Track dimensions for zero convs
        dims = [self.dim] + [self.dim * m for m in biflownet_model.dim_mults if m is not None] # Check dim_mults usage in BiFlowNet
        # Actually BiFlowNet: dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        # We assume init_dim == dim
        
        # Let's dynamically create zero convs based on what we see in downs
        # But we need to know the channels.
        # We can inspect the layers in downs.
        
        for blocks in self.downs:
            # blocks is a ModuleList: [block1, attn1, block2, attn2, downsample]
            # block1 output channels?
            # block1 is ResnetBlock.
            # We can check block1.res_conv.out_channels if it exists, or infer.
            # Actually, let's look at BiFlowNet constructor again.
            # block1(dim_in, dim_out)
            # block2(dim_out, dim_out)
            # So output is dim_out.
            # We need 2 zero convs per level (after block1 and block2).
            # The 'dim_out' is encoded in the block.
            
            # Let's inspect block1
            # block1 is ResnetBlock. It has block2 (Block) which has norm (GroupNorm).
            # The num_channels of norm is dim_out.
            dim_out = blocks[0].block2.norm.num_channels
            self.zero_convs_downs.append(ZeroConv3d(dim_out))
            self.zero_convs_downs.append(ZeroConv3d(dim_out))
            
        # Mid block output
        mid_dim = self.mid_block2.block2.norm.num_channels
        self.zero_convs_mid = ZeroConv3d(mid_dim)

    def unpatchify_voxels(self, x0):
        c = self.dim
        p = self.patch_size
        x,y,z = torch.tensor(self.sub_volume_size) // self.patch_size
        # assert x * y * z == x0.shape[1]

        x0 = x0.reshape(shape=(x0.shape[0], x, y, z, p, p, p, c))
        x0 = torch.einsum('nxyzpqrc->ncxpyqzr', x0)
        volume = x0.reshape(shape=(x0.shape[0], c, x * p, y * p, z * p))
        return volume

    def forward(self, x, control, time, y=None, res=None):
        b = x.shape[0]
        ori_shape = (x.shape[2]*8,x.shape[3]*8,x.shape[4]*8) # assuming 8x AE
        
        x_in = torch.cat([x, control], dim=1)
        
        # Ensure control channels match expected spatial dims for patching
        # The BiFlowNet architecture is sensitive to resolution.
        # If x is 192 (image space) but we are passing latents (48)?
        # Ah, we are training on LATENTS.
        # But 'control' is constructed in dataset/Control_dataset.py.
        # In Control_dataset:
        #   D, H, W = self.resolution (which is latent res 48)
        #   control = torch.zeros((2, D, H, W))
        # But wait, BiFlowNet / ControlNet might expect INPUT image resolution if they contain patch embedding?
        # BiFlowNet 4x:
        #   patch_size = 2.
        #   sub_volume_size = (8,8,8) ? Or something else?
        # Let's check train_ControlNet.py args for 4x.
        #   resolution 48 (latent)
        #   patch-size 2
        #   downsample-factor 4
        # Wait, BiFlowNet input is the LATENT if we are training Latent Diffusion.
        # Is BiFlowNet a Latent Diffusion Model or Image Diffusion Model?
        # The paper says MedDiffusion uses VQ-GAN. So it's Latent Diffusion.
        # So input 'x' to BiFlowNet is the latent volume.
        # 4x latent resolution: 48x48x48.
        # 4x BiFlowNet config:
        #   model-dim 72
        #   dim-mults 1 1 2 4 8
        #   volume-channels 8
        #   patch-size 2
        # 
        # In forward():
        #   ori_shape = (x.shape[2]*8, ...) -> This assumes 8x?
        #   This ori_shape is used for rearranging Unet_feature.
        #   
        # The error: "Sizes of tensors must match except in dimension 1. Expected size 192 but got size 48"
        # x is (B, 8, 48, 48, 48).
        # control is (B, 2, 48, 48, 48).
        # torch.cat([x, control], dim=1) -> Should be (B, 10, 48, 48, 48).
        # 
        # Why expected size 192?
        # Maybe control is 192?
        # In Control_dataset:
        #   D, H, W = self.resolution
        #   If self.resolution is [48, 48, 48], control is 48.
        #
        # Let's check if 'x' is somehow 192?
        # If train_ControlNet.py loads data from pre-computed latents, 'z' is 48.
        #
        # Wait, if we use pre-computed latents, z is 48.
        # If we DON'T use pre-computed latents (e.g. 8x training), z was generated from image inside the loop?
        # In 8x training loop:
        #   if AE is not None:
        #       embeddings = AE.encode(...)
        #       z = ...
        #   z matches latent res.
        # 
        # Let's debug sizes in ControlNet forward.
        if x.shape[2:] != control.shape[2:]:
             # Resize control to match x
             control = torch.nn.functional.interpolate(control, size=x.shape[2:], mode='nearest')
             
        x_in = torch.cat([x, control], dim=1)
        
        x_IntraPatch = x_in.clone()
        p = self.sub_volume_size[0]
        x_IntraPatch = x_IntraPatch.unfold(2,p,p).unfold(3,p,p).unfold(4,p,p)
        p1 , p2 , p3= x_IntraPatch.size(2) , x_IntraPatch.size(3) , x_IntraPatch.size(4)
        x_IntraPatch = rearrange(x_IntraPatch , 'b c p1 p2 p3 d h w -> (b p1 p2 p3) c d h w')
        
        x = self.init_conv(x_in)
        # r = x.clone() # Not used in encoder path

        t = self.time_mlp(time) if exists(self.time_mlp) else None
        c = t.shape[-1]
        t_DiT = t.unsqueeze(1).repeat(1,p1*p2*p3,1).view(-1,c)

        if self.cond_classes:
            assert y.shape == (x.shape[0],)
            cond_emb = self.cond_emb(y)
            cond_emb_DiT = cond_emb.unsqueeze(1).repeat(1,p1*p2*p3,1).view(-1,c)
            t = t + cond_emb
            t_DiT = t_DiT + cond_emb_DiT
        if self.res_condition:
            if len(res.shape) == 1:
                res = res.unsqueeze(0)
            res_condition_emb = self.res_mlp(res)
            t = torch.cat((t,res_condition_emb),dim=1)
            res_condition_emb_DiT = res_condition_emb.unsqueeze(1).repeat(1,p1*p2*p3,1).view(-1,c)
            t_DiT = torch.cat((t_DiT,res_condition_emb_DiT),dim=1)
            
        x_IntraPatch = self.x_embedder(x_IntraPatch)
        x_IntraPatch = x_IntraPatch + self.pos_embed
        
        h_DiT_ctrl = []
        h_Unet = [] # We need this to simulate logic, but we don't output it for injection?
        # Actually BiFlowNet uses h_Unet in downs.
        # But we are not injecting into downs. We just compute features.
        
        for i, (Block, MlpLayer) in enumerate(self.IntraPatchFlow_input):
            x_IntraPatch = Block(x_IntraPatch,t_DiT)
            
            # This is where we capture h_DiT for injection
            # In BiFlowNet: h_DiT.append(x_IntraPatch)
            # We process it with ZeroConv (requires unpatchify? No, DiT path is patches)
            # BiFlowNet h_DiT is patches.
            # But ZeroConv3d expects Voxels (B, C, D, H, W)?
            # Wait, standard ControlNet ZeroConv is convolution.
            # If x_IntraPatch is (N, P, C), we can't use Conv3d directly unless we reshape.
            # BiFlowNet uses Block (Attention) on it.
            # If we want to inject into h_DiT, we should match the format.
            # h_DiT contains patches.
            # If we use ZeroConv3d, we need to reshape to voxels, conv, then back to patches?
            # Or use a ZeroLinear?
            # Let's assume we use ZeroConv3d on the spatial representation (Unet_feature) and inject THAT?
            # No, h_DiT is used by IntraPatchFlow_output Block as 'skip'.
            # The 'skip' in DiTBlock: x = self.skip_linear(torch.cat([x,skip], dim = -1))
            # So skip is concatenated.
            # Standard ControlNet ADDS residuals.
            # If BiFlowNet concatenates, we might need to change BiFlowNet to ADD if we want standard ControlNet behavior.
            # Or we project our control feature to match the concatenated dimension?
            # But standard ControlNet is "Add to decoder features".
            # If BiFlowNet uses concat, then adding to the skip connection works if the skip connection is just added.
            # But here it's concatenated.
            # If we add to the skip connection, we are changing the values being concatenated.
            # This seems correct for "ControlNet adaptation".
            
            # Issue: x_IntraPatch is (B*patches, T, Dim).
            # ZeroConv3d expects (B, C, D, H, W).
            # We should probably use a linear layer initialized to zero for DiT path.
            # But let's look at self.zero_convs_intra.
            # I defined them as ZeroConv3d(dim). This is wrong for patch data.
            # I should use ZeroLinear or just Conv1d?
            # Patches are 1D sequence per subvolume.
            # I'll change ZeroConv3d to be generic or check type.
            # Or reshape to 3D for the conv?
            # The patches correspond to a subvolume (8x8x8).
            # We can reshape (T, C) -> (C, 8, 8, 8)? No, T=patches.
            # x_IntraPatch shape: (N, T, C).
            # T = num_patches per subvolume.
            # subvolume=16x16x16, patch=2 -> 8x8x8 patches = 512.
            # It has spatial structure.
            # I can reshape to (N, C, 8, 8, 8), apply Conv3d, reshape back.
            # This preserves spatial inductive bias of ControlNet.
            
            # Capture for injection
            feat = x_IntraPatch
            # Reshape to spatial
            # T = x_IntraPatch.shape[1]
            # d = h = w = int(round(T**(1/3)))
            # feat = rearrange(feat, 'n (d h w) c -> n c d h w', d=d, h=h, w=w)
            # feat = self.zero_convs_intra[i](feat)
            # feat = rearrange(feat, 'n c d h w -> n (d h w) c')
            # h_DiT_ctrl.append(feat)
            
            Unet_feature = self.unpatchify_voxels(MlpLayer(x_IntraPatch,t_DiT))
            Unet_feature = rearrange(Unet_feature, '(b p) c d h w -> b p c d h w', b=b) 
            Unet_feature = rearrange(Unet_feature, 'b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)',
                        p1=ori_shape[0]//self.vq_size, p2=ori_shape[1]//self.vq_size, p3=ori_shape[2]//self.vq_size)
            h_Unet.append(Unet_feature)
            
            # Handle the zero conv for DiT
            # T = 512 for default config?
            # Let's assume cubic.
            T_patches = x_IntraPatch.shape[1]
            side = int(round(T_patches**(1/3)))
            feat_spatial = rearrange(x_IntraPatch, 'n (d h w) c -> n c d h w', d=side, h=side, w=side)
            feat_spatial = self.zero_convs_intra[i](feat_spatial)
            h_DiT_ctrl.append(rearrange(feat_spatial, 'n c d h w -> n (d h w) c'))

        h_ctrl = []
        zero_idx = 0
        
        for idx, (block1, spatial_attn1, block2, spatial_attn2,downsample) in enumerate(self.downs):
            if idx < self.feature_fusion :
                x = x + h_Unet.pop(0)
            
            x = block1(x, t)
            x = spatial_attn1(x)
            # Collect
            h_ctrl.append(self.zero_convs_downs[zero_idx](x))
            zero_idx += 1
            
            x = block2(x, t)
            x = spatial_attn2(x)
            # Collect
            h_ctrl.append(self.zero_convs_downs[zero_idx](x))
            zero_idx += 1
            
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_block2(x, t)
        
        mid_ctrl = self.zero_convs_mid(x)
        
        return {
            'h_DiT': h_DiT_ctrl,
            'h': h_ctrl,
            'mid': mid_ctrl
        }

class ControlledBiFlowNet(nn.Module):
    def __init__(self, biflownet, controlnet, scale=1.0):
        super().__init__()
        self.biflownet = biflownet
        self.controlnet = controlnet
        self.scale = scale
        
    def forward(self, x, t, y=None, res=None, hint=None, control=None):
        if control is None:
            control = hint
        
        control_states = self.controlnet(x, control, t, y, res)
        # Apply scale if needed? Usually implicit in ZeroConv logic (starts at 0).
        # But global scale is also useful.
        if self.scale != 1.0:
            for k in control_states:
                if isinstance(control_states[k], list):
                    control_states[k] = [v * self.scale for v in control_states[k]]
                else:
                    control_states[k] = control_states[k] * self.scale
                    
        return self.biflownet(x, t, y=y, res=res, control_states=control_states)
