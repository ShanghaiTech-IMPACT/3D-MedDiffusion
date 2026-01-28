
import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial
import torchio as tio
from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T, utils
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import numpy as np
from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many
from timm.models.vision_transformer import Attention
from timm.models.layers import to_2tuple
from torch.utils.data import Dataset, DataLoader



# helpers functions

class PatchEmbed_Voxel(nn.Module):
    """ Voxel to Patch Embedding
    """
    def __init__(self, voxel_size=(16,16,16,), patch_size=2, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        patch_size = (patch_size, patch_size, patch_size)
        num_patches = (voxel_size[0] // patch_size[0]) * (voxel_size[1] // patch_size[1]) * (voxel_size[2] // patch_size[2])
        self.patch_xyz = (voxel_size[0] // patch_size[0], voxel_size[1] // patch_size[1], voxel_size[2] // patch_size[2])
        self.voxel_size = voxel_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        B, C, X, Y, Z = x.shape
        x = x.float()
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT block.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(4*hidden_size*2, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0., eta=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

        if eta is not None: # LayerScale Initialization (no layerscale when None)
            self.gamma1 = nn.Parameter(eta * torch.ones(hidden_features), requires_grad=True)
            self.gamma2 = nn.Parameter(eta * torch.ones(out_features), requires_grad=True)
        else:
            self.gamma1, self.gamma2 = 1.0, 1.0

    def forward(self, x):
        x = self.gamma1 * self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.gamma2 * self.fc2(x)
        x = self.drop2(x)
        return x

def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    print('grid_size:', grid_size)

    grid_x = np.arange(grid_size[0], dtype=np.float32)
    grid_y = np.arange(grid_size[1], dtype=np.float32)
    grid_z = np.arange(grid_size[2], dtype=np.float32)

    grid = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')  # here y goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size[0], grid_size[1], grid_size[2]])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    # assert embed_dim % 3 == 0

    # use half of dimensions to encode grid_h
    emb_x = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (X*Y*Z, D/3)
    emb_y = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (X*Y*Z, D/3)
    emb_z = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (X*Y*Z, D/3)

    emb = np.concatenate([emb_x, emb_y, emb_z], axis=1) # (X*Y*Z, D)
    return emb
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, skip=False,**block_kwargs):
        super().__init__()
        self.skip_linear = nn.Linear(2*hidden_size, hidden_size) if skip else None
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(4 * hidden_size*2, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c , skip= None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x,skip], dim = -1))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x



def exists(x):
    return x is not None


def noop(*args, **kwargs):
    pass


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

# relative positional bias


class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads=8,
        num_buckets=32,
        max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance /
                                                        max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

# small helper modules


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (4, 4, 4), (2, 2, 2), (1, 1, 1))


def Downsample(dim):
    return nn.Conv3d(dim, dim, (4, 4, 4), (2, 2, 2), (1, 1, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)




class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=6):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (3, 3, 3), padding=(1, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=6):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads # 256
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, z, h, w = x.shape
        x = rearrange(x,'b c z x y -> b (z x y) c').contiguous()
        qkv = self.to_qkv(x).chunk(3, dim=2)
        q, k, v = rearrange_many(
            qkv, 'b d (h c) -> b h d c ', h=self.heads)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale, dropout_p=0.0, is_causal=False)
        out = rearrange(out, 'b h (z x y) c -> b (h c) z x y ',z = z, x = h ,y = w ).contiguous()
        out = self.to_out(out)
        return out

class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(
            tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(
            x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x




class BiFlowNet(nn.Module):
    def __init__(
        self,
        dim,
        learn_sigma = False,
        cond_classes=None,
        dim_mults=(1, 1, 2, 4, 8),
        sub_volume_size = (8,8,8),
        patch_size = 2,
        channels=3,
        attn_heads=8,#
        init_dim=None,
        init_kernel_size=3,
        use_sparse_linear_attn=[0,0,0,1,1],
        resnet_groups=24, #
        DiT_num_heads = 8, #
        mlp_ratio=4,
        vq_size=64,
        res_condition=True,
        num_mid_DiT=1
    ):
        self.cond_classes = cond_classes
        self.res_condition=res_condition

        super().__init__()
        self.channels = channels
        self.vq_size = vq_size
        out_dim = 2*channels if learn_sigma else channels
        self.dim = dim
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2

        self.init_conv = nn.Conv3d(channels, init_dim, (init_kernel_size, init_kernel_size,
                                   init_kernel_size), padding=(init_padding, init_padding, init_padding))

    
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.feature_fusion = np.asarray([item[0]==item[1] for item in in_out ]).sum()
        self.num_mid_DiT= num_mid_DiT
        # time conditioning

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # text conditioning

        if self.cond_classes is not None:
            self.cond_emb = nn.Embedding(cond_classes, time_dim)
        if self.res_condition is not None:
            self.res_mlp =nn.Sequential(nn.Linear(3, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
            time_dim = 2* time_dim
        # layers
        ### miniDiT blocks 
        self.sub_volume_size = sub_volume_size
        self.patch_size = patch_size
        self.x_embedder = PatchEmbed_Voxel(sub_volume_size, patch_size, channels, dim, bias=True)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim), requires_grad=False)
        self.IntraPatchFlow_input = nn.ModuleList()
        for i in range(self.feature_fusion):
            temp = [DiTBlock(dim, 
                     DiT_num_heads, 
                     mlp_ratio=mlp_ratio,
                     )]
            temp.append(FinalLayer(dim,self.patch_size,dim))
            self.IntraPatchFlow_input.append(nn.ModuleList(temp))
        self.IntraPatchFlow_input = nn.ModuleList(self.IntraPatchFlow_input)

        self.IntraPatchFlow_mid = []
        for i in range(self.num_mid_DiT):
            self.IntraPatchFlow_mid.append(DiTBlock(dim, 
                     DiT_num_heads, 
                     mlp_ratio=mlp_ratio,
                     ))
        self.IntraPatchFlow_mid = nn.ModuleList(self.IntraPatchFlow_mid)


        self.IntraPatchFlow_output = nn.ModuleList()
        for i in range(self.feature_fusion):
            temp = [DiTBlock(dim, 
                     DiT_num_heads, 
                     mlp_ratio=mlp_ratio,
                     skip= True
                     )]
            temp.append(FinalLayer(dim,self.patch_size,dim))
            self.IntraPatchFlow_output.append(nn.ModuleList(temp))
        self.IntraPatchFlow_output = nn.ModuleList(self.IntraPatchFlow_output)
        ###

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type

        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=time_dim)

        # modules for all layers

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind == (num_resolutions - 1)
            is_first = ind < self.feature_fusion - 1
            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                Residual(PreNorm(dim_out, AttentionBlock(
                    dim_out, heads=attn_heads))) if use_sparse_linear_attn[ind] else nn.Identity(),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, AttentionBlock(
                    dim_out, heads=attn_heads))) if use_sparse_linear_attn[ind] else nn.Identity(),
                Downsample(dim_out) if not is_last and not is_first else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)



        self.mid_spatial_attn = Residual(PreNorm(mid_dim, AttentionBlock(
                    mid_dim, heads=attn_heads)))

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 2)
            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_out),
                Residual(PreNorm(dim_out, AttentionBlock(
                    dim = dim_out, heads=attn_heads))) if use_sparse_linear_attn[len(in_out)- ind -1] else nn.Identity(),
                block_klass_cond(dim_out * 2, dim_in),
                Residual(PreNorm(dim_in, AttentionBlock(
                    dim = dim_in, heads=attn_heads))) if use_sparse_linear_attn[len(in_out)- ind -1] else nn.Identity(),
                Upsample(dim_in) if not is_last  else nn.Identity()
            ]))


        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )
        self.initialize_weights()
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], (self.sub_volume_size[0]//self.patch_size, self.sub_volume_size[1]//self.patch_size , self.sub_volume_size[2]//self.patch_size))
        self.pos_embed.data.copy_(torch.Tensor(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        for blocks in self.IntraPatchFlow_input:
            for block in blocks:
                if isinstance(block,DiTBlock):
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
                else:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
                    nn.init.constant_(block.linear.weight, 0)
                    nn.init.constant_(block.linear.bias, 0)
                    
        for block in self.IntraPatchFlow_mid:
            if isinstance(block,DiTBlock):
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            else:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
                nn.init.constant_(block.linear.weight, 0)
                nn.init.constant_(block.linear.bias, 0)
        
        for blocks in self.IntraPatchFlow_output:
            for block in blocks:
                if isinstance(block,DiTBlock):
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
                else:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
                    nn.init.constant_(block.linear.weight, 0)
                    nn.init.constant_(block.linear.bias, 0)



    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=2.,
        **kwargs
    ):
        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        y=None,
        res=None,
    ):
        assert (y is not None) == (
            self.cond_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        # y = (y*0).to(torch.int)
        
        b = x.shape[0]
        ori_shape = (x.shape[2]*8,x.shape[3]*8,x.shape[4]*8) 
        # time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)
        x_IntraPatch = x.clone()
        # x_IntraPatch.retain_grad() 
        p = self.sub_volume_size[0]
        x_IntraPatch = x_IntraPatch.unfold(2,p,p).unfold(3,p,p).unfold(4,p,p)
        p1 , p2 , p3= x_IntraPatch.size(2) , x_IntraPatch.size(3) , x_IntraPatch.size(4)
        x_IntraPatch = rearrange(x_IntraPatch , 'b c p1 p2 p3 d h w -> (b p1 p2 p3) c d h w')
        x = self.init_conv(x)
        r = x.clone()


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
        h_DiT , h_Unet,h,=[],[],[]
        for Block, MlpLayer in self.IntraPatchFlow_input:
            x_IntraPatch = Block(x_IntraPatch,t_DiT)
            h_DiT.append(x_IntraPatch)
            Unet_feature = self.unpatchify_voxels(MlpLayer(x_IntraPatch,t_DiT))
            Unet_feature = rearrange(Unet_feature, '(b p) c d h w -> b p c d h w', b=b) 
            Unet_feature = rearrange(Unet_feature, 'b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)',
                        p1=ori_shape[0]//self.vq_size, p2=ori_shape[1]//self.vq_size, p3=ori_shape[2]//self.vq_size)
            h_Unet.append(Unet_feature)

        for Block in self.IntraPatchFlow_mid:
            x_IntraPatch = Block(x_IntraPatch,t_DiT)

        for Block, MlpLayer in self.IntraPatchFlow_output:
            x_IntraPatch = Block(x_IntraPatch,t_DiT , h_DiT.pop())
            Unet_feature = self.unpatchify_voxels(MlpLayer(x_IntraPatch,t_DiT))
            Unet_feature = rearrange(Unet_feature, '(b p) c d h w -> b p c d h w', b=b) 
            Unet_feature = rearrange(Unet_feature, 'b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)',
                        p1=ori_shape[0]//self.vq_size, p2=ori_shape[1]//self.vq_size, p3=ori_shape[2]//self.vq_size)
            h_Unet.append(Unet_feature)
        

        for idx, (block1, spatial_attn1, block2, spatial_attn2,downsample) in enumerate(self.downs):
            if idx <self.feature_fusion :
                x = x + h_Unet.pop(0)
            x = block1(x, t)
            x = spatial_attn1(x)
            h.append(x)
            x = block2(x, t)
            x = spatial_attn2(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_block2(x, t)

        for idx, (block1, spatial_attn1,block2, spatial_attn2,  upsample) in enumerate(self.ups):
            if len(self.ups)-idx <= 2:
                x = x + h_Unet.pop(0)
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = spatial_attn1(x)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = spatial_attn2(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)
    def unpatchify_voxels(self, x0):
        """
        input: (N, T, patch_size * patch_size * patch_size * C)    (N, 64, 8*8*8*3)
        voxels: (N, C, X, Y, Z)          (N, 3, 32, 32, 32)
        """
        c = self.dim
        p = self.patch_size
        x,y,z = np.asarray(self.sub_volume_size) // self.patch_size
        assert x * y * z == x0.shape[1]

        x0 = x0.reshape(shape=(x0.shape[0], x, y, z, p, p, p, c))
        x0 = torch.einsum('nxyzpqrc->ncxpyqzr', x0)
        volume = x0.reshape(shape=(x0.shape[0], c, x * p, y * p, z * p))
        return volume
# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        text_use_bert_cls=False,
        channels=3,
        timesteps=1000,
        loss_type='l1',
        use_dynamic_thres=False,  # from the Imagen paper
        dynamic_thres_percentile=0.9,

    ):
        super().__init__()
        self.channels = channels





        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffer helper function that casts float64 to float32

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas *
                        torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev)
                        * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # text conditioning parameters

        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, denoise_fn,x, t, clip_denoised: bool, y=None, res=None,hint = None,cond_scale=1.):
        if hint == None:
            x_recon = self.predict_start_from_noise(
                        x, t=t, noise=denoise_fn(x, t, y=y,res=res))
        else:
            x_recon = self.predict_start_from_noise(
            x, t=t, noise=denoise_fn(x, t, y=y, res=res,hint = hint))

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, denoise_fn , x, t, y=None, res= None,hint = None,cond_scale=1., clip_denoised=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(denoise_fn,
            x=x, t=t, clip_denoised=clip_denoised, y=y, res=res,hint = hint,cond_scale=cond_scale)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, denoise_fn, z, y=None, res=None,cond_scale=1.,hint = None):
        device = self.betas.device

        b = z.shape[0]
        img = default(z, lambda: torch.randn_like(z , device= device))

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(denoise_fn, img, torch.full(
                (b,), i, device=device, dtype=torch.long), y=y, res=res,cond_scale=cond_scale,hint= hint)

        return img

    @torch.inference_mode()
    def sample(self, denoise_fn, z, y=None, res=None,cond_scale=1.,hint = None, strategy='ddpm', eta=0.0, ddim_steps= 100):
        if strategy == 'ddpm':
            return self.p_sample_loop(
                        denoise_fn, z, y, res, cond_scale, hint
                    )
        elif strategy == 'ddim':
            return self.p_sample_loop_ddim(
                denoise_fn, z, y, res, cond_scale, hint, eta, ddim_steps
            )
        else:
            raise NotImplementedError


    @torch.inference_mode()
    def ddim_sample(self, denoise_fn, x, x_cond, t, t_prev, y=None, hint=None, cond_scale=1., clip_denoised=False):
        b, *_, device = *x.shape, x.device

        # Get predicted x0
        if hint is None:
            x_recon = denoise_fn(torch.cat((x, x_cond), dim=1), t, y=y)
        else:
            x_recon = denoise_fn(torch.cat((x, x_cond), dim=1), t, y=y, hint=hint)

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )
                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))
            x_recon = x_recon.clamp(-s, s) / s

        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_prev = self.alphas_cumprod[t_prev].view(-1, 1, 1, 1)
        sqrt_alpha_t = alpha_t.sqrt()
        sqrt_alpha_prev = alpha_prev.sqrt()

        # compute direction pointing to x_t
        eps = (x - sqrt_alpha_t * x_recon) / (1 - alpha_t).sqrt()

        # deterministic DDIM update
        x_prev = sqrt_alpha_prev * x_recon + (1 - alpha_prev).sqrt() * eps
        return x_prev


    @torch.inference_mode()
    def p_sample_loop_ddim(self, denoise_fn, z, y=None, res=None, cond_scale=1., hint=None, eta=0.0, ddim_steps=100):
        device = self.betas.device
        b = z.shape[0]
        img = default(z, lambda: torch.randn_like(z, device=device))

        # DDIM time schedule
        times = torch.linspace(0, self.num_timesteps - 1, steps=ddim_steps).long().flip(0).to(device)


        for i in tqdm(range(ddim_steps), desc="DDIM Sampling", total=ddim_steps):
            t = times[i]
            t_prev = times[i + 1] if i < ddim_steps - 1 else torch.tensor(0, device=device)
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            t_prev_batch = torch.full((b,), t_prev, device=device, dtype=torch.long)

            img = self.p_sample_ddim(
                denoise_fn, img, t_batch, t_prev_batch,
                y=y, res=res, hint=hint, cond_scale=cond_scale,
                eta=eta, clip_denoised=True
            )

        return img

    @torch.inference_mode()
    def p_sample_ddim(self, denoise_fn, x, t, t_prev, y=None, res=None, hint=None, cond_scale=1., clip_denoised=True, eta=0.0):
        # Predict noise (ε_theta)
        if hint is None:
            noise_pred = denoise_fn(x, t, y=y, res=res)
        else:
            noise_pred = denoise_fn(x, t, y=y, res=res, hint=hint)

        # Predict x0
        x0 = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x0, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )
                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x0.ndim - 1)))
            x0 = x0.clamp(-s, s) / s

        # DDIM formula
        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_prev = self.alphas_cumprod[t_prev].view(-1, 1, 1, 1)
        sigma = eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
        
        pred_dir = (1 - alpha_prev).sqrt() * noise_pred
        x_prev = alpha_prev.sqrt() * x0 + pred_dir

        if eta > 0:
            noise = torch.randn_like(x)
            x_prev = x_prev + sigma * noise

        return x_prev

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    def p_losses(self, denoise_fn,x_start, t, y=None, res=None,noise=None, hint = None,**kwargs):
        b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if is_list_str(y):
            y = bert_embed(
                tokenize(y), return_cls_repr=self.text_use_bert_cls)
            y = y.to(device)
        if hint == None:
            x_recon = denoise_fn(x_noisy, t, y=y,res=res)
        else:
            x_recon = denoise_fn(x_noisy, t, y=y, res=res,hint = hint)
        # time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)
        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss