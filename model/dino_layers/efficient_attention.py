import os
import logging
import warnings
import torch
from torch import nn
from torch import Tensor
from xformers.ops import memory_efficient_attention

from typing import Callable, List, Any, Tuple, Dict
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp

from itertools import repeat
import collections.abc
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind
        XFORMERS_AVAILABLE = True
        # warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")

class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        rope=None, 
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope 

    def forward(self, x: Tensor, xpos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # BxHxNxC'

        if self.rope is not None and xpos is not None:
            q = self.rope(q, xpos)  # BxHxNxC' -> BxHxNxC'
            k = self.rope(k, xpos)  # BxHxNxC' -> BxHxNxC'
        
        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MemEffSelfAttention(SelfAttention):
    def forward(self, x: Tensor, xpos=None, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads) # BxNx3xHxC'
        q, k, v = [qkv[:, :, i] for i in range(3)]  # BxNxHxC'
        
        if self.rope is not None and xpos is not None:
            assert(q.shape[0] == xpos.shape[0] 
                   and q.shape[1] == xpos.shape[1]
                   and xpos.shape[2] == 2), (q.shape, xpos.shape)
            q = self.rope(q.transpose(1, 2), xpos).transpose(1, 2) # BxHxNxC', BxNx2 -> BxHxNxC'
            k = self.rope(k.transpose(1, 2), xpos).transpose(1, 2) # BxHxNxC', BxNx2 -> BxHxNxC'
        
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        rope = None, 
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope 

    def forward(self, x: Tensor, y: Tensor, xpos=None, ypos=None) -> Tensor:
        B, Nx, C = x.shape
        B, Ny, C = y.shape
        
        q = self.q(x).reshape(B, Nx, self.num_heads, C // self.num_heads).transpose(1, 2) # BxHxNxC'
        k = self.k(y).reshape(B, Ny, self.num_heads, C // self.num_heads).transpose(1, 2) # BxHxNxC'
        v = self.v(y).reshape(B, y, self.num_heads, C // self.num_heads).transpose(1, 2) # BxHxNxC'

        if self.rope is not None:
            if xpos is not None: 
                q = self.rope(q, xpos)  # BxHxNxC' -> BxNxHxC'
            if ypos is not None:
                k = self.rope(k, ypos)  # BxHxNxC' -> BxNxHxC'

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nx, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MemEffCrossAttention(CrossAttention):
    def forward(self, x: Tensor, y: Tensor, xpos=None, ypos=None, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, Nx, C = x.shape
        B, Ny, C = y.shape
        q = self.q(x).reshape(B, Nx, self.num_heads, C // self.num_heads)#.transpose(1, 2) # BxNxHxC'
        k = self.k(y).reshape(B, Ny, self.num_heads, C // self.num_heads)#.transpose(1, 2) # BxMxHxC'
        v = self.v(y).reshape(B, Ny, self.num_heads, C // self.num_heads)#.transpose(1, 2) # BxMxHxC'

        if self.rope is not None:
            if xpos is not None:
                assert(q.shape[0] == xpos.shape[0] 
                       and q.shape[1] == xpos.shape[1]
                       and xpos.shape[2] == 2), (q.shape, xpos.shape)
                q = self.rope(q.transpose(1, 2), xpos).transpose(1, 2)
            if ypos is not None:
                assert(k.shape[0] == ypos.shape[0] 
                       and k.shape[1] == ypos.shape[1]
                       and ypos.shape[2] == 2), (k.shape, ypos.shape)
                k = self.rope(k.transpose(1, 2), ypos).transpose(1, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, Nx, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class MemEffEncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        rope = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values = 1.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = MemEffSelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            rope=rope,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=proj_drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor, xpos=None):
        def attn_residual_func(x: Tensor, xpos=None) -> Tensor:
            return self.ls1(self.attn(x=self.norm1(x), xpos=xpos))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))
        
        if self.training:
            x = x + self.drop_path1(attn_residual_func(x, xpos=xpos))
            x = x + self.drop_path2(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x, xpos=xpos)
            x = x + ffn_residual_func(x)
        
        return x

class MemEffDecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        rope = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        norm_kv: bool = True, # whether to normalize the key and value
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values = 1.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = MemEffCrossAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            rope=rope,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=proj_drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm_kv = norm_layer(dim) if norm_kv else nn.Identity()
        
    def forward(self, x, y, xpos=None, ypos=None):
        def attn_residual_func(x: Tensor, y: Tensor, xpos=None, ypos=None) -> Tensor:
            return self.ls1(self.attn(x=self.norm1(x), y=self.norm_kv(y), xpos=xpos, ypos=ypos))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))
        
        if self.training:
            x = x + self.drop_path1(attn_residual_func(x, y, xpos, ypos))
            x = x + self.drop_path2(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x, y, xpos, ypos)
            x = x + ffn_residual_func(x)

        return x


# patch embedding
class PositionGetter(object):
    """ return positions of patches """

    def __init__(self):
        self.cache_positions = {}
        
    def __call__(self, b, h, w, device):
        if not (h,w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache_positions[h,w] = torch.cartesian_prod(y, x) # (h, w, 2)
        pos = self.cache_positions[h,w].view(1, h*w, 2).expand(b, -1, 2).clone()
        return pos

class PatchEmbed(nn.Module):
    """ just adding _init_weights + position getter compared to timm.models.layers.patch_embed.PatchEmbed"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
        self.position_getter = PositionGetter()
        
    def forward(self, x):
        B, C, H, W = x.shape
        torch._assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        torch._assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, pos
        
    def _init_weights(self):
        w = self.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1])) 

