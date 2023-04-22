from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import List, Optional, Any, Tuple

from ldm.modules.diffusionmodules.util import checkpoint


try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = False
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None,ori_shape=None):
        if ori_shape is not None :
            print('ori_shape', ori_shape)
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        print(q.shape, k.shape, v.shape, sim.shape)
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

ATTENTION_BIAS_CACHE = {}
import numpy as np
class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        # print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
        #       f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward_mfr(self, x, context=None, mask=None,ori_shape=None,ref_kv_hists:List[Tuple[torch.Tensor, torch.Tensor]]=[]): # ref_kv_hist is a list of (k, v, (optionally flow)) from different reference images
        #print('using MemoryEfficientCrossAttention')
        q = self.to_q(x)
        is_self_attn = False
        if context is None :
            is_self_attn = True
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        kv_hist = None

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        if is_self_attn :
            kv_hist = (k, v)
            attn_entire_reference_image = True
            ql = q.shape[1]
            kl = k.shape[1]
            attn_bias = torch.zeros(ql, kl).type(q.dtype).to(q.device)
            n, ch, h, w = ori_shape
            if w > h :
                ext = int(w / (h / 1.5))
            else :
                ext = 1
            #print('added head', q.shape, k.shape, v.shape, attn_bias.shape, 'ext=',ext)
            attn_radius = 6
            attn_w = 4
            attn_sigma = 0.8
            sigma_from_h = {
                96: 1,
                48: 0.8,
                24: 0.6,
                12: 0.4
            }
            if ext > 1 :
                assert ext == 2
                w //= ext
                key = (ext, h, w)
                if key in ATTENTION_BIAS_CACHE :
                    attn_bias = ATTENTION_BIAS_CACHE[key]
                else :
                    # attn_bias.fill_(-100)
                    # for i in range(h) :
                    #     # make sure reference image's self attn works within itself
                    #     for j in range(h) :
                    #         x = j * 2 + 1
                    #         y = i * 2 + 1
                    #         attn_bias[y * w: (y + 1) * w, x * w: (x + 1) * w] = 0
                    if attn_entire_reference_image :
                        pass
                    attn_sigma = sigma_from_h[h]
                    wblock = attn_w * torch.eye(w).to(q.device).to(q.dtype) # increase weight by attn_w
                    for j in range(-attn_radius, attn_radius + 1) :
                        if j == 0 :
                            continue
                        dist = np.abs(j)
                        val = attn_w * np.exp(-dist / attn_sigma)
                        for k2 in range(w) :
                            x = k2 + j
                            y = k2
                            if x >= 0 and x < w and y >= 0 and y < w :
                                wblock[y, x] = val
                    for i in range(h) :
                        x = 1 + 2 * i
                        y = x - 1
                        attn_bias[y * w: (y + 1) * w, x * w: (x + 1) * w] = wblock
                        for j in range(-attn_radius, attn_radius + 1) : # vertical
                            if j == 0 :
                                continue
                            wblock_offdiag = attn_w * torch.eye(w).to(q.device).to(q.dtype)
                            for k2 in range(-attn_radius, attn_radius + 1) : # horizontal
                                dist = np.sqrt(j * j + k2 * k2)
                                val = attn_w * np.exp(-dist / attn_sigma)
                                for l in range(w) : # for all pixels in this horizontal line
                                    x3 = l + k2
                                    y3 = l
                                    if x3 >= 0 and x3 < w and y3 >= 0 and y3 < w :
                                        wblock_offdiag[y3, x3] = val
                            x2 = x + 2 * j
                            y2 = y
                            if x2 >= 0 and x2 * w < kl :
                                attn_bias[y2 * w: (y2 + 1) * w, x2 * w: (x2 + 1) * w] = wblock_offdiag
                    ATTENTION_BIAS_CACHE[key] = attn_bias
        else :
            attn_bias = None
        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=attn_bias, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out), kv_hist
    
    def forward(self, x, context=None, mask=None,ori_shape=None,ref_kv_hists:List[Tuple[torch.Tensor, torch.Tensor]]=[]): # ref_kv_hist is a list of (k, v, (optionally flow)) from different reference images
        #print('using MemoryEfficientCrossAttention')
        q = self.to_q(x)
        is_self_attn = False
        if context is None :
            is_self_attn = True
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        kv_hist = None

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )
        ql = q.size(1)
        kl = k.size(1)

        attn_bias = None
        use_attn_bias = False
        if is_self_attn :
            nhead = self.heads
            kv_hist = (k.cpu(), v.cpu())
            unet_layer = 0
            if ref_kv_hists :
                unet_layer = ref_kv_hists[0][2]

            if ref_kv_hists :
                unet_layer = ref_kv_hists[0][2]
                #print('using k and v from another image!')
                k2 = torch.cat([tk for tk, tv, _ in ref_kv_hists], dim = 1).to(q.device)
                v2 = torch.cat([tv for tk, tv, _ in ref_kv_hists], dim = 1).to(q.device)
                if k2.shape[0] != q.shape[0] :
                    # cross attention only applicable for positive image
                    k[nhead:] = k2
                    v[nhead:] = v2
                else :
                    k = k2
                    v = v2
                if use_attn_bias and unet_layer >= 1 and unet_layer <= 13 : # attention on decode at ds=[2,1]
                    ext = k.size(1) // kl
                    n, ch, h, w = ori_shape
                    key = (h, w)
                    attn_bias = torch.zeros(ql, kl).type(q.dtype).to(q.device)
                    attn_bias.fill_(-1000)
                    attn_radius = 8
                    attn_w = 0
                    attn_sigma = 0.8
                    sigma = 2
                    sigma_from_h = {
                        96: 10001,
                        48: 10000.8,
                        24: 10000.6,
                        12: 10000.4
                    }
                    if key in ATTENTION_BIAS_CACHE :
                        attn_bias = ATTENTION_BIAS_CACHE[key]
                    else :
                        attn_sigma = sigma_from_h[h] * sigma
                        wblock = attn_w * torch.eye(w).to(q.device).to(q.dtype) # increase weight by attn_w
                        for j in range(-attn_radius, attn_radius + 1) :
                            if j == 0 :
                                continue
                            dist = np.abs(j)
                            val = attn_w * np.exp(-dist / attn_sigma)
                            for k2 in range(w) :
                                x = k2 + j
                                y = k2
                                if x >= 0 and x < w and y >= 0 and y < w :
                                    wblock[y, x] = val
                        for i in range(h) :
                            x = i
                            y = x
                            attn_bias[y * w: (y + 1) * w, x * w: (x + 1) * w] = wblock
                            for j in range(-attn_radius, attn_radius + 1) : # vertical
                                if j == 0 :
                                    continue
                                wblock_offdiag = attn_w * torch.eye(w).to(q.device).to(q.dtype)
                                for k2 in range(-attn_radius, attn_radius + 1) : # horizontal
                                    dist = np.sqrt(j * j + k2 * k2)
                                    val = attn_w * np.exp(-dist / attn_sigma)
                                    for l in range(w) : # for all pixels in this horizontal line
                                        x3 = l + k2
                                        y3 = l
                                        if x3 >= 0 and x3 < w and y3 >= 0 and y3 < w :
                                            wblock_offdiag[y3, x3] = val
                                x2 = x + j
                                y2 = y
                                if x2 >= 0 and x2 * w < kl :
                                    attn_bias[y2 * w: (y2 + 1) * w, x2 * w: (x2 + 1) * w] = wblock_offdiag
                        if ext > 1 :
                            attn_bias = torch.concatenate([attn_bias] * ext, dim = 1)
                        ATTENTION_BIAS_CACHE[key] = attn_bias
            
        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=attn_bias, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out), kv_hist

class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        XFORMERS_IS_AVAILBLE=True
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None,ori_shape=None,ref_kv_hists=[]):
        return checkpoint(self._forward, (x, context, ori_shape, ref_kv_hists), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, ori_shape=None,ref_kv_hists=[]):
        attn1_output, kv_hist1 = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None,ori_shape=ori_shape,ref_kv_hists=ref_kv_hists)
        x = attn1_output + x # self attn
        x = self.attn2(self.norm2(x), context=context,ori_shape=ori_shape)[0] + x # cross attn
        x = self.ff(self.norm3(x)) + x
        return x, kv_hist1


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None,reference_kv=[]):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        ori_shape=x.shape
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        kv_hists = []
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x, kv_hist = block(x, context=context[i],ori_shape=ori_shape,ref_kv_hists=reference_kv)
            kv_hists.append(kv_hist)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in, kv_hists

