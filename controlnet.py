

from typing import Any, List, Optional, Tuple
import einops
import torch
import torch as th
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import cv2

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from ldm.modules.diffusionmodules.util import noise_like
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import k_diffusion.sampling

class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, reference_kv:List[List[Tuple[torch.Tensor, torch.Tensor]]] = [], **kwargs):
        if control is None :
            return super().forward(x, timesteps, context, **kwargs)
        hs = []
        kv_hists = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                #print(type(module))
                h, kv_hists_cur = module(h, emb, context, reference_kv = reference_kv)
                #print('reference_kv[0].len()', len(reference_kv[0]) if reference_kv else 'N/A')
                kv_hists.extend(kv_hists_cur)
                hs.append(h)
            #print('self.middle_block', type(self.middle_block))
            h, kv_hists_cur = self.middle_block(h, emb, context, reference_kv = reference_kv)
            #print('reference_kv[0].len()', len(reference_kv[0]) if reference_kv else 'N/A')
            kv_hists.extend(kv_hists_cur)

        h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h, kv_hists_cur = module(h, emb, context, reference_kv = reference_kv)
            #print('reference_kv[0].len()', len(reference_kv[0]) if reference_kv else 'N/A')
            kv_hists.extend(kv_hists_cur)

        h = h.type(x.dtype)
        return self.out(h), kv_hists


class ControlNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
                    conv_nd(dims, hint_channels, 16, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(dims, 16, 16, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(dims, 16, 32, 3, padding=1, stride=2),
                    nn.SiLU(),
                    conv_nd(dims, 32, 32, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(dims, 32, 96, 3, padding=1, stride=2),
                    nn.SiLU(),
                    conv_nd(dims, 96, 96, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(dims, 96, 256, 3, padding=1, stride=2),
                    nn.SiLU(),
                    zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                            disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint, _ = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h, _ = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h, _ = module(h, emb, context)
            outs.append(zero_conv(h, emb, context)[0])

        h, _ = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context)[0])

        return outs

from dataclasses import dataclass

@dataclass
class SingleControlNet :
    weight: float
    model: str
    args: dict
    condition: np.ndarray
    guidance_start: float = 0
    guidance_end: float = 1
    result: Optional[Any] = None
    net: Optional[ControlNet] = None

LOADED_CNETS = {}

from controlnet_models.hed import apply_hed
MODEL_HED = None

def extract_control(x_noisy, t, cond_txt, condition_img, net: ControlNet, model: str, control_args: dict) :
    if model == 'canny' :
        edges = cv2.Canny(condition_img, control_args['low_threshold'], control_args['high_threshold'])
        #cv2.imwrite('cnet_canny.png', edges)
        control = einops.repeat(torch.from_numpy(edges), 'h w -> n c h w', n = 2, c = 3).cuda().float() / 255.0
        return net(x=x_noisy, hint=control, timesteps=t, context=cond_txt)
    elif model == 'hed' :
        edges = apply_hed(condition_img)
        #cv2.imwrite('cnet_hed.png', edges)
        control = einops.repeat(torch.from_numpy(edges), 'h w -> n c h w', n = 2, c = 3).cuda().float() / 255.0
        return net(x=x_noisy, hint=control, timesteps=t, context=cond_txt)
    elif model == 'depth' :
        pass
    elif model == 'mlsd' :
        pass
    elif model == 'scribble' :
        pass
    elif model == 'inpaint' :
        detected_mask = control_args['mask']
        H, W, _ = condition_img.shape
        detected_map = condition_img.astype(np.float32).copy()
        detected_map[detected_mask > 127] = -255.0  # use -1 as inpaint value
        cv2.imwrite('cnet_inpaint.png', np.clip(detected_map, 0, 255).astype(np.uint8))

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(1)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        return net(x=x_noisy, hint=control, timesteps=t, context=cond_txt)
    
import safetensors.torch
def get_controlnet_instance(model: str) :
    if model in LOADED_CNETS :
        return LOADED_CNETS[model]
    cnet = ControlNet(
        image_size= 32,
        in_channels= 4,
        hint_channels= 3,
        model_channels= 320,
        attention_resolutions= [ 4, 2, 1 ],
        num_res_blocks= 2,
        channel_mult= [ 1, 2, 4, 4 ],
        num_heads= 8,
        use_spatial_transformer= True,
        transformer_depth= 1,
        context_dim= 768,
        use_checkpoint= True,
        legacy= False
    ).cuda()
    if model == 'canny' :
        # sd = safetensors.torch.load_file('controlnet_models/control_canny-fp16.safetensors')
        # cnet.load_state_dict(sd)
        sd = torch.load('controlnet_models/control_v11p_sd15_canny.pth')
        sd2 = {}
        for k in sd.keys() :
            k = k.replace('control_model.', '')
            sd2[k] = sd['control_model.' + k]
        cnet.load_state_dict(sd2)
    if model == 'hed' :
        sd = safetensors.torch.load_file('controlnet_models/control_hed-fp16.safetensors')
        cnet.load_state_dict(sd)
    if model == 'inpaint' :
        sd = torch.load('controlnet_models/control_v11p_sd15_inpaint.pth')
        sd2 = {}
        for k in sd.keys() :
            k = k.replace('control_model.', '')
            sd2[k] = sd['control_model.' + k]
        cnet.load_state_dict(sd2)
    return cnet

def apply_multi_controlnet(x_noisy, t, cond_txt, denoise_percentage, control_nets: List[SingleControlNet]) :
    for c in control_nets :
        if c.net is None :
            c.net = get_controlnet_instance(c.model)
            x = extract_control(x_noisy, t, cond_txt, c.condition, c.net, c.model, c.args)
            if isinstance(x, tuple) :
                print('x is tuple')
            c.result = x
    controls = []
    for r in control_nets[0].result :
        weight = control_nets[0].weight
        if denoise_percentage < control_nets[0].guidance_start or denoise_percentage > control_nets[0].guidance_end :
            weight = 0
        controls.append(r * weight)
    for cnet in control_nets[1:] :
        for i, r in enumerate(cnet.result) :
            weight = cnet.weight
            if denoise_percentage < cnet.guidance_start or denoise_percentage > cnet.guidance_end :
                weight = 0
            controls[i] += r * weight
    return controls
    
    
def test_load() :
    from ldm.util import instantiate_from_config
    cnet = ControlNet(
        image_size= 32,
        in_channels= 4,
        hint_channels= 3,
        model_channels= 320,
        attention_resolutions= [ 4, 2, 1 ],
        num_res_blocks= 2,
        channel_mult= [ 1, 2, 4, 4 ],
        num_heads= 8,
        use_spatial_transformer= True,
        transformer_depth= 1,
        context_dim= 768,
        use_checkpoint= True,
        legacy= False
    )
    import safetensors.torch
    sd = safetensors.torch.load_file('controlnet_models/control_canny-fp16.safetensors')
    cnet.load_state_dict(sd)

if __name__ == '__main__' :
    test_load()
