

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

class GuidedDDIMSample(DDIMSampler) :
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None,guidance=None,guidance_strength=0,guidance_space='pixel'):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [torch.cat([
                            unconditional_conditioning[k][i],
                            c[k][i]]) for i in range(len(c[k]))]
                    else:
                        c_in[k] = torch.cat([
                                unconditional_conditioning[k],
                                c[k]])
            elif isinstance(c, list):
                c_in = list()
                assert isinstance(unconditional_conditioning, list)
                for i in range(len(c)):
                    c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # blend pred_x0
        if guidance is not None :
            if guidance_space == 'latent' :
                if isinstance(guidance_strength, float) :
                    pred_x0 = pred_x0 * (1 - guidance_strength) + guidance * guidance_strength
                elif isinstance(guidance_strength, np.ndarray) :
                    guidance_strength = cv2.resize(guidance_strength, (pred_x0.shape[3], pred_x0.shape[2]), cv2.INTER_CUBIC)
                    guidance_strength = torch.from_numpy(guidance_strength).to(pred_x0.dtype).to(pred_x0.device)
                    pred_x0 = pred_x0 * (1 - guidance_strength) + guidance * guidance_strength
                else :
                    raise NotImplemented()
            elif guidance_space == 'pixel' :
                decoded_x0 = self.model.decode_first_stage(pred_x0)
                if isinstance(guidance_strength, float) :
                    decoded_x0 = decoded_x0 * (1 - guidance_strength) + guidance * guidance_strength
                elif isinstance(guidance_strength, np.ndarray) :
                    guidance_strength = torch.from_numpy(guidance_strength).to(decoded_x0.dtype).to(decoded_x0.device)
                    decoded_x0 = decoded_x0 * (1 - guidance_strength) + guidance * guidance_strength
                else :
                    raise NotImplemented()
                decoded_x0.clamp_(-1, 1)
                pred_x0 = self.model.get_first_stage_encoding(self.model.encode_first_stage(decoded_x0))
            else :
                raise NotImplemented()

        # get new eps
        e_t = (x - a_t.sqrt() * pred_x0) / sqrt_one_minus_at

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
    
    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None,
               guidance=None,guidance_schedule_func=lambda x:0.1,guidance_schedule_func_aux={},guidance_space='latent'):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        total_steps = len(timesteps)
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running Guided DDIM Sampling with {len(timesteps)} timesteps, t_start={t_start}")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            p = (i + (total_steps - t_start) + 1) / (total_steps)
            index = total_steps - i - 1
            gs = guidance_schedule_func(p, guidance_schedule_func_aux)
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning,
                                          guidance=guidance,guidance_strength=gs,guidance_space=guidance_space)
            if callback: callback(i)
        return x_dec
    
class GuidedLDM(LatentDiffusion):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('Using Guided LDM!!!!!!!!!!!!!!!!!!!!')


    @torch.no_grad()
    def img2img(
        self,
        img: torch.Tensor, 
        c_text: str, 
        uc_text: str, 
        denoising_strength: float = 0.3, 
        ddim_steps = 50, 
        target_img: torch.Tensor = None,
        guidance_space = 'latent',
        guidance_schedule_func = lambda x: 0.1,
        guidance_schedule_func_aux = {},
        **kwargs) -> torch.Tensor :
        ddim_sampler = GuidedDDIMSample(self)
        self.cond_stage_model.cuda()
        c_text = self.get_learned_conditioning([c_text])
        uc_text = self.get_learned_conditioning([uc_text])
        cond = {"c_crossattn": [c_text]}
        uc_cond = {"c_crossattn": [uc_text]}
        init_latent = self.get_first_stage_encoding(self.encode_first_stage(img))
        if guidance_space == 'latent' :
            if target_img is not None :
                target_latent = self.get_first_stage_encoding(self.encode_first_stage(target_img))
            else :
                target_latent = None
        else :
            target_latent = target_img
        self.first_stage_model.cpu()
        self.cond_stage_model.cpu()
        steps = ddim_steps
        t_enc = int(min(denoising_strength, 0.999) * steps)
        eta = 0
        print('steps', steps, 't_enc', t_enc)

        noise = torch.randn_like(init_latent)
        ddim_sampler.make_schedule(ddim_num_steps=steps, ddim_eta=eta, ddim_discretize="uniform", verbose=False)
        x1 = ddim_sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * int(init_latent.shape[0])).cuda(), noise=noise)

        self.model.cuda()
        decoded = ddim_sampler.decode(
            x1, 
            cond,
            t_enc,
            unconditional_guidance_scale = 7,
            unconditional_conditioning = uc_cond,
            guidance = target_latent,
            guidance_schedule_func = guidance_schedule_func,
            guidance_schedule_func_aux = guidance_schedule_func_aux,
            guidance_space = guidance_space
            )
        self.model.cpu()

        self.first_stage_model.cuda()
        x_samples = self.decode_first_stage(decoded)
        return torch.clip(x_samples, -1, 1)

import os
import torch

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model
