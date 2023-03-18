

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

class GuidedDDIMSample(DDIMSampler) :
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None,guidance=None,guidance_strength=0,guidance_weight=None):
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

        # merge pred_x0
        # if guidance_strength > 0 and guidance is not None :
        #     if guidance_weight is not None :
        #         pred_x0 = (pred_x0 * (1 - guidance_strength) + guidance * guidance_strength) * guidance_weight + (1 - guidance_weight) * pred_x0
        #     else :
        #         pred_x0 = pred_x0 * (1 - guidance_strength) + guidance * guidance_strength

        # get new eps
        #e_t = (x - a_t.sqrt() * pred_x0) / sqrt_one_minus_at

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
    
    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, init_latent=None, nmask=None, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None,guidance=None,guidance_schedule_func=lambda x:0.1,guidance_weight=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        total_steps = len(timesteps)
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running Guided DDIM Sampling with {len(timesteps)} timesteps, t_start={t_start}")
        import cv2
        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        last_gs = 1
        for i, step in enumerate(iterator):
            p = (i + (total_steps - t_start) + 1) / (total_steps)
            index = total_steps - i - 1
            gs = guidance_schedule_func(p)
            last_gs = gs
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            if nmask is not None and gs > 0 :
                noised_input = self.model.q_sample(init_latent.cuda(), ts.cuda())
                x_dec = (1 - nmask) * noised_input + nmask * x_dec
            else :
                print('no gs p')
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning,
                                          guidance=guidance,guidance_strength=gs,guidance_weight=guidance_weight)
            if callback: callback(i)
        return x_dec, last_gs
    
def get_inpainting_image_condition(model, image, mask) :
    conditioning_mask = np.array(mask.convert("L"))
    conditioning_mask = conditioning_mask.astype(np.float32) / 255.0
    conditioning_mask = torch.from_numpy(conditioning_mask[None, None])
    conditioning_mask = torch.round(conditioning_mask)
    conditioning_mask = conditioning_mask.to(device=image.device, dtype=image.dtype)
    conditioning_image = torch.lerp(
        image,
        image * (1.0 - conditioning_mask),
        1
    )
    conditioning_image = model.get_first_stage_encoding(model.encode_first_stage(conditioning_image))
    conditioning_mask = torch.nn.functional.interpolate(conditioning_mask, size=conditioning_image.shape[-2:])
    conditioning_mask = conditioning_mask.expand(conditioning_image.shape[0], -1, -1, -1)
    image_conditioning = torch.cat([conditioning_mask, conditioning_image], dim=1)
    return image_conditioning

def get_empty_image_condition(latent) :
    return latent.new_zeros(latent.shape[0], 5, latent.shape[2], latent.shape[3])

from PIL import Image, ImageFilter, ImageOps

def fill_mask_input(image, mask):
    """fills masked regions with colors from image using blur. Not extremely effective."""

    image_mod = Image.new('RGBA', (image.width, image.height))

    image_masked = Image.new('RGBa', (image.width, image.height))
    image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask.convert('L')))

    image_masked = image_masked.convert('RGBa')

    for radius, repeats in [(256, 1), (64, 1), (16, 2), (4, 4), (2, 2), (0, 1)]:
        blurred = image_masked.filter(ImageFilter.GaussianBlur(radius)).convert('RGBA')
        for _ in range(repeats):
            image_mod.alpha_composite(blurred)

    return image_mod.convert("RGB")

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
        denoising_strength: float = 0.05, 
        ddim_steps = 50, 
        target_img: torch.Tensor = None,
        mask: np.ndarray = None,
        mask_blur_ks: int = 0,
        mask_blur: float = 0,
        guidance_weight=None,
        guidance_schedule_func=lambda x: 0.1,
        **kwargs) -> torch.Tensor :
        ddim_sampler = GuidedDDIMSample(self)
        self.cond_stage_model.cuda()
        self.first_stage_model.cuda()
        c_text = self.get_learned_conditioning([c_text])
        uc_text = self.get_learned_conditioning([uc_text])
        cond = {"c_crossattn": [c_text]}
        uc_cond = {"c_crossattn": [uc_text]}
        init_latent = self.get_first_stage_encoding(self.encode_first_stage(img))
        if target_img is not None :
            target_latent = self.get_first_stage_encoding(self.encode_first_stage(target_img))
            if guidance_weight is not None :
                guidance_weight = torch.from_numpy(cv2.resize(guidance_weight, (target_latent.shape[3], target_latent.shape[2]))).half().cuda()
        else :
            target_latent = None
            
        if mask is not None :
            image_mask = np.array(Image.fromarray(mask * 255).convert('L').filter(ImageFilter.GaussianBlur(16)))
            #latent_mask = torch.from_numpy(cv2.resize(image_mask, (init_latent.shape[3], init_latent.shape[2]), interpolation=cv2.INTER_LINEAR)).cuda()
            image_mask = torch.from_numpy(image_mask).cuda()
            masked_image = img# * (1 - image_mask.float() / 255.0) + target_img * (image_mask.float() / 255.0)
            masked_image = np.clip(einops.rearrange(masked_image.cpu().squeeze(0).numpy(), 'c h w -> h w c') * 127.5 + 127.5, 0, 255).astype(np.uint8)
            masked_image = fill_mask_input(Image.fromarray(masked_image), Image.fromarray(image_mask.cpu().numpy()).convert('L'))
            masked_image = einops.rearrange(torch.from_numpy(np.array(masked_image)), 'h w c -> c h w').unsqueeze(0).float().cuda() / 127.5 - 1.0
            init_latent = self.get_first_stage_encoding(self.encode_first_stage(masked_image)) # <-- correct
            #if denoising_strength >= 0.99 :
            image_mask = image_mask.float() / 255.0
            latent_mask = torch.from_numpy(cv2.resize(image_mask.cpu().numpy(), (init_latent.shape[3], init_latent.shape[2]), interpolation=cv2.INTER_CUBIC)).cuda()
            init_latent = (1 - latent_mask) * init_latent + latent_mask * torch.randn_like(init_latent)
            # add conditioning
            image_cdt = get_inpainting_image_condition(self, masked_image, image_mask)
            cond["c_concat"] = [image_cdt]
            uc_cond["c_concat"] = [image_cdt]
            self.model.conditioning_key = 'hybrid'
        else :
            image_mask = None
            latent_mask = None
            self.model.conditioning_key = 'crossattn'
            # image_cdt = get_empty_image_condition(init_latent)
            # cond["c_concat"] = [image_cdt]
            # uc_cond["c_concat"] = [image_cdt]
            masked_image = torch.zeros_like(img)
        print(init_latent.shape)
        steps = ddim_steps
        t_enc = int(min(denoising_strength, 0.999) * steps)
        eta = 0
        print('steps', steps, 't_enc', t_enc)

        noise = torch.randn_like(init_latent)
        ddim_sampler.make_schedule(ddim_num_steps=steps, ddim_eta=eta, ddim_discretize="uniform", verbose=False)
        x1 = ddim_sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * int(init_latent.shape[0])).cuda(), noise=noise)

        decoded, last_gs = ddim_sampler.decode(x1, cond,t_enc,init_latent=init_latent,nmask=latent_mask,unconditional_guidance_scale=7,unconditional_conditioning=uc_cond,guidance=target_latent,guidance_schedule_func=guidance_schedule_func,guidance_weight=guidance_weight)
        if mask is not None :
            decoded = init_latent * (1 - latent_mask) + decoded * latent_mask

        # shape = (self.channels, h // 8, w // 8)
        # samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False,unconditional_guidance_scale=5,unconditional_conditioning=uc_cond, **kwargs)
        self.first_stage_model.cuda()
        x_samples = self.decode_first_stage(decoded)
        print(x_samples.shape)
        return torch.clip(x_samples, -1, 1), masked_image
    
    @torch.no_grad()
    def img2img_inpaint(
        self,
        image: Image.Image, 
        c_text: str, 
        uc_text: str, 
        denoising_strength: float = 0.05, 
        ddim_steps = 50, 
        reference_img: Image.Image = None,
        mask: Image.Image = None,
        mask_blur: int = 16,
        guidance_weight=None,
        guidance_schedule_func=lambda x: 0.1,
        **kwargs) -> Image.Image :
        ddim_sampler = GuidedDDIMSample(self)
        self.cond_stage_model.cuda()
        self.first_stage_model.cuda()
        c_text = self.get_learned_conditioning([c_text])
        uc_text = self.get_learned_conditioning([uc_text])
        cond = {"c_crossattn": [c_text]}
        uc_cond = {"c_crossattn": [uc_text]}
        # init_latent = self.get_first_stage_encoding(self.encode_first_stage(img))
        # if target_img is not None :
        #     target_latent = self.get_first_stage_encoding(self.encode_first_stage(target_img))
        #     if guidance_weight is not None :
        #         guidance_weight = torch.from_numpy(cv2.resize(guidance_weight, (target_latent.shape[3], target_latent.shape[2]))).half().cuda()
        # else :
        #     target_latent = None
            
        if mask is not None :
            image_mask = mask
            image_mask = image_mask.convert('L')
            image_mask = image_mask.filter(ImageFilter.GaussianBlur(mask_blur))
            latent_mask = image_mask
            if reference_img is None :
                image = fill_mask_input(image, latent_mask)
            else :
                image = Image.composite(reference_img, image, image_mask)
            image = np.array(image).astype(np.float32) / 127.5 - 1.0
            image = np.moveaxis(image, 2, 0)
            image = torch.from_numpy(image).cuda()[None]
            init_latent = self.get_first_stage_encoding(self.encode_first_stage(image))
            init_mask = latent_mask
            latmask = init_mask.convert('RGB').resize((init_latent.shape[3], init_latent.shape[2]))
            latmask = np.moveaxis(np.array(latmask, dtype=np.float32), 2, 0) / 255
            latmask = latmask[0]
            latmask = np.around(latmask)
            latmask = np.tile(latmask[None], (4, 1, 1))
            nmask = torch.asarray(latmask).to(init_latent.device).float()
            if reference_img is None :
                init_latent = (1 - nmask) * init_latent + nmask * torch.randn_like(init_latent)
                denoising_strength = 1
            image_cdt = get_inpainting_image_condition(self, image, image_mask)
            cond["c_concat"] = [image_cdt]
            uc_cond["c_concat"] = [image_cdt]
            self.model.conditioning_key = 'hybrid'
        else :
            image_mask = None
            latent_mask = None
            self.model.conditioning_key = 'crossattn'
            # image_cdt = get_empty_image_condition(init_latent)
            # cond["c_concat"] = [image_cdt]
            # uc_cond["c_concat"] = [image_cdt]
            masked_image = torch.zeros_like(image)
        print(init_latent.shape)
        steps = ddim_steps
        t_enc = int(min(denoising_strength, 0.999) * steps)
        eta = 0
        print('steps', steps, 't_enc', t_enc)

        noise = torch.randn_like(init_latent)
        ddim_sampler.make_schedule(ddim_num_steps=steps, ddim_eta=eta, ddim_discretize="uniform", verbose=False)
        x1 = ddim_sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * int(init_latent.shape[0])).cuda(), noise=noise)

        decoded, last_gs = ddim_sampler.decode(x1, cond,t_enc,init_latent=init_latent,nmask=nmask,unconditional_guidance_scale=7,unconditional_conditioning=uc_cond,guidance=reference_img,guidance_schedule_func=guidance_schedule_func,guidance_weight=guidance_weight)

        if mask is not None and last_gs > 0 :
            decoded = init_latent * (1 - nmask) + decoded * nmask

        # shape = (self.channels, h // 8, w // 8)
        # samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False,unconditional_guidance_scale=5,unconditional_conditioning=uc_cond, **kwargs)
        self.first_stage_model.cuda()
        x_samples = self.decode_first_stage(decoded)
        init_latent_decoded = self.decode_first_stage(init_latent).clip(-1, 1)
        return torch.clip(x_samples, -1, 1), image, init_latent_decoded
    
    @torch.no_grad()
    def img2img_latent_inpaint(
        self,
        warped_latent: torch.Tensor,
        reference_img: torch.Tensor,
        mask: Image.Image,
        c_text: str, 
        uc_text: str, 
        denoising_strength: float = 0.05, 
        ddim_steps = 50,
        mask_blur: int = 16,
        guidance_weight=None,
        guidance_schedule_func=lambda x: 0.1,
        **kwargs) -> Image.Image :
        ddim_sampler = GuidedDDIMSample(self)
        self.cond_stage_model.cuda()
        self.first_stage_model.cuda()
        c_text = self.get_learned_conditioning([c_text])
        uc_text = self.get_learned_conditioning([uc_text])
        cond = {"c_crossattn": [c_text]}
        uc_cond = {"c_crossattn": [uc_text]}
            
        image_mask = mask
        image_mask = image_mask.convert('L')
        image_mask = image_mask.filter(ImageFilter.GaussianBlur(mask_blur))
        latent_mask = image_mask

        image_from_warped_latent = self.decode_first_stage(warped_latent).clip(-1, 1)

        
        init_latent = self.get_first_stage_encoding(self.encode_first_stage(reference_img))
        init_mask = latent_mask
        latmask = init_mask.convert('RGB').resize((init_latent.shape[3], init_latent.shape[2]))
        latmask = np.moveaxis(np.array(latmask, dtype=np.float32), 2, 0) / 255
        latmask = latmask[0]
        latmask = np.around(latmask)
        latmask = np.tile(latmask[None], (4, 1, 1))
        nmask = torch.asarray(latmask).to(init_latent.device).float()
        init_latent = (1 - nmask) * warped_latent + nmask * init_latent
        init_latent_decoded = self.decode_first_stage(init_latent).clip(-1, 1)
        #image_cdt = get_inpainting_image_condition(self, image_from_warped_latent, image_mask)
        image_cdt = get_inpainting_image_condition(self, init_latent_decoded, image_mask)
        cond["c_concat"] = [image_cdt]
        uc_cond["c_concat"] = [image_cdt]
        self.model.conditioning_key = 'hybrid'
        
        steps = ddim_steps
        t_enc = int(min(denoising_strength, 0.999) * steps)
        eta = 0
        print('steps', steps, 't_enc', t_enc)

        noise = torch.randn_like(init_latent)
        ddim_sampler.make_schedule(ddim_num_steps=steps, ddim_eta=eta, ddim_discretize="uniform", verbose=False)
        x1 = ddim_sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * int(init_latent.shape[0])).cuda(), noise=noise)

        decoded, last_gs = ddim_sampler.decode(x1, cond,t_enc,init_latent=init_latent,nmask=nmask,unconditional_guidance_scale=7,unconditional_conditioning=uc_cond,guidance=reference_img,guidance_schedule_func=guidance_schedule_func,guidance_weight=guidance_weight)

        if mask is not None and last_gs > 0 :
            decoded = warped_latent * (1 - nmask) + decoded * nmask
        else :
            print('no gs')

        # shape = (self.channels, h // 8, w // 8)
        # samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False,unconditional_guidance_scale=5,unconditional_conditioning=uc_cond, **kwargs)

        x_samples = self.decode_first_stage(decoded)
        print(x_samples.shape)
        return torch.clip(x_samples, -1, 1), decoded, init_latent_decoded

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
