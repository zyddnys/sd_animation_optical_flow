
import cv2
import numpy as np
import torch
import einops
import numpy as np
from guided_ldm import create_model, load_state_dict
from PIL import Image
from hack import hack_everything
from booru_tagger import Tagger
import safetensors
import safetensors.torch

from pdcnet_of import warp_frame_latent as warp_frame_latent_pdcnet, warp_frame as warp_frame_pdcnet

def load_ldm_sd(model, path) :
    if path.endswith('.safetensor') :
        sd = safetensors.torch.load_file(path)
    else :
        sd = load_state_dict(path)
    print(sd.keys())
    model.load_state_dict(sd, strict = False)

def resize_keep_aspect(img: np.ndarray, size: int):
    ratio = size / min(img.shape[0], img.shape[1])
    new_width = round(img.shape[1] * ratio)
    new_height = round(img.shape[0] * ratio)
    img2 = cv2.resize(img, (new_width, new_height), cv2.INTER_LANCZOS4)
    return img2


class namespace:
    def __contains__(self,m):
        return hasattr(self, m)

class RAFT_2 :
    def __init__(self) -> None:
        import sys
        sys.path.append('../RAFT/core')
        from raft import RAFT
        from utils import flow_viz
        from utils.utils import InputPadder
        args = namespace()
        args.model = '../RAFT/models/raft-things.pth'
        args.small = False
        args.mixed_precision = False
        args.alternate_corr = False
        self.model = torch.nn.DataParallel(RAFT(args)).cuda()
        self.model.load_state_dict(torch.load(args.model))

    @torch.no_grad()
    def calc(self, img1, img2) :
        img1 = torch.from_numpy(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None].cuda()
        img2 = torch.from_numpy(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None].cuda()
        from utils.utils import InputPadder
        padder = InputPadder(img1.shape)
        image1, image2 = padder.pad(img1, img2)
        flow_low, flow_up = self.model(image1, image2, iters=20, test_mode=True)
        flo = flow_up[0].permute(1,2,0).cpu().numpy()
        return flo

def create_of_algo() :
    #model = RAFT_2()
    from pdcnet_of import create_of_algo
    model = create_of_algo('../DenseMatching/pre_trained_models/PDCNet_plus_m.pth.tar')
    return model


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def warp_frame(frame, flow) :
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    nextImg = cv2.remap(frame, flow, None, cv2.INTER_CUBIC)
    return nextImg

def warp_frame_latent(latent, flow) :
    latent = einops.rearrange(latent.cpu().numpy().squeeze(0), 'c h w -> h w c')
    lh, lw = latent.shape[:2]
    h, w = flow.shape[:2]
    latent = cv2.resize(latent, (w, h), interpolation=cv2.INTER_CUBIC)
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    nextImg = cv2.remap(latent, flow, None, cv2.INTER_CUBIC)
    nextImg = cv2.resize(nextImg, (lw, lh), interpolation=cv2.INTER_CUBIC)
    nextImg = torch.from_numpy(einops.rearrange(nextImg, 'h w c -> 1 c h w'))
    return nextImg

def of_calc(frame1, frame2, algo) :
    flow, confidence, log_confidence = algo.calc(frame1, frame2)
    h, w = flow.shape[:2]
    disp_x, disp_y = flow[:, :, 0], flow[:, :, 1]
    X, Y = np.meshgrid(np.linspace(0, w - 1, w),
                       np.linspace(0, h - 1, h))
    map_x = (X+disp_x).astype(np.float32)
    map_y = (Y+disp_y).astype(np.float32)
    map_x -= np.arange(w)
    map_y -= np.arange(h)[:,np.newaxis]
    v = np.sqrt(map_x*map_x+map_y*map_y)
    v[confidence < 0.9] = 0
    print('v.max()', v.max(), 'v.min()', v.min())
    return flow, confidence, v, log_confidence

def unsharp(img) :
    gaussian_3 = cv2.GaussianBlur(img, (0, 0), 2.0)
    unsharp_image = cv2.addWeighted(img, 1.3, gaussian_3, -0.3, 0)
    return unsharp_image

def img2img(model, model_tagger: Tagger, source_np_bgr_u8, denoise_strength, target_np_bgr_u8, *args, **kwargs) :
    blacklist = set()#set(['aqua_hair', 'headphones'])
    tags = model_tagger.label_cv2_bgr(source_np_bgr_u8)
    pos_prompt = ','.join([x for x in tags.keys() if x not in blacklist]).replace('_', ' ')
    pos_prompt = 'masterpiece,best quality,hatsune miku,' + pos_prompt
    frame_rgb = cv2.cvtColor(source_np_bgr_u8, cv2.COLOR_BGR2RGB)
    img_np = frame_rgb.astype(np.float32) / 127.5 - 1.
    img_torch = torch.from_numpy(img_np)
    img_torch = einops.rearrange(img_torch, 'h w c -> 1 c h w').cuda()
    if target_np_bgr_u8 is not None :
        target_img = einops.rearrange(torch.from_numpy(cv2.cvtColor(target_np_bgr_u8, cv2.COLOR_BGR2RGB)), 'h w c -> 1 c h w').cuda()
        target_img = target_img.float() / 127.5 - 1.
    else :
        target_img = None
    with torch.autocast(enabled=True, device_type = 'cuda') :
        img2, *_ = model.img2img(
            img_torch,
            pos_prompt,
            'worst quality, low quality, normal quality',
            denoise_strength,
            target_img = target_img,
            *args,
            **kwargs,
            )
    img2_np = (einops.rearrange(img2, '1 c h w -> h w c').cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
    del img2, img_torch, img_np
    return cv2.cvtColor(img2_np, cv2.COLOR_RGB2BGR)

def get_latent(model, source_np_bgr_u8) :
    frame_rgb = cv2.cvtColor(source_np_bgr_u8, cv2.COLOR_BGR2RGB)
    img_np = frame_rgb.astype(np.float32) / 127.5 - 1.
    img_torch = torch.from_numpy(img_np)
    img_torch = einops.rearrange(img_torch, 'h w c -> 1 c h w').cuda()
    model.first_stage_model.cuda()
    return model.get_first_stage_encoding(model.encode_first_stage(img_torch))

def decode_latent(model, latent) :
    return cv2.cvtColor((einops.rearrange(model.decode_first_stage(latent.cuda()).clip(-1, 1), '1 c h w -> h w c').cpu().numpy() * 127.5 + 127.5).astype(np.uint8), cv2.COLOR_RGB2BGR)

def confidence_to_mask(confidence, flow, dist, mask_aux) :
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = np.zeros((confidence.shape[0], confidence.shape[1]), dtype = np.uint8)
    mask[confidence < 0.9] = 255
    mask_aux.pixel_travel_dist = warp_frame_pdcnet(mask_aux.pixel_travel_dist, flow) + dist
    mask_aux.pixel_travel_dist[confidence < 0.9] = 0
    mask[mask_aux.pixel_travel_dist > mask_aux.thres] = 255
    mask_aux.pixel_travel_dist[mask_aux.pixel_travel_dist > mask_aux.thres] = 0
    mask = cv2.dilate(mask, kern)
    return mask

def run_inpainting(model_inpaint, model_tagger: Tagger, image: np.ndarray, reference: np.ndarray, mask: np.ndarray, denoising_strength, guidance_schedule_func) :
    tags = model_tagger.label_cv2_bgr(reference)
    blacklist = set([])
    pos_prompt = ','.join([x for x in tags.keys() if x not in blacklist]).replace('_', ' ')
    pos_prompt = 'masterpiece,best quality,hatsune miku,' + pos_prompt
    with torch.autocast(enabled = True, device_type = 'cuda') :
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        img, _, init_latent_decoded = model_inpaint.img2img_inpaint(
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
            c_text = pos_prompt,
            uc_text = 'worst quality, low quality, normal quality',
            denoising_strength = denoising_strength,
            reference_img = Image.fromarray(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)),
            mask = Image.fromarray(mask),
            mask_blur = 16,
            guidance_schedule_func = guidance_schedule_func
            )
    img = (einops.rearrange(img, '1 c h w -> h w c').cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
    init_latent_decoded = (einops.rearrange(init_latent_decoded, '1 c h w -> h w c').cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.cvtColor(init_latent_decoded, cv2.COLOR_RGB2BGR)

def mix_propagated_ai_frame(raw_ai_frame, warped_propagated_ai_frame, mask, propagated_pixel_weight = 1.0) :
    weights = np.zeros((raw_ai_frame.shape[0], raw_ai_frame.shape[1]), dtype = np.float32)
    weights[mask <= 127] = propagated_pixel_weight
    weights[mask > 127] = 1 - propagated_pixel_weight
    weights = weights[:, :, None]
    ai_frame = raw_ai_frame.astype(np.float32) * (1 - weights) + warped_propagated_ai_frame.astype(np.float32) * weights
    return np.clip(ai_frame, 0, 255).astype(np.uint8)

def generate_mask(cum_confidence: np.ndarray, log_confidence: np.ndarray, thres = 0.9) :
    mask = np.zeros((cum_confidence.shape[0], cum_confidence.shape[1]), dtype = np.uint8)
    mask[cum_confidence < thres] = 255
    log_confidence[cum_confidence < thres] = 0 # reset pixels to full confidence that will be inpainted
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.dilate(mask, kern), log_confidence

def run_exp(model, model_inpaint, model_tagger, name: str, denoising_strength, pixel_dist_thres, guidance_schedule_func) :
    name = f'pixel_warp-{name}'
    import os
    os.makedirs(f'out_{name}', exist_ok=True)
    print(name)
    video = cv2.VideoCapture('videos/out.mp4')
    frame = None
    last_frame = None
    last_converted_latent = None
    last_ai_frame = None
    of_algo = create_of_algo()
    ctr = -1
    mask_aux = None
    log_confidence = None
    while True :
        ctr += 1
        ret, frame = video.read()
        frame = cv2.resize(frame, (512, 768), interpolation=cv2.INTER_AREA)
        if log_confidence is None :
            log_confidence = np.zeros((frame.shape[0], frame.shape[1]), dtype = np.float64)
        if ctr % 1 != 0 :
            continue
        if not ret :
            break
        
        if last_frame is not None :
            flow, confidence, dist, cur_log_confidence = of_calc(last_frame, frame, of_algo)
            # distance visualization
            dist_p99 = np.percentile(dist, 99)
            dist_vis = (np.clip(dist / dist_p99, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(f'out_{name}/flow_dist_{ctr:06d}.png', dist_vis)
            # update cumlative confidence
            cur_log_confidence = cur_log_confidence.astype(np.float64)
            log_confidence = warp_frame_pdcnet(log_confidence, flow) + cur_log_confidence
            cum_confidence = np.exp(log_confidence)
            cum_confidence_u8 = np.clip(cum_confidence * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(f'out_{name}/pixel_confidence_{ctr:06d}.png', cum_confidence_u8)
            warped_ai_frame = warp_frame_pdcnet(last_ai_frame, flow)
            cv2.imwrite(f'out_{name}/warped_ai_{ctr:06d}.png', warped_ai_frame)
            mask, log_confidence = generate_mask(cum_confidence, log_confidence)
            raw_ai_frame, init_latent_decoded = run_inpainting(
                model_inpaint, 
                model_tagger, 
                warped_ai_frame, 
                frame, 
                mask,
                denoising_strength, 
                guidance_schedule_func
                )
            cv2.imwrite(f'out_{name}/ai_frame_{ctr:06d}.png', raw_ai_frame)
            ai_frame = mix_propagated_ai_frame(raw_ai_frame, warped_ai_frame, mask)
            last_ai_frame = ai_frame
            cv2.imwrite(f'out_{name}/mask_{ctr:06d}.png', mask)
            cv2.imwrite(f'out_{name}/init_latent_decoded_{ctr:06d}.png', init_latent_decoded)
        else :
            ai_frame = img2img(model, model_tagger, frame, 0.4, None)
        cv2.imwrite(f'out_{name}/raw_{ctr:06d}.png', frame)
        cv2.imwrite(f'out_{name}/converted_{ctr:06d}.png', ai_frame)
        last_frame = frame
        last_ai_frame = ai_frame
        if ctr >= 60 * 30 :
            break
    video.release()

def guidance_schedule(p) -> float :
    if p < 0.92 :
        return 1
    else :
        return 1

def main() :
    model = create_model('guided_ldm_inpaint4_v15.yaml').cuda()
    model_inpaint = create_model('guided_ldm_inpaint9_v15.yaml').cuda()
    hack_everything()
    load_ldm_sd(model, 'grapefruitHentaiModel_grapefruitv41.safetensors')
    load_ldm_sd(model_inpaint, 'grapefruitHentaiModel_grapefruitv41_inpainting.safetensors')
    tagger = Tagger()
    run_exp(model, model_inpaint, tagger, '60fps-ds0.4-fixseed-confidence-thres0.9_propagation-weight1.0', denoising_strength = 0.4, pixel_dist_thres = 160, guidance_schedule_func = guidance_schedule)
    #run_exp(model, model_inpaint, tagger, 'PDCNet-cubic-warponly', denoising_strength = 0.4, pixel_dist_thres = 1, guidance_schedule_func = guidance_schedule)

if __name__ == '__main__' :
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()
