
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
import math
import sys
sys.path.append('../DenseMatching')

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

#---------------------------------
# Copied from PySceneDetect
def mean_pixel_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return the mean average distance in pixel values between `left` and `right`.
    Both `left and `right` should be 2 dimensional 8-bit images of the same shape.
    """
    assert len(left.shape) == 2 and len(right.shape) == 2
    assert left.shape == right.shape
    num_pixels: float = float(left.shape[0] * left.shape[1])
    return (np.sum(np.abs(left.astype(np.int32) - right.astype(np.int32))) / num_pixels)


def estimated_kernel_size(frame_width: int, frame_height: int) -> int:
    """Estimate kernel size based on video resolution."""
    size: int = 4 + round(math.sqrt(frame_width * frame_height) / 192)
    if size % 2 == 0:
        size += 1
    return size

_kernel = None

def _detect_edges(lum: np.ndarray) -> np.ndarray:
    global _kernel
    """Detect edges using the luma channel of a frame.
    Arguments:
        lum: 2D 8-bit image representing the luma channel of a frame.
    Returns:
        2D 8-bit image of the same size as the input, where pixels with values of 255
        represent edges, and all other pixels are 0.
    """
    # Initialize kernel.
    if _kernel is None:
        kernel_size = estimated_kernel_size(lum.shape[1], lum.shape[0])
        _kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Estimate levels for thresholding.
    sigma: float = 1.0 / 3.0
    median = np.median(lum)
    low = int(max(0, (1.0 - sigma) * median))
    high = int(min(255, (1.0 + sigma) * median))

    # Calculate edges using Canny algorithm, and reduce noise by dilating the edges.
    # This increases edge overlap leading to improved robustness against noise and slow
    # camera movement. Note that very large kernel sizes can negatively affect accuracy.
    edges = cv2.Canny(lum, low, high)
    return cv2.dilate(edges, _kernel)

#---------------------------------

def detect_edges(frame):
    hue, sat, lum = cv2.split(cv2.cvtColor(frame , cv2.COLOR_BGR2HSV))
    return _detect_edges(lum)


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
    if propagated_pixel_weight < 0.001 :
        return raw_ai_frame
    weights = np.zeros((raw_ai_frame.shape[0], raw_ai_frame.shape[1]), dtype = np.float32)
    weights[mask <= 127] = propagated_pixel_weight
    weights[mask > 127] = 1 - propagated_pixel_weight
    weights = weights[:, :, None]
    # TODO: employ poisson blending
    ai_frame = raw_ai_frame.astype(np.float32) * (1 - weights) + warped_propagated_ai_frame.astype(np.float32) * weights
    return np.clip(ai_frame, 0, 255).astype(np.uint8)

def generate_mask(cum_confidence: np.ndarray, log_confidence: np.ndarray, thres = 0.8) :
    mask = np.zeros((cum_confidence.shape[0], cum_confidence.shape[1]), dtype = np.uint8)
    mask[cum_confidence < thres] = 255
    log_confidence[cum_confidence < thres] = 0 # reset pixels to full confidence that will be inpainted
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.dilate(mask, kern), log_confidence

def enhance_ai_frame(frame: np.ndarray) :
    return frame

def frame_generator(video_file, size, keep_every = 1, neighbor_frame_count = 0, th = 8.5, min_gap = -1, max_gap = -1) :
    video = cv2.VideoCapture(video_file)
    fps = video.get(cv2.CAP_PROP_FPS)
    if min_gap == -1:
        min_gap = int(10 * fps/30)
    else:
        min_gap = max(1, min_gap)
        min_gap = int(min_gap * fps/30)
        
    if max_gap == -1:
        max_gap = int(300 * fps/30)
    else:
        max_gap = max(10, max_gap)
        max_gap = int(max_gap * fps/30)
    ctr = -1
    ctr_valid = -1
    gap = 0
    key_edges = None
    while True :
        ctr += 1
        gap += 1
        ret, frame = video.read()
        if ret is None :
            break
        if ctr % keep_every != 0 :
            continue
        frame = cv2.resize(frame, size, interpolation = cv2.INTER_AREA)
        ctr_valid += 1
        if key_edges is None :
            key_edges = detect_edges(frame)
            yield frame, True, ctr_valid
        else :
            edges = detect_edges(frame)
            delta = mean_pixel_distance(edges, key_edges)
            _th = th * (max_gap - gap) / max_gap
            if _th < delta:
                key_edges = edges
                gap = 0
                yield frame, True, ctr_valid
            else :
                yield frame, False, ctr_valid
    video.release()

def run_exp(video, save_dir, model, model_inpaint, model_tagger, name: str, confidence_thres, propagated_pixel_weight, key_frame_thres, denoising_strength, guidance_schedule_func) :
    name = f'pixel_warp-{name}'
    import os
    os.makedirs(f'{save_dir}_{name}', exist_ok=True)
    print(name)
    frame = None
    reference_frame = None
    reference_ai_frame = None
    of_algo = create_of_algo()
    for current_frame, is_key_frame, counter in frame_generator(video, (512, 768), keep_every = 3, th = key_frame_thres) :
        if is_key_frame :
            # TODO: make key frames generated using reference
            ai_frame = img2img(model, model_tagger, current_frame, 0.4, None)
            reference_ai_frame = ai_frame
            reference_frame = current_frame
            # reference_frame, raw frame, raw ai frame, mixed ai frame, warped frame, masked warped ai frame
            vis = np.concatenate([reference_frame, current_frame, ai_frame, ai_frame, current_frame, ai_frame], axis = 1)
            cv2.imwrite(f'{save_dir}_{name}/vis_{counter:06d}.png', vis)
            cv2.imwrite(f'{save_dir}_{name}/pixel_confidence_{counter:06d}.png', np.ones((current_frame.shape[0], current_frame.shape[1]), dtype = np.uint8))
        else :
            flow, confidence, dist, cur_log_confidence = of_calc(reference_frame, current_frame, of_algo)
            confidence_u8 = np.clip(confidence * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(f'{save_dir}_{name}/pixel_confidence_{counter:06d}.png', confidence_u8)
            warped_ai_frame = warp_frame_pdcnet(reference_ai_frame, flow)
            mask, log_confidence = generate_mask(confidence, cur_log_confidence, thres = confidence_thres)
            raw_ai_frame, init_latent_decoded = run_inpainting(
                model_inpaint, 
                model_tagger, 
                warped_ai_frame, 
                current_frame, 
                mask,
                denoising_strength, 
                guidance_schedule_func
                )
            ai_frame = mix_propagated_ai_frame(raw_ai_frame, warped_ai_frame, mask, propagated_pixel_weight)
            ai_frame = enhance_ai_frame(ai_frame)
            # reference_frame, raw frame, raw ai frame, mixed ai frame, warped frame, masked warped ai frame
            masked_warped_ai_frame = np.copy(warped_ai_frame)
            masked_warped_ai_frame[mask > 127] = np.array([0, 0, 255]) # mask inpainted region with color red
            vis = np.concatenate([reference_frame, current_frame, raw_ai_frame, ai_frame, warped_ai_frame, masked_warped_ai_frame], axis = 1)
            cv2.imwrite(f'{save_dir}_{name}/vis_{counter:06d}.png', vis)
        cv2.imwrite(f'{save_dir}_{name}/converted_{counter:06d}.png', ai_frame)

def guidance_schedule(p) -> float :
    if p < 0.92 :
        return 1
    else :
        return 1

def main(video, save_dir) :
    model = create_model('guided_ldm_inpaint4_v15.yaml').cuda()
    model_inpaint = create_model('guided_ldm_inpaint9_v15.yaml').cuda()
    hack_everything()
    load_ldm_sd(model, 'grapefruitHentaiModel_grapefruitv41.safetensors')
    load_ldm_sd(model_inpaint, 'grapefruitHentaiModel_grapefruitv41_inpainting.safetensors')
    tagger = Tagger()
    run_exp(
        video,
        save_dir,
        model, model_inpaint, tagger,
        '20fps_keyframethres24_ds0.4_fixseed_conf-thres0.95_ppw0.0', 
        denoising_strength = 0.4, 
        confidence_thres = 0.95, 
        propagated_pixel_weight = 0.0,
        key_frame_thres = 24,
        guidance_schedule_func = guidance_schedule
        )
    #run_exp(model, model_inpaint, tagger, 'PDCNet-cubic-warponly', denoising_strength = 0.4, pixel_dist_thres = 1, guidance_schedule_func = guidance_schedule)

if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser(description = 'experiment')
    parser.add_argument('-i', '--input', default='', type=str, help='Path to video files')
    parser.add_argument('-o', '--output', default='', type=str, help='Path to output')
    args = parser.parse_args()
    main(video = args.input, save_dir = args.output)
