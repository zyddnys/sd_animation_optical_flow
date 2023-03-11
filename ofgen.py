
import cv2
import numpy as np
import torch
import einops
import numpy as np
from guided_ldm import create_model, load_state_dict
from PIL import Image
from hack import hack_everything
from booru_tagger import Tagger
import argparse
import os

def load_ldm_sd(model, path) :
    sd = load_state_dict(path)
    model.load_state_dict(sd, strict = False)

def resize_keep_aspect(img: np.ndarray, size: int):
    ratio = size / min(img.shape[0], img.shape[1])
    new_width = round(img.shape[1] * ratio)
    new_height = round(img.shape[0] * ratio)
    img2 = cv2.resize(img, (new_width, new_height), cv2.INTER_LANCZOS4)
    return img2

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

def of_calc(frame1, frame2, guidance_weight_p) :
    # optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    # flow = optical_flow.calc(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), None)
    flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), None, 0.5, 5, 15, 3, 5, 1.2, 0)
    fx, fy = flow[:,:,0], flow[:,:,1]
    v = np.sqrt(fx*fx+fy*fy)
    return flow, v

def unsharp(img) :
    gaussian_3 = cv2.GaussianBlur(img, (0, 0), 2.0)
    unsharp_image = cv2.addWeighted(img, 1.12, gaussian_3, -0.12, 0)
    return unsharp_image

def img2img(model, model_tagger: Tagger, source_np_bgr_u8: np.ndarray, denoise_strength: float, target_np_bgr_u8: np.ndarray, *args, **kwargs) :
    blacklist = set()
    tags = model_tagger.label_cv2_bgr(source_np_bgr_u8)
    pos_prompt = ','.join([x for x in tags.keys() if x not in blacklist]).replace('_', ' ')
    pos_prompt = 'masterpiece,best quality,' + pos_prompt
    frame_rgb = cv2.cvtColor(source_np_bgr_u8, cv2.COLOR_BGR2RGB)
    img_np = frame_rgb.astype(np.float32) / 127.5 - 1.
    img_torch = torch.from_numpy(img_np)
    img_torch = einops.rearrange(img_torch, 'h w c -> 1 c h w').cuda()
    if target_np_bgr_u8 is not None :
        target_img = einops.rearrange(torch.from_numpy(cv2.cvtColor(target_np_bgr_u8, cv2.COLOR_BGR2RGB)), 'h w c -> 1 c h w').cuda()
        target_img = target_img.float() / 127.5 - 1.
    else :
        target_img = None
    with torch.autocast(enabled = True, device_type = 'cuda') :
        img2 = model.img2img(
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

def run_exp(model, model_tagger, video: str, save_dir: str, denoise_strength: float, guidance_schedule_func) :
    os.makedirs(save_dir, exist_ok=True)
    print(save_dir)
    video = cv2.VideoCapture(video)
    last_frame = None
    last_converted_frame = None
    frame_pred_ai = None
    ctr = -1
    while True :
        ctr += 1
        ret, frame = video.read()
        if ctr % 1 != 0 :
            continue
        if not ret :
            break
        #frame = cv2.resize(frame, (512, 768), interpolation=cv2.INTER_AREA)
        mean_dist = 0
        if last_frame is not None :
            flow, dist = of_calc(last_frame, frame, 99)
            mean_dist = np.mean(dist)
            print('mean_dist', mean_dist)
            frame_pred_ai = unsharp(warp_frame(last_converted_frame, flow))
            cv2.imwrite(f'{save_dir}/wrapped_{ctr:06d}.png', frame_pred_ai)
        else :
            dist = np.zeros((frame.shape[0], frame.shape[1]))
        guidance_schedule_func_aux = {
            'mean_of_dist': mean_dist,
            'dist_mat': dist,
            'denoise_strength': denoise_strength
        }
        img2_np = img2img(model, model_tagger, frame, denoise_strength, frame_pred_ai, guidance_schedule_func = guidance_schedule_func, guidance_schedule_func_aux = guidance_schedule_func_aux)
        cv2.imwrite(f'{save_dir}/raw_{ctr:06d}.png', frame)
        cv2.imwrite(f'{save_dir}/converted_{ctr:06d}.png', img2_np)
        last_frame = frame
        last_converted_frame = img2_np
    video.release()

def guidance_schedule(denoise_percentage, aux: dict) -> float | np.ndarray :
    denoise_strength = aux['denoise_strength']
    dist = aux['dist_mat']
    thres = 1.5 # 1.5 pixels away
    weights = np.ones((dist.shape[0], dist.shape[1]), dtype = np.float32)
    if denoise_percentage < 0.8 :
        weights *= 0.6
    else :
        weights *= 0.4
    weights[dist > thres] = 0.1
    return weights

def main(video: str, save_dir: str) :
    model = create_model('guided_ldm_v15.yaml').cuda()
    hack_everything()
    load_ldm_sd(model, 'ACertainModel.ckpt')
    tagger = Tagger()
    run_exp(model, tagger, video = video, save_dir = save_dir, denoise_strength = 0.4, guidance_schedule_func = guidance_schedule)

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description = 'experiment')
    parser.add_argument('-i', '--input', default='', type=str, help='Path to video files')
    parser.add_argument('-o', '--output', default='', type=str, help='Path to output')
    args = parser.parse_args()
    main(video = args.input, save_dir = args.output)
