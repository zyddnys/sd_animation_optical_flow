
from functools import cached_property
import glob
import pickle
import shutil
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
import einops
import numpy as np
from controlnet import SingleControlNet
from guided_ldm import create_model, load_state_dict
from PIL import Image
from hack import hack_everything
from booru_tagger import Tagger
import safetensors
import safetensors.torch
import math
import sys
import os
from tqdm import tqdm
import copy
sys.path.append('../DenseMatching')

from pdcnet_of import warp_frame_latent as warp_frame_latent_pdcnet, warp_frame as warp_frame_pdcnet

def load_ldm_sd(model, path) :
    if path.endswith('.safetensor') :
        sd = safetensors.torch.load_file(path)
    else :
        sd = load_state_dict(path)
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
    # frame1 = cv2.bilateralFilter(frame1, 7, 20, 20)
    # frame2 = cv2.bilateralFilter(frame2, 7, 20, 20)
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
    # v = v / np.percentile(v,guidance_weight_p)
    # v = np.clip(v, 0, 1)
    # thres = np.percentile(v, 50)
    # confidence = np.ones_like(v)
    # confidence[v > thres] = 0
    print('v.max()', v.max(), 'v.min()', v.min())
    return flow, confidence, v, log_confidence
    #return cv2.calcOpticalFlowFarneback(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), None, 0.5, 5, 15, 3, 5, 1.2, 0)

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


def img2img(model, model_tagger: Tagger, source_np_bgr_u8, denoise_strength, target_np_bgr_u8, override_tagger_frame = None, *args, **kwargs) :
    blacklist = set()#set(['aqua_hair', 'headphones'])
    if override_tagger_frame is not None :
        tags = model_tagger.label_cv2_bgr(override_tagger_frame)
    else :
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
        img2, _, kv_hist = model.img2img(
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
    return cv2.cvtColor(img2_np, cv2.COLOR_RGB2BGR), kv_hist

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
    #mask = ((1 - confidence) * 240).astype(np.uint8)
    mask = cv2.dilate(mask, kern)
    # TODO: counter mask
    return mask
    mask_aux.counter += 1
    if mask_aux.counter % 2 == 0 :
        return mask_aux.even
    else :
        return mask_aux.odd

def run_inpainting(
        model_inpaint, 
        model_tagger: Tagger, 
        image: np.ndarray, 
        reference: np.ndarray, 
        mask: np.ndarray, 
        denoising_strength, 
        guidance_schedule_func, 
        tagger_frame: np.ndarray = None, 
        *args, 
        **kwargs) :
    if tagger_frame is not None :
        tags = model_tagger.label_cv2_bgr(tagger_frame)
    else :
        tags = model_tagger.label_cv2_bgr(reference)
    blacklist = set([])
    pos_prompt = ','.join([x for x in tags.keys() if x not in blacklist]).replace('_', ' ')
    pos_prompt = 'masterpiece,best quality,' + pos_prompt
    with torch.autocast(enabled = True, device_type = 'cuda') :
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        img, _, init_latent_decoded, new_history, kv_hist_denoise = model_inpaint.img2img_inpaint(
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
            c_text = pos_prompt,
            uc_text = 'worst quality, low quality, normal quality',
            denoising_strength = denoising_strength,
            reference_img = Image.fromarray(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)) if reference is not None else None,
            mask = Image.fromarray(mask),
            mask_blur = 4,
            guidance_schedule_func = guidance_schedule_func,
            *args,
            **kwargs
            )
    img = (einops.rearrange(img, '1 c h w -> h w c').cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
    init_latent_decoded = (einops.rearrange(init_latent_decoded, '1 c h w -> h w c').cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.cvtColor(init_latent_decoded, cv2.COLOR_RGB2BGR), new_history, kv_hist_denoise

def create_mask_aux(h, w, pixel_dist_thres) :
    aux = namespace()
    # grid_size = 16
    # aux.odd = np.zeros((h, w), dtype = np.uint8)
    # for i in range(aux.odd.shape[0] // grid_size) :
    #     for j in range(aux.odd.shape[1] // grid_size) :
    #         if i % 2 == j % 2:
    #             aux.odd[i * grid_size: (i + 1) * grid_size, j * grid_size: (j + 1) * grid_size] = 240
    # aux.even = 240 - aux.odd
    # aux.counter = 0
    aux.pixel_travel_dist = np.zeros((h, w), dtype = np.float32)
    aux.thres = pixel_dist_thres
    return aux

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
        if ctr >= 60 * 60 :
            break
    video.release()

class VideoData :
    def __init__(self, path: str, size: Tuple[int, int], workspace_dir: str, keep_every: int = 1, max_len_sec: int = -1) -> None:
        self.workspace_dir = workspace_dir
        os.makedirs(workspace_dir, exist_ok=True)
        if os.path.exists(os.path.join(workspace_dir, 'raw-frames')) :
            num_frames = len(glob.glob(os.path.join(workspace_dir, 'raw-frames', '*.png')))
            video = cv2.VideoCapture(path)
            l = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video.release()
            if num_frames * keep_every >= l - keep_every :
                self.num_frames = num_frames
                return
        os.makedirs(os.path.join(workspace_dir, 'raw-frames'), exist_ok = True)
        os.makedirs(os.path.join(workspace_dir, 'ai-frames'), exist_ok = True)
        os.makedirs(os.path.join(workspace_dir, 'pdcnet'), exist_ok = True)
        os.makedirs(os.path.join(workspace_dir, 'crossattn'), exist_ok = True)
        os.makedirs(os.path.join(workspace_dir, 'seed'), exist_ok = True)
        video = cv2.VideoCapture(path)
        self.fps = video.get(cv2.CAP_PROP_FPS) / keep_every
        if max_len_sec == -1 :
            target_frames = 100000000000
        else :
            target_frames = self.fps * max_len_sec
        ctr = -1
        ctr_valid = -1
        print('Extracting frames ...')
        while True :
            ctr += 1
            ret, frame = video.read()
            if ret is None :
                break
            if ctr % keep_every != 0 :
                continue
            ctr_valid += 1
            dst = os.path.join(workspace_dir, 'raw-frames', f'{ctr_valid:05d}.png')
            if not os.path.exists(dst) :
                frame = cv2.resize(frame, size, interpolation = cv2.INTER_AREA)
                cv2.imwrite(dst, frame)
            if ctr_valid >= target_frames :
                break
        self.num_frames = ctr_valid
        self.size = size
        video.release()
        self.kv_hist_map = {}

    def get_raw_frame(self, n) :
        assert n < self.num_frames
        filename = os.path.join(self.workspace_dir, 'raw-frames', f'{n:05d}.png')
        return cv2.imread(filename)
    
    def get_ai_frame(self, n) :
        assert n < self.num_frames
        filename = os.path.join(self.workspace_dir, 'ai-frames', f'{n:05d}.png')
        if os.path.exists(filename) :
            return cv2.imread(filename)
        else :
            return None

    def generated(self, n) :
        filename = os.path.join(self.workspace_dir, 'ai-frames', f'{n:05d}.png')
        return os.path.exists(filename)

    def put_ai_frame(self, n, frame: np.ndarray) :
        assert n < self.num_frames
        filename = os.path.join(self.workspace_dir, 'ai-frames', f'{n:05d}.png')
        cv2.imwrite(filename, frame)

    @cached_property
    def size_hw(self) :
        return (self.size[1], self.size[0])
    
    def key_frames(self, th = 48, min_gap = -1, max_gap = -1) :
        if min_gap == -1:
            min_gap = int(10 * self.fps / 30)
        else:
            min_gap = max(1, min_gap)
            min_gap = int(min_gap * self.fps / 30)
            
        if max_gap == -1:
            max_gap = int(300 * self.fps / 30)
        else:
            max_gap = max(10, max_gap)
            max_gap = int(max_gap * self.fps / 30)
        gap = 0
        key_edges = None
        for i in range(self.num_frames) :
            frame = self.get_raw_frame(i)
            if key_edges is None :
                key_edges = detect_edges(frame)
                yield frame, i
            else :
                edges = detect_edges(frame)
                delta = mean_pixel_distance(edges, key_edges)
                _th = th * (max_gap - gap) / max_gap
                if _th < delta:
                    key_edges = edges
                    gap = 0
                    yield frame, i

    def put_kv(self, frame_idx: int, kv) :
        filename = os.path.join(self.workspace_dir, 'crossattn', f'{frame_idx:05d}.bin')
        with open(filename, 'wb') as fp :
            pickle.dump(kv, fp)

    def get_kv(self, frame_idx: int) :
        filename = os.path.join(self.workspace_dir, 'crossattn', f'{frame_idx:05d}.bin')
        with open(filename, 'rb') as fp :
            return pickle.load(fp)
    
    def remove_kv(self, frame_idx: int) :
        filename = os.path.join(self.workspace_dir, 'crossattn', f'{frame_idx:05d}.bin')
        os.remove(filename)

class VideoFrameIndices :
    """
    Record a set of video frame indices into a VideoData instance
    """
    def __init__(self, indices = []) -> None:
        self.indices = list(sorted(set(indices)))

    @staticmethod
    def from_n(n) :
        idx = list(range(n))
        return VideoFrameIndices(idx)
    
    def conv_indices(self, kernel_size: int = 17, stride: int = 8, dilation = 1) :
        idx = 0
        while idx < len(self.indices) :
            yield VideoFrameIndices(self.indices[idx: idx + kernel_size][0::dilation])
            idx += stride

    def remove(self, other) :
        self.indices = set(self.indices)
        self.indices.difference_update(set(other.indices))
        self.indices = list(sorted(set(self.indices)))

    def add(self, other) :
        if isinstance(other, int) :
            other = VideoFrameIndices([other])
        self.indices = set(self.indices)
        self.indices.update(set(other.indices))
        self.indices = list(sorted(set(self.indices)))

    def adjacent_frames(self, idx: int, n: int) :
        if len(self) <= n :
            return self
        r = None
        min_dist = 100000000000
        for i in range(0, len(self) - n) :
            candidate = self.indices[i: i + n]
            dist = np.sum(np.abs(np.asarray(candidate) - idx))
            if dist < min_dist :
                min_dist = dist
                r = candidate
            # if candidate[n // 2] >= idx :
            #     r = candidate
            #     break
            # if n == 1 :
            #     if self.indices[i] <= idx and self.indices[i + 1] >= idx :
            #         r = candidate
            #         break
            # else :
            #     if candidate[n // 2 - 1] <= idx and candidate[n // 2] >= idx :
            #         r = candidate
            #         break
        if r is None :
            r = candidate
        return VideoFrameIndices(r)

    def __len__(self) :
        return len(self.indices)
    
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class PDCNetAux :
    def __init__(self, pdcnet_model, workspace_dir: str, batch_size: int = 16, device = torch.device('cuda:0')) -> None:
        self.workspace_dir = workspace_dir
        self.cached_pair = set()
        self.batch_size = batch_size
        self.device = device
        self.pdcnet_model = pdcnet_model.to(device)
        self.pair_dir = os.path.join(workspace_dir, 'pdcnet')
        if os.path.exists(self.pair_dir) :
            for f in glob.glob(os.path.join(self.pair_dir, '*.npy')) :
                name = os.path.split(f)[-1].split('.')[0]
                [s, t] = name.split('-')
                s, t = int(s), int(t)
                self.cached_pair.add((s, t))
            if len(self.cached_pair) > 0 :
                print(f'[PDCNetAux] Loaded {len(self.cached_pair)} existing pairs')

    def purge(self) :
        self.cached_pair = set()
        for f in glob.glob(os.path.join(self.pair_dir, '*.npy')) :
            os.remove(f)

    def load_cached(self, s, t) :
        assert (s, t) in self.cached_pair
        filename = os.path.join(self.pair_dir, f'{s:05d}-{t:05d}.npy')
        return np.load(filename)

    def calcualte_single(self, video: VideoData, s, t) :
        if (s, t) in self.cached_pair :
            return self.load_cached(s, t)
        else :
            ret = np.zeros((1, 1, *video.size_hw, 3), dtype = np.float32)
            self.calculate_given_pairs(video, [(s, t)], {s: 0}, {t: 0}, ret)
            self.cached_pair.add((s, t))
            return ret[0, 0]

    def calculate_given_pairs(self, video: VideoData, to_calculate_pairs: List[Tuple[int, int]], s2i_map: Dict[int, int], t2i_map: Dict[int, int], ret: np.ndarray) :
        for pair_batch in chunks(to_calculate_pairs, self.batch_size) :
            bs = len(pair_batch)
            inp_source = np.zeros((bs, *video.size_hw, 3), dtype = np.uint8)
            inp_target = np.zeros((bs, *video.size_hw, 3), dtype = np.uint8)
            for i, (s, t) in enumerate(pair_batch) :
                inp_source[i] = cv2.cvtColor(video.get_raw_frame(s), cv2.COLOR_BGR2RGB)
                inp_target[i] = cv2.cvtColor(video.get_raw_frame(t), cv2.COLOR_BGR2RGB)
            #with torch.autocast(enabled = True, device_type = 'cuda') :
            flow_est, confidence = self.pdcnet_model.calc_batch(torch.from_numpy(inp_source).to(self.device), torch.from_numpy(inp_target).to(self.device))
            for i, (s, t) in enumerate(pair_batch) :
                si = s2i_map[s]
                ti = t2i_map[t]
                ret[si, ti, :, :, 0: 2] = flow_est[i]
                ret[si, ti, :, :, 2] = confidence[i]
                np.save(os.path.join(self.pair_dir, f'{s:05d}-{t:05d}.npy'), ret[si, ti])

    def calculate_multiple_to_one(self, video: VideoData, source_indices: VideoFrameIndices, target_index: int) -> np.ndarray :
        """
        returns [N (source), 1, 3, H, W]
        """
        to_calculate_pairs: List[Tuple[int, int]] = []
        n = len(source_indices)
        s2i_map = {}
        t2i_map = {target_index: 0}
        for i, s in enumerate(source_indices.indices) :
            s2i_map[s] = i
            if s != target_index :
                if (s, target_index) not in self.cached_pair :
                    to_calculate_pairs.append((s, target_index))
        ret = np.zeros((n, 1, *video.size_hw, 3), dtype = np.float32)
        self.calculate_given_pairs(video, to_calculate_pairs, s2i_map, t2i_map, ret)
        for i, s in enumerate(source_indices.indices) :
            if s != target_index :
                if (s, target_index) in self.cached_pair :
                    ret[i, 0] = self.load_cached(s, target_index)
            elif s == target_index :
                ret[i, 0, :, :, 0: 2] = 0
                ret[i, 0, :, :, 2] = 1
        self.cached_pair.update(to_calculate_pairs)
        return ret

    def calculate_pairwise(self, video: VideoData, indices: VideoFrameIndices) -> np.ndarray :
        """
        returns [N (source), N (target), 3, H, W]
        """
        n = len(indices)
        to_calculate_pairs: List[Tuple[int, int]] = []
        s2i_map = {}
        t2i_map = {}
        for i, s in enumerate(indices.indices) :
            s2i_map[s] = i
            for j, t in enumerate(indices.indices) :
                t2i_map[t] = j
                if s != t :
                    if (s, t) not in self.cached_pair :
                        to_calculate_pairs.append((s, t))
        ret = np.zeros((n, n, *video.size_hw, 3), dtype = np.float32)
        self.calculate_given_pairs(video, to_calculate_pairs, s2i_map, t2i_map, ret)
        for i, s in enumerate(indices.indices) :
            for j, t in enumerate(indices.indices) :
                if s != t :
                    if (s, t) in self.cached_pair :
                        ret[i, j] = self.load_cached(s, t)
                elif s == t :
                    ret[i, j, :, :, 0: 2] = 0
                    ret[i, j, :, :, 2] = 1
        self.cached_pair.update(to_calculate_pairs)
        return ret

def KeyframeConv(pdcnet: PDCNetAux, workspace: str, video: VideoData, frames: VideoFrameIndices, kernel_size: int = 17, stride: int = 8, dilation = 2) -> VideoFrameIndices :
    if os.path.exists(workspace) :
        files = glob.glob(os.path.join(workspace, '*.png'))
        idx_file = [int(os.path.split(x)[-1].split('.')[0]) for x in files]
        if len(idx_file) > 0 :
            return VideoFrameIndices(idx_file)
    else :
        os.makedirs(workspace)
    ret = set()
    for local_indices in tqdm(frames.conv_indices(kernel_size, stride, dilation)) :
        flow_mat = pdcnet.calculate_pairwise(video, local_indices)
        confidence_values = einops.reduce(flow_mat[:, :, :, :, 2], 's t h w -> s', 'sum')
        print('confidence_values', confidence_values)
        idx = local_indices.indices[np.argmax(confidence_values)]
        print(' local_indices', local_indices.indices, 'becomes', idx)
        ret.add(idx)
    for idx in ret :
        dst = os.path.join(workspace, f'{idx:05d}.png')
        cv2.imwrite(dst, video.get_raw_frame(idx))
    return VideoFrameIndices(ret)

def merge_images(base_image: np.ndarray, second_image: np.ndarray, mask: np.ndarray, method = 'naive') :
    if method == 'naive' :
        base_image = np.copy(base_image)
        mask2 = (mask / 255).astype(np.uint8)[:, :, None]
        base_image = base_image * (1 - mask2) + second_image * (mask2)
        return base_image
    elif method == 'poisson' :
        base_image = np.copy(base_image)
        mask2 = (mask / 255).astype(np.uint8)[:, :, None]
        base_image = np.copy(base_image * (1 - mask2) + second_image * (mask2))
        # kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # mask = cv2.dilate(mask, kern)
        return cv2.seamlessClone(second_image, base_image, mask, (second_image.shape[1] // 2, second_image.shape[0] // 2), cv2.NORMAL_CLONE)

def draw_mask(mat: np.ndarray, mask) :
    a = mat.astype(np.float32)
    a[mask < 127] *= np.array([0.3, 0.3, 1])
    a = a.astype(np.uint8)
    return a

def merge_denoise_history(
        workspace: str,
        video: VideoData,
        cur_frame_idx: int,
        pdcnet: PDCNetAux,
        conf_thres: float,
        denoise_history = [],
        denoise_history_ref_frames = []
    ) :
    return None
    if denoise_history and denoise_history_ref_frames :
        dst_dir = os.path.join(workspace, 'denoise_hist')
        os.makedirs(dst_dir, exist_ok = True)
        denoise_history = denoise_history[0]
        ref_frame = denoise_history_ref_frames[0]
        flow = pdcnet.calcualte_single(video, ref_frame, cur_frame_idx)
        keys = list(denoise_history.keys())
        for k in keys :
            cv2.imwrite(os.path.join(dst_dir, f'{cur_frame_idx:05d}-{k:02d}-0.png'), cv2.cvtColor((denoise_history[k] * 127.5 + 127.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
            denoise_history[k] = warp_frame_pdcnet(denoise_history[k], flow[:, :, 0: 2])
            cv2.imwrite(os.path.join(dst_dir, f'{cur_frame_idx:05d}-{k:02d}-1.png'), cv2.cvtColor((denoise_history[k] * 127.5 + 127.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
            denoise_history[str(k) + '_confidence'] = (flow[:, :, 2] > conf_thres).astype(np.float32)
        return denoise_history
    else :
        return None

def generate_ai_frame_with_ref_warp_and_inpaint(
        level, 
        workspace, 
        model_paint, 
        model_inpainting, 
        model_tagger: Tagger, 
        video: VideoData, 
        frame_idx: int, 
        reference_frames: VideoFrameIndices, 
        flow_mat: np.ndarray, 
        pdc: PDCNetAux,
        denoise_history = [],
        denoise_history_ref_frames = [],
        thres: float = 0.5, 
        ds: float = 0.6,
        guidance_schedule_func = None
        ) :
    vis_dir = os.path.join(workspace, 'render_vis', f'u{level:02d}')
    os.makedirs(vis_dir, exist_ok = True)
    flow_mat[:, :, :, :, 2] = (flow_mat[:, :, :, :, 2] > thres).astype(np.float32) # only confidence>thres regions are considered
    mask = np.zeros((flow_mat.shape[2], flow_mat.shape[3]), dtype = np.uint8)
    ret_frame = None
    vis = np.zeros((flow_mat.shape[2] * 4, flow_mat.shape[3] * (len(reference_frames.indices) + 1), 3), dtype = np.uint8)
    for i in range(len(reference_frames)) :
        confidence_values = einops.reduce(flow_mat[:, :, :, :, 2], 's t h w -> s', 'sum')
        ref_rel_idx = np.argmax(confidence_values)
        ref_idx = reference_frames.indices[ref_rel_idx]
        reference_ai_frame = video.get_ai_frame(ref_idx)
        assert reference_ai_frame is not None
        warped_frame = warp_frame_pdcnet(reference_ai_frame, flow_mat[ref_rel_idx, 0, :, :, 0: 2])
        assert warped_frame is not None
        last_confidence_map = flow_mat[ref_rel_idx, 0, :, :, 2]
        cur_mask = (last_confidence_map * 255).astype(np.uint8)
        mask = cv2.bitwise_or(mask, cur_mask)
        if ret_frame is None :
            ret_frame = np.copy(warped_frame)
        else :
            ret_frame = merge_images(ret_frame, warped_frame, cur_mask, method = 'naive')
        #ret_frame[mask < 127] = 0
        vis[ret_frame.shape[0] * 0: ret_frame.shape[0] * 1, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]] = video.get_raw_frame(frame_idx)
        vis[ret_frame.shape[0] * 1: ret_frame.shape[0] * 2, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]] = ret_frame
        vis[ret_frame.shape[0] * 1: ret_frame.shape[0] * 2, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]] = draw_mask(vis[ret_frame.shape[0] * 1: ret_frame.shape[0] * 2, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]], mask)
        #vis[ret_frame.shape[0] * 1: ret_frame.shape[0] * 2, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]][mask < 127] = np.array([0, 0, 255])#draw_mask(vis[ret_frame.shape[0] * 1: ret_frame.shape[0] * 2, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]], mask)
        vis[ret_frame.shape[0] * 2: ret_frame.shape[0] * 3, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]] = warped_frame
        #vis[ret_frame.shape[0] * 2: ret_frame.shape[0] * 3, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]][cur_mask < 127] = np.array([0, 0, 255])#draw_mask(vis[ret_frame.shape[0] * 2: ret_frame.shape[0] * 3, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]], cur_mask)
        vis[ret_frame.shape[0] * 2: ret_frame.shape[0] * 3, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]] = draw_mask(vis[ret_frame.shape[0] * 2: ret_frame.shape[0] * 3, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]], cur_mask)
        vis[ret_frame.shape[0] * 3: ret_frame.shape[0] * 4, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]] = video.get_raw_frame(ref_idx)
        flow_mat[:, 0, :, :, 2] -= last_confidence_map[None, :, :] # subtract regions already warped, make them zero
        flow_mat[:, 0, :, :, 2] = np.clip(flow_mat[:, 0, :, :, 2], 0, 1)
    original_frame = video.get_raw_frame(frame_idx)
    mask2 = 255 - mask
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask2 = cv2.dilate(mask2, kern)
    cnets = [
        SingleControlNet(
            weight = 0.7,
            model = 'hed',
            args = {},
            condition = original_frame,
            guidance_start = 0,
            guidance_end = 1
        ),
        SingleControlNet(
            weight = 0.3,
            model = 'canny',
            args = {
                'low_threshold': 100,
                'high_threshold': 200
            },
            condition = original_frame,
            guidance_start = 0,
            guidance_end = 0.9
        )
    ]
    dnhist = merge_denoise_history(workspace, video, frame_idx, pdc, thres, denoise_history, denoise_history_ref_frames)
    #ans, _, _ = run_inpainting(model_inpainting, model_tagger, ret_frame, original_frame, mask2, ds, lambda p: 0 if p > stop_p else 1, control_nets = cnets)
    ans, _, new_history, kv_hist_denoise = run_inpainting(model_paint, model_tagger, ret_frame, original_frame, mask2, ds, control_nets = cnets, history_guidance = dnhist, guidance_schedule_func = guidance_schedule_func)
    i = len(reference_frames)
    vis[ret_frame.shape[0] * 1: ret_frame.shape[0] * 2, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]] = ans
    cv2.imwrite(os.path.join(vis_dir, f'{frame_idx:05d}.png'), vis)
    return ans, new_history, kv_hist_denoise

def generate_ai_frame_with_ref_self_attn(
        level, 
        workspace, 
        model_paint, 
        model_inpainting, 
        model_tagger: Tagger, 
        video: VideoData, 
        frame_idx: int, 
        reference_frames: VideoFrameIndices, 
        flow_mat: np.ndarray, 
        pdc: PDCNetAux,
        denoise_history = [],
        denoise_history_ref_frames = [],
        thres: float = 0.5, 
        ds: float = 0.6,
        guidance_schedule_func = None
        ) :
    vis_dir = os.path.join(workspace, 'render_vis', f'u{level:02d}')
    os.makedirs(vis_dir, exist_ok = True)
    _, _, h, w, _ = flow_mat.shape
    all_frames = np.zeros((h, w * (len(reference_frames) + 1), 3), dtype = np.uint8)
    all_frames[:, 0 * w: 1 * w] = video.get_raw_frame(frame_idx)
    for i, idx in enumerate(reference_frames.indices) :
        all_frames[:, (i + 1) * w: (i + 2) * w] = video.get_ai_frame(idx)
    mask = np.zeros((h, w * (len(reference_frames) + 1)), dtype = np.uint8)
    mask[:, 0 * w: 1 * w] = 255
    all_frames2 = np.copy(all_frames)
    all_frames2[:, 0 * w: 1 * w] = 0
    #cv2.imwrite(os.path.join(vis_dir, f'{frame_idx:05d}.png'), all_frames)
    cnets = [
        SingleControlNet(
            weight = 0.7,
            model = 'hed',
            args = {},
            condition = all_frames,
            guidance_start = 0,
            guidance_end = 1
        ),
        SingleControlNet(
            weight = 0.3,
            model = 'canny',
            args = {
                'low_threshold': 100,
                'high_threshold': 200
            },
            condition = all_frames,
            guidance_start = 0,
            guidance_end = 1
        )
    ]
    dnhist = merge_denoise_history(workspace, video, frame_idx, pdc, thres, denoise_history, denoise_history_ref_frames)
    ans, decoded, new_history, kv_hist_denoise = run_inpainting(model_paint, model_tagger, all_frames, None, mask, ds, tagger_frame = all_frames[:, 0 * w: 1 * w], control_nets = cnets, history_guidance = dnhist, guidance_schedule_func = guidance_schedule_func)
    #ans, _, new_history = run_inpainting(model_inpainting, model_tagger, all_frames2, all_frames, mask, ds, lambda p: 0 if p > stop_p else 1, control_nets = cnets)
    cv2.imwrite(os.path.join(vis_dir, f'{frame_idx:05d}.png'), all_frames)
    return ans[:, : w], new_history, kv_hist_denoise

def generate_ai_frame_with_ref_both(
        level, 
        workspace, 
        model_paint, 
        model_inpainting, 
        model_tagger: Tagger, 
        video: VideoData, 
        frame_idx: int, 
        reference_frames: VideoFrameIndices, 
        flow_mat: np.ndarray, 
        pdc: PDCNetAux,
        denoise_history = [],
        denoise_history_ref_frames = [],
        thres: float = 0.5, 
        ds: float = 0.6,
        guidance_schedule_func = None
        ) :
    add_prev_frame_as_reference = False
    if frame_idx > 0 and video.get_ai_frame(frame_idx - 1) is not None and not ((frame_idx - 1) in reference_frames.indices) :
        add_prev_frame_as_reference = True
    vis_dir = os.path.join(workspace, 'render_vis', f'u{level:02d}')
    os.makedirs(vis_dir, exist_ok = True)
    _, _, h, w, _ = flow_mat.shape
    flow_mat[:, :, :, :, 2] = (flow_mat[:, :, :, :, 2] > thres).astype(np.float32) # only confidence>thres regions are considered
    mask = np.zeros((flow_mat.shape[2], flow_mat.shape[3]), dtype = np.uint8)
    ret_frame = None
    vis = np.zeros((flow_mat.shape[2] * 4, flow_mat.shape[3] * (len(reference_frames.indices) + 1 + (1 if add_prev_frame_as_reference else 0)), 3), dtype = np.uint8)
    for i in range(len(reference_frames)) :
        confidence_values = einops.reduce(flow_mat[:, :, :, :, 2], 's t h w -> s', 'sum')
        ref_rel_idx = np.argmax(confidence_values)
        ref_idx = reference_frames.indices[ref_rel_idx]
        reference_ai_frame = video.get_ai_frame(ref_idx)
        assert reference_ai_frame is not None
        warped_frame = warp_frame_pdcnet(reference_ai_frame, flow_mat[ref_rel_idx, 0, :, :, 0: 2])
        assert warped_frame is not None
        last_confidence_map = flow_mat[ref_rel_idx, 0, :, :, 2]
        cur_mask = (last_confidence_map * 255).astype(np.uint8)
        # kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        # cur_mask = cv2.erode(cur_mask, kern)
        mask = cv2.bitwise_or(mask, cur_mask)
        if ret_frame is None :
            ret_frame = np.copy(warped_frame)
            vis[ret_frame.shape[0] * 0: ret_frame.shape[0] * 1, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]] = video.get_raw_frame(frame_idx)
            vis[ret_frame.shape[0] * 1: ret_frame.shape[0] * 2, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]] = ret_frame
            vis[ret_frame.shape[0] * 1: ret_frame.shape[0] * 2, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]] = draw_mask(vis[ret_frame.shape[0] * 1: ret_frame.shape[0] * 2, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]], mask)
        else :
            pass
            # ---------------------------------------------------------------------------------------------
            #ret_frame = merge_images(ret_frame, warped_frame, cur_mask, method = 'naive')
        #ret_frame[mask < 127] = 0#vis[ret_frame.shape[0] * 1: ret_frame.shape[0] * 2, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]][mask < 127] = np.array([0, 0, 255])#draw_mask(vis[ret_frame.shape[0] * 1: ret_frame.shape[0] * 2, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]], mask)
        vis[ret_frame.shape[0] * 2: ret_frame.shape[0] * 3, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]] = reference_ai_frame
        #vis[ret_frame.shape[0] * 2: ret_frame.shape[0] * 3, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]][cur_mask < 127] = np.array([0, 0, 255])#draw_mask(vis[ret_frame.shape[0] * 2: ret_frame.shape[0] * 3, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]], cur_mask)
        #vis[ret_frame.shape[0] * 2: ret_frame.shape[0] * 3, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]] = draw_mask(vis[ret_frame.shape[0] * 2: ret_frame.shape[0] * 3, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]], cur_mask)
        vis[ret_frame.shape[0] * 3: ret_frame.shape[0] * 4, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]] = video.get_raw_frame(ref_idx)

        flow_mat[:, 0, :, :, 2] -= last_confidence_map[None, :, :] # subtract regions already warped, make them zero
        flow_mat[:, 0, :, :, 2] = np.clip(flow_mat[:, 0, :, :, 2], 0, 1)
    import copy
    reference_frames = copy.deepcopy(reference_frames)
    if add_prev_frame_as_reference :
        i = len(reference_frames)
        reference_frames.add(frame_idx - 1)
        vis[ret_frame.shape[0] * 2: ret_frame.shape[0] * 3, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]] = video.get_ai_frame(frame_idx - 1)
        vis[ret_frame.shape[0] * 3: ret_frame.shape[0] * 4, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]] = video.get_raw_frame(frame_idx - 1)
    print(f'[Both] Generating {frame_idx} from {reference_frames.indices}')
    original_frame = video.get_raw_frame(frame_idx)
    mask2 = 255 - mask
    # kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    # mask2 = cv2.dilate(mask2, kern)
    to_inpaint_frames = np.zeros((h, w * (len(reference_frames) + 1), 3), dtype = np.uint8)
    to_inpaint_ref_frames = np.zeros((h, w * (len(reference_frames) + 1), 3), dtype = np.uint8)
    to_inapint_mask = np.zeros((h, w * (len(reference_frames) + 1)), dtype = np.uint8)
    to_inpaint_frames[:, 0 * w: 1 * w] = ret_frame
    to_inpaint_ref_frames[:, 0 * w: 1 * w] = original_frame
    to_inapint_mask[:, 0 * w: 1 * w] = mask2
    for i, idx in enumerate(reference_frames.indices) :
        to_inpaint_frames[:, (i + 1) * w: (i + 2) * w] = video.get_ai_frame(idx)
        to_inpaint_ref_frames[:, (i + 1) * w: (i + 2) * w] = video.get_ai_frame(idx)
    cnets = [
        SingleControlNet(
            weight = 0.7,
            model = 'hed',
            args = {},
            condition = to_inpaint_ref_frames,
            guidance_start = 0,
            guidance_end = 1
        ),
        SingleControlNet(
            weight = 0.3,
            model = 'canny',
            args = {
                'low_threshold': 100,
                'high_threshold': 200
            },
            condition = to_inpaint_ref_frames,
            guidance_start = 0,
            guidance_end = 0.9
        )
    ]
    dnhist = merge_denoise_history(workspace, video, frame_idx, pdc, thres, denoise_history, denoise_history_ref_frames)
    ans, _, new_history, kv_hist_denoise = run_inpainting(model_paint, model_tagger, to_inpaint_frames, None, to_inapint_mask, ds, tagger_frame = to_inpaint_ref_frames[:, 0 * w: 1 * w], control_nets = cnets, history_guidance = dnhist, guidance_schedule_func = guidance_schedule_func)
    #ans, _, new_history = run_inpainting(model_inpainting, model_tagger, to_inpaint_frames, to_inpaint_ref_frames, to_inapint_mask, ds, lambda p: 0 if p > stop_p else 1, control_nets = cnets, history_guidance = dnhist)
    ans = ans[:, : w]
    i = len(reference_frames)
    vis[ret_frame.shape[0] * 1: ret_frame.shape[0] * 2, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]] = ans
    cv2.imwrite(os.path.join(vis_dir, f'{frame_idx:05d}.png'), vis)
    return ans, new_history, kv_hist_denoise

def expand_mask(mask: np.ndarray, ori_image: np.ndarray) -> np.ndarray :
    laplacian = (cv2.cvtColor(np.absolute(cv2.Laplacian(ori_image, cv2.CV_64F)).astype(np.uint8), cv2.COLOR_RGB2GRAY) > 20).astype(np.uint8) * 255
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    laplacian = cv2.dilate(laplacian, kern)
    mask = cv2.bitwise_or(mask, laplacian)
    return mask

def generate_ai_frame_with_ref_warp_and_inpaint_crossattn(
        level, 
        workspace, 
        model_paint, 
        model_inpainting, 
        model_tagger: Tagger, 
        video: VideoData, 
        frame_idx: int, 
        reference_frames: VideoFrameIndices, 
        flow_mat: np.ndarray, 
        pdc: PDCNetAux,
        denoise_history = [],
        denoise_history_ref_frames = [],
        reference_kv = [],
        thres: float = 0.5, 
        ds: float = 0.6,
        guidance_schedule_func = None
        ) :
    vis_dir = os.path.join(workspace, 'render_vis', f'u{level:02d}')
    os.makedirs(vis_dir, exist_ok = True)
    flow_mat[:, :, :, :, 2] = (flow_mat[:, :, :, :, 2] > thres).astype(np.float32) # only confidence>thres regions are considered
    mask = np.zeros((flow_mat.shape[2], flow_mat.shape[3]), dtype = np.uint8)
    ret_frame = None
    vis = np.zeros((flow_mat.shape[2] * 4, flow_mat.shape[3] * (len(reference_frames.indices) + 1), 3), dtype = np.uint8)
    for i in range(len(reference_frames)) :
        confidence_values = einops.reduce(flow_mat[:, :, :, :, 2], 's t h w -> s', 'sum')
        ref_rel_idx = np.argmax(confidence_values)
        ref_idx = reference_frames.indices[ref_rel_idx]
        reference_ai_frame = video.get_ai_frame(ref_idx)
        assert reference_ai_frame is not None
        warped_frame = warp_frame_pdcnet(reference_ai_frame, flow_mat[ref_rel_idx, 0, :, :, 0: 2])
        assert warped_frame is not None
        last_confidence_map = flow_mat[ref_rel_idx, 0, :, :, 2]
        cur_mask = (last_confidence_map * 255).astype(np.uint8)
        mask = cv2.bitwise_or(mask, cur_mask)
        if ret_frame is None :
            ret_frame = np.copy(warped_frame)
        else :
            ret_frame = merge_images(ret_frame, warped_frame, cur_mask, method = 'naive')
        #ret_frame[mask < 127] = 0
        vis[ret_frame.shape[0] * 0: ret_frame.shape[0] * 1, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]] = video.get_raw_frame(frame_idx)
        vis[ret_frame.shape[0] * 1: ret_frame.shape[0] * 2, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]] = ret_frame
        vis[ret_frame.shape[0] * 1: ret_frame.shape[0] * 2, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]] = draw_mask(vis[ret_frame.shape[0] * 1: ret_frame.shape[0] * 2, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]], mask)
        #vis[ret_frame.shape[0] * 1: ret_frame.shape[0] * 2, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]][mask < 127] = np.array([0, 0, 255])#draw_mask(vis[ret_frame.shape[0] * 1: ret_frame.shape[0] * 2, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]], mask)
        vis[ret_frame.shape[0] * 2: ret_frame.shape[0] * 3, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]] = warped_frame
        #vis[ret_frame.shape[0] * 2: ret_frame.shape[0] * 3, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]][cur_mask < 127] = np.array([0, 0, 255])#draw_mask(vis[ret_frame.shape[0] * 2: ret_frame.shape[0] * 3, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]], cur_mask)
        vis[ret_frame.shape[0] * 2: ret_frame.shape[0] * 3, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]] = draw_mask(vis[ret_frame.shape[0] * 2: ret_frame.shape[0] * 3, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]], cur_mask)
        vis[ret_frame.shape[0] * 3: ret_frame.shape[0] * 4, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]] = video.get_raw_frame(ref_idx)
        flow_mat[:, 0, :, :, 2] -= last_confidence_map[None, :, :] # subtract regions already warped, make them zero
        flow_mat[:, 0, :, :, 2] = np.clip(flow_mat[:, 0, :, :, 2], 0, 1)
    original_frame = video.get_raw_frame(frame_idx)
    mask2 = 255 - mask
    mask2 = expand_mask(mask2, original_frame)
    cnets = [
        SingleControlNet(
            weight = 0.7,
            model = 'hed',
            args = {},
            condition = original_frame,
            guidance_start = 0,
            guidance_end = 1
        ),
        SingleControlNet(
            weight = 0.3,
            model = 'canny',
            args = {
                'low_threshold': 100,
                'high_threshold': 200
            },
            condition = original_frame,
            guidance_start = 0,
            guidance_end = 0.9
        )
    ]
    dnhist = merge_denoise_history(workspace, video, frame_idx, pdc, thres, denoise_history, denoise_history_ref_frames)
    #ans, _, _ = run_inpainting(model_inpainting, model_tagger, ret_frame, original_frame, mask2, ds, lambda p: 0 if p > stop_p else 1, control_nets = cnets)
    ans, _, new_history, kv_hist_denoise = run_inpainting(model_paint, model_tagger, ret_frame, original_frame, mask2, ds, control_nets = cnets, history_guidance = dnhist, guidance_schedule_func = guidance_schedule_func, reference_kv = reference_kv)
    i = len(reference_frames)
    vis[ret_frame.shape[0] * 1: ret_frame.shape[0] * 2, i * ret_frame.shape[1]: (i + 1) * ret_frame.shape[1]] = ans
    cv2.imwrite(os.path.join(vis_dir, f'{frame_idx:05d}.png'), vis)
    return ans, new_history, kv_hist_denoise

def generate_ai_frame_with_ref(
        level, 
        workspace, 
        model_paint,
        inpaint_model, 
        tagger, 
        pdcnet: PDCNetAux, 
        video: VideoData, 
        frame_idx: int, 
        reference_frames: VideoFrameIndices, 
        mode = 'warp_and_inpaint', 
        denoise_history = [],
        denoise_history_ref_frames = [],
        reference_kv = [],
        thres: float = 0.95, 
        ds: float = 0.6,
        guidance_schedule_func = None
        ) :
    """
    modes: warp_and_inpaint, self_attn (use MFR/keyframer), both (use both)
    """
    flow_mat = pdcnet.calculate_multiple_to_one(video, reference_frames, frame_idx)
    if mode == 'warp_and_inpaint' :
        return generate_ai_frame_with_ref_warp_and_inpaint(level, workspace, model_paint, inpaint_model, tagger, video, frame_idx, reference_frames, flow_mat, pdcnet, denoise_history, denoise_history_ref_frames, thres, ds, guidance_schedule_func = guidance_schedule_func)
    elif mode == 'self_attn' :
        return generate_ai_frame_with_ref_self_attn(level, workspace, model_paint, inpaint_model, tagger, video, frame_idx, reference_frames, flow_mat, pdcnet, denoise_history, denoise_history_ref_frames, thres, ds, guidance_schedule_func = guidance_schedule_func)
    elif mode == 'both' :
        return generate_ai_frame_with_ref_both(level, workspace, model_paint, inpaint_model, tagger, video, frame_idx, reference_frames, flow_mat, pdcnet, denoise_history, denoise_history_ref_frames, thres, ds, guidance_schedule_func = guidance_schedule_func)
    elif mode == 'warp_and_inpaint_crossattn' :
        return generate_ai_frame_with_ref_warp_and_inpaint_crossattn(level, workspace, model_paint, inpaint_model, tagger, video, frame_idx, reference_frames, flow_mat, pdcnet, denoise_history, denoise_history_ref_frames, reference_kv, thres, ds, guidance_schedule_func = guidance_schedule_func)
    
def generate_seed_frames(model_gen, model_tagger, video: VideoData, seed_indices: VideoFrameIndices, ds = 0.6) :
    frames = []
    for idx in seed_indices.indices :
        frames.append(video.get_raw_frame(idx))
    large_frame = np.concatenate(frames, axis = 1)
    cnets = [
        SingleControlNet(
            weight = 0.7,
            model = 'hed',
            args = {},
            condition = large_frame,
            guidance_start = 0,
            guidance_end = 1
        ),
        SingleControlNet(
            weight = 0.3,
            model = 'canny',
            args = {
                'low_threshold': 100,
                'high_threshold': 200
            },
            condition = large_frame,
            guidance_start = 0,
            guidance_end = 0.9
        )
    ]
    #ds = 0.4
    ai_frame, kv_hist = img2img(model_gen, model_tagger, large_frame, ds, None, override_tagger_frame = frames[0], control_nets = cnets) # use the first seed frame's prompt
    ai_frames = np.split(ai_frame, len(seed_indices), axis = 1)
    return ai_frames, kv_hist

def KeyframeDeconv(video: VideoData, to_generate: VideoFrameIndices, reference_frames: VideoFrameIndices) :
    pass

def run_exp(model, model_inpaint, model_tagger, name: str, key_frame_thres, num_ref_for_generation, ds = 0.6, guidance_schedule_func = None) :
    name = f'out-{name}'
    workspace = f'/data2/video-ws/{name}'
    import os
    os.makedirs(workspace, exist_ok=True)
    (h, w) = 768, 512
    pdcnet = PDCNetAux(create_of_algo(), workspace, batch_size = 16, device = torch.device('cuda:0'))
    video = VideoData('videos/out.mp4', (w, h), workspace, keep_every = 3, max_len_sec = 30)
    level = 0
    n_seed_frames = 1
    num_ref_for_generation = num_ref_for_generation
    history: List[VideoFrameIndices] = [VideoFrameIndices.from_n(video.num_frames)] # initially all frames
    if False :
        # generate frames at level 0
        cur_level_frames = []
        print('[Main] Analysing frames at level 0')
        for frame, i in video.key_frames(key_frame_thres) :
            dst = os.path.join(workspace, f'd{level:02d}', f'{i:05d}.png')
            if os.path.exists(dst) :
                cur_level_frames.append(i)
                continue
            os.makedirs(os.path.join(workspace, f'd{level:02d}'), exist_ok = True)
            cv2.imwrite(dst, frame)
            cur_level_frames.append(i)
        frame_indices = VideoFrameIndices(cur_level_frames)
        history.append(frame_indices)
    else :
        level = 0
        frame_indices = VideoFrameIndices.from_n(video.num_frames)
    use_first_frame_as_keyframe = False
    # generate frames at each level until we can proceed with seed frame rendering
    while len(frame_indices) > n_seed_frames :
        level += 1
        print(f'[Main] Analysing frames at level {level}')
        if use_first_frame_as_keyframe :
            frame_indices = VideoFrameIndices([0])
        else :
            frame_indices = KeyframeConv(pdcnet, os.path.join(workspace, f'd{level:02d}'), video, frame_indices, kernel_size = 30, stride = 15, dilation = 2)
        history.append(frame_indices)
    torch.cuda.empty_cache()
    pdcnet.purge() # free up space
    frame_indices.add(0) # add the first frame
    # generate seed frames
    seed_frames, seed_kv_hist = generate_seed_frames(model, model_tagger, video, frame_indices, ds = 0.8)
    for i, idx in enumerate(frame_indices.indices) :
        dst = os.path.join(workspace, f'seed', f'{idx:05d}.png')
        cv2.imwrite(dst, seed_frames[i])
        video.put_ai_frame(idx, seed_frames[i]) # store in result
        video.put_kv(idx, seed_kv_hist)
    denoise_process_dst = os.path.join(workspace, 'denoise_proc')
    os.makedirs(denoise_process_dst, exist_ok = True)
    generated_frames = history.pop() # remove last in history (seed frames), as reference for generting 
    while len(history) > 0 : # while we are not back at the top level
        level -= 1
        denoise_history = None
        cur_level_frames = history.pop()
        print(f'[Main] Generating frames at level {level}')
        print(f'-----------------------------------------')
        print(cur_level_frames.indices)
        print(f'-----------------------------------------')
        print(generated_frames.indices)
        print(f'-----------------------------------------')
        last_frame_idx = -1
        cur_level_frames.remove(generated_frames)
        for idx in tqdm(cur_level_frames.indices) :
            reference_frames = generated_frames.adjacent_frames(idx, num_ref_for_generation)
            if level != 0 :
                denoise_history = None
            # if last_frame_idx != -1 and level != 0 :
            #     if last_frame_idx not in reference_frames.indices :
            #         reference_frames.indices.append(last_frame_idx)
            # if idx == 1 and use_first_frame_as_keyframe :
            #     reference_frames.indices = [0]
            # TODO: generate frame idx using reference_frames
            print(name)
            kv_hist_ref = []
            input_kv_hist = []
            for ref_frame in reference_frames.indices :
                kv_ref = video.get_kv(ref_frame)
                assert kv_ref
                input_kv_hist.append(kv_ref)
                kv_hist_ref.append(ref_frame)
            if last_frame_idx != -1 and level == 0 :
                kv_ref = video.get_kv(last_frame_idx)
                assert kv_ref
                input_kv_hist.append(kv_ref)
                kv_hist_ref.append(last_frame_idx)
            print(f'[Main] Generating {idx} from {reference_frames.indices} with cross attention from {kv_hist_ref}')
            frame, new_history, kv_hist_denoise = generate_ai_frame_with_ref(
                level, 
                workspace, 
                model, 
                model_inpaint, 
                model_tagger, 
                pdcnet, 
                video, 
                idx, 
                reference_frames, 
                mode = 'warp_and_inpaint_crossattn', 
                ds = ds,
                denoise_history = [denoise_history] if denoise_history is not None else [],
                denoise_history_ref_frames = [last_frame_idx],
                guidance_schedule_func = guidance_schedule_func,
                reference_kv = input_kv_hist
                )
            video.put_ai_frame(idx, frame)
            video.put_kv(idx, kv_hist_denoise)
            denoise_history = {}
            # for k in new_history.keys() :
            #     denoise_history[k] = new_history[k][:, :w]
            #     cv2.imwrite(os.path.join(denoise_process_dst, f'{idx:05d}-{k:02d}.png'), cv2.cvtColor((denoise_history[k] * 127.5 + 127.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
            if last_frame_idx != -1 and level == 0 :
                video.remove_kv(last_frame_idx)
            print(video.kv_hist_map.keys())
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            last_frame_idx = idx
        generated_frames.add(cur_level_frames)
    pdcnet.purge() # free up space

def guidance_schedule(p) -> float :
    if p <= 0.9 :
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
    # run_exp(model, model_inpaint, tagger,
    #     '60fps_new-keyframethres12-sch0.8_ds0.4_fixseed_conf-thres0.5_ppw0.0', 
    #     denoising_strength = 0.4, 
    #     confidence_thres = 0.5, 
    #     propagated_pixel_weight = 0.0,
    #     key_frame_thres = 48,
    #     guidance_schedule_func = guidance_schedule
    #     )
    run_exp(model, model_inpaint, tagger, 'test', key_frame_thres = 12, num_ref_for_generation = 1, ds = 0.8, guidance_schedule_func = guidance_schedule)

if __name__ == '__main__' :
    main()
