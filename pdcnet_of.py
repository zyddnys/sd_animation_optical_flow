import os
import torch
import torchvision.transforms as transforms

import sys
sys.path.append('../DenseMatching')

from models.PDCNet.PDCNet import PDCNet_vgg16

from utils_data.image_transforms import ArrayToTensor
from utils_flow.pixel_wise_mapping import remap_using_flow_fields
from utils_flow.util_optical_flow import flow_to_image
from model_selection import load_network

import einops
import cv2
import numpy as np

def warp_frame_latent(latent, flow) :
    latent = einops.rearrange(latent.cpu().numpy().squeeze(0), 'c h w -> h w c')
    lh, lw = latent.shape[:2]
    h, w = flow.shape[:2]
    disp_x, disp_y = flow[:, :, 0], flow[:, :, 1]
    latent = cv2.resize(latent, (w, h), interpolation=cv2.INTER_CUBIC)
    X, Y = np.meshgrid(np.linspace(0, w - 1, w),
                       np.linspace(0, h - 1, h))
    map_x = (X+disp_x).astype(np.float32)
    map_y = (Y+disp_y).astype(np.float32)
    remapped_latent = cv2.remap(latent, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    remapped_latent = cv2.resize(remapped_latent, (lw, lh), interpolation=cv2.INTER_CUBIC)
    remapped_latent = torch.from_numpy(einops.rearrange(remapped_latent, 'h w c -> 1 c h w'))
    return remapped_latent

def warp_frame(frame, flow) :
    h, w = flow.shape[:2]
    disp_x, disp_y = flow[:, :, 0], flow[:, :, 1]
    X, Y = np.meshgrid(np.linspace(0, w - 1, w),
                       np.linspace(0, h - 1, h))
    map_x = (X+disp_x).astype(np.float32)
    map_y = (Y+disp_y).astype(np.float32)
    frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return frame


class PDCNetPlus() :
    def __init__(self, ckpt_path = 'pre_trained_models/PDCNet_plus_m.pth.tar') -> None:
        local_optim_iter = 16
        global_gocor_arguments = {'optim_iter': 6, 'steplength_reg': 0.1, 'train_label_map': False,
                                    'apply_query_loss': True,
                                    'reg_kernel_size': 3, 'reg_inter_dim': 14, 'reg_output_dim': 16}
        local_gocor_arguments = {'optim_iter': local_optim_iter, 'steplength_reg': 0.1}
        network = PDCNet_vgg16(global_corr_type='GlobalGOCor', global_gocor_arguments=global_gocor_arguments,
                                    normalize='leakyrelu', same_local_corr_at_all_levels=True,
                                    local_corr_type='LocalGOCor', local_gocor_arguments=local_gocor_arguments,
                                    local_decoder_type='OpticalFlowEstimatorResidualConnection',
                                    global_decoder_type='CMDTopResidualConnection',
                                    corr_for_corr_uncertainty_decoder='corr',
                                    give_layer_before_flow_to_uncertainty_decoder=True,
                                    var_2_plus=520 ** 2, var_2_plus_256=256 ** 2, var_1_minus_plus=1.0, var_2_minus=2.0,
                                    make_two_feature_copies=True)
        network = load_network(network, checkpoint_path=ckpt_path).cuda()
        network.eval()
        self.network = network

    @torch.no_grad()
    def calc(self, frame1, frame2) :
        source_img = einops.rearrange(torch.from_numpy(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)), 'h w c -> 1 c h w')
        target_img = einops.rearrange(torch.from_numpy(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)), 'h w c -> 1 c h w')

        flow_est, uncertainty_est = self.network.estimate_flow_and_confidence_map(source_img, target_img)

        flow_est = flow_est.permute(0, 2, 3, 1)[0].cpu().numpy()
        confidence = uncertainty_est['weight_map'].softmax(dim=1).cpu().numpy()[0][0]
        log_confidence = uncertainty_est['weight_map'].log_softmax(dim=1).cpu().numpy()[0][0]
        return flow_est, confidence, log_confidence

def create_of_algo(ckpt) :
    algo = PDCNetPlus(ckpt)
    return algo

@torch.no_grad()
def main():
    target_transform = transforms.Compose([ArrayToTensor()])  # only put channel first
    input_transform = transforms.Compose([ArrayToTensor(get_float=False)])  # only put channel first

    local_optim_iter = 6
    global_gocor_arguments = {'optim_iter': 6, 'steplength_reg': 0.1, 'train_label_map': False,
                                'apply_query_loss': True,
                                'reg_kernel_size': 3, 'reg_inter_dim': 16, 'reg_output_dim': 16}
    local_gocor_arguments = {'optim_iter': local_optim_iter, 'steplength_reg': 0.1}
    network = PDCNet_vgg16(global_corr_type='GlobalGOCor', global_gocor_arguments=global_gocor_arguments,
                                normalize='leakyrelu', same_local_corr_at_all_levels=True,
                                local_corr_type='LocalGOCor', local_gocor_arguments=local_gocor_arguments,
                                local_decoder_type='OpticalFlowEstimatorResidualConnection',
                                global_decoder_type='CMDTopResidualConnection',
                                corr_for_corr_uncertainty_decoder='corr',
                                give_layer_before_flow_to_uncertainty_decoder=True,
                                var_2_plus=520 ** 2, var_2_plus_256=256 ** 2, var_1_minus_plus=1.0, var_2_minus=2.0,
                                make_two_feature_copies=True)
    network = load_network(network, checkpoint_path='pre_trained_models/PDCNet_plus_m.pth.tar').cuda()
    network.eval()


    from PIL import Image
    import numpy as np
    source_img = Image.open('mmd_in/raw_000000.png')
    target_img = Image.open('mmd_in/raw_000001.png')
    source_img_np = np.array(source_img)
    source_img = input_transform(source_img).unsqueeze(0)
    target_img = target_transform(target_img).unsqueeze(0)

    flow_est, uncertainty_est = network.estimate_flow_and_confidence_map(source_img, target_img)

    flow_est = flow_est.permute(0, 2, 3, 1)[0].cpu().numpy()
    rgb_es_flow = flow_to_image(flow_est)
    confidence = (uncertainty_est['weight_map'].softmax(dim=1)[0][0] * 255).cpu().numpy().astype(np.uint8)
    print(flow_est.shape)
    remapped_est = remap_using_flow_fields(source_img_np, flow_est[:,:,0], flow_est[:,:,1]).astype(np.uint8)
    import cv2
    cv2.imwrite('rgb_es_flow.png', rgb_es_flow)
    cv2.imwrite('confidence.png', confidence)
    cv2.imwrite('warped.png', cv2.cvtColor(remapped_est, cv2.COLOR_RGB2BGR))
    remapped_est[confidence < 0.6 * 255] = np.array([255, 0, 0])
    cv2.imwrite('warped_masked.png', cv2.cvtColor(remapped_est, cv2.COLOR_RGB2BGR))
    # breakpoint()
    # print(uncertainty_est.shape)



if __name__ == "__main__":
    torch.cuda.empty_cache()
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # either gpu or cpu
    main()

