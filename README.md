# Showcase
Source https://www.youtube.com/watch?v=eroDb6bRSKA
![Showcase](showcase/output_v2023-03-12.mp4.gif) \
Left - Img2img unguided \
Middle - Original MMD \
Right - Guided \
Created using the following schedule and denoise strength of 0.4
```python
def guidance_schedule(denoise_percentage, aux: dict) -> float | np.ndarray :
    dist = aux['dist_mat']
    thres = 8 # 8 pixels away
    weights = np.ones((dist.shape[0], dist.shape[1]), dtype = np.float32)
    if denoise_percentage < 0.8 :
        weights *= 0.7
    else :
        weights *= 0.5
    weights[dist > thres] = 0.05
    return weights
```

# Optical flow guided AI animation
AI generated animation with stable diffusion often suffers from flicking due to the inherent randomness in the generation process and the lack of information between frames. This project intents to solve this issue by guiding image generation process using frame predicted by optical flow.
# How it works
First, the first frame is generated using img2img without any guidance. Then the second frame is generate while being guided by the prediction frame created from the previous AI generated frame. The prediction frame is created with the previous AI generated and optical flow calculated from the original video. The process repeats for every new frame.
## Guidance
It is well known that in the reverse diffusion process we can obtain the predicted $\hat{z}_0$ using
$$\hat{z}_0=\frac{z_t-\sqrt{1-\alpha_t}\epsilon(z_t,t)}{\sqrt{\alpha_t}}$$
where $\epsilon(z_t,t)$ is the model output.

Now we are given the optical flow predicted frame $k_0$, then we can blend $\hat{z}_0$ and $k_0$ with $\hat{z}'_0=(1-w_t)\cdot \hat{z}_0+w_t\cdot k_0$ where $w_t$ is the weight in range of $[0,1]$, $w_t$ can be a single number or a 2D array weighting each pixel (in latent space in the case of stable diffusion) and can vary based on the current denoising step.

We can obtain the new $\epsilon'$ value using
$$\epsilon'=\frac{z_t-\sqrt{\alpha_t}\hat{z}'_0}{\sqrt{1-\alpha_t}}$$
and plug the new $\epsilon'$ into the rest of sampling method.

## Weighted Guidance
$w_t$ is a 2D array in our implementation, we use a simple heuristic that a pixel moving too far in the optical flow will be assign a low weight and a pixel not moving very far will be assigned a high weight.

# How to run
This repo is for people who have basic knowledge of stable diffusion and Python.
1. You need a base model, here I use [ACertainModel](https://huggingface.co/JosephusCheung/ACertainModel)
2. You need a booru tagger, here I use [wd-v1-4-swinv2-tagger-v2](https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2)
3. You need [PDCNet+](https://github.com/PruneTruong/DenseMatching), it can generate optical flow and confidence
3. Get a video to process, resize it to resolution acceptable by stable diffusion (e.g. 512x768)
4. Run `python ofgen.py --i <video_file> --o <save_dir>`
5. Output frames are named `<save_dir>/converted_%06d.png`, use ffmpeg to create a video from them
6. Denoise strength and weight $w_t$ schedule can be changed in `ofgen.py`

# Other attempts

## 1
Fixing noise (seed) helps

# Failed attempts

## 1
Guiding images during the denoising process always leads to blurry image, I suspect this is due to the Unet not knowing what should it do, the Unet trying to generate one image but the guidance tries to lead it to another image

## 2
Code is in [ofgen_pixel_inpaint.py](ofgen_pixel_inpaint.py) \
So I tried feeding optical warped image. Pixels produced by optical flow with high confidence are kept and the low confidence pixels are masked for inpainting. \
Two issues here: 
1. Pixels warped from optical flow continue to worse despite having high confidence
![](failed/optical_flow_artifacts.png)
2. SD's VAE when applied repeatedly (in video it means result from one frame is used to generate the next) leads artifacts
![](failed/vae_artifact.bmp)

# Ideas pending
1. Generate multiple frames simultaneously instead of one after another, during the denoise process minimize energy term that ensure temporary smoothness across frames
2. Train a network to remove SD VAE's artifact
3. Train a control net that use optical flow warped frame as reference to generate next frame, however I don't have any video dataset

# Discussion
QQ群: 164153710\
Discord https://discord.gg/Ak8APNy4vb

# Known issues and future work
1. No A1111 stable-diffusion-webui plugin which makes this repo a mere experiment, more work is required to bring this to the general public
2. <s>We use Farneback for optical flow calculation, this can be improved with other newer optical flow algorithm</s> We use [PDCNet+](https://github.com/PruneTruong/DenseMatching) for optical flow.
3. We only use img2img for frame generation due to its simplicity, better result can be achieved using ControlNet and custom character LoRA
4. Multiple passes can be used for better quality
5. The predication frame can be created from optical flow from both side, not just in the forward direction
6. Error from the first frame will accumulate across the entire video

# Credits
先吹爆一喵 This repo is based on lllyasviel's [ControlNet](https://github.com/lllyasviel/ControlNet) repo, a lot of code are copied from there. \
The whole idea turned out to be very similar to [disco-diffusion](https://github.com/alembics/disco-diffusion), so I encourage people to check out their work.
