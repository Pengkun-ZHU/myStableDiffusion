import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import transformers
import diffusers
from tqdm.auto import tqdm
import unet
from huggingface_hub import hf_hub_download

dev = "cuda" if torch.cuda.is_available() else "cpu"

# Load variational autoencoder ( measured by KL entropy ) model
vae = diffusers.AutoencoderKL.from_pretrained( "CompVis/stable-diffusion-v1-4",
                                               subfolder="vae",
                                               local_files_only=True )

# UNet for generating latent space
unet = diffusers.UNet2DConditionModel.from_pretrained( "CompVis/stable-diffusion-v1-4",
                                                       subfolder="unet",
                                                       local_files_only=True )

# Scheduler guiding what step to take when adding noise. See the link below for details on each parameter
# https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#quick-summary
scheduler = diffusers.LMSDiscreteScheduler( beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                            num_train_timesteps=1000 )

textTokenizer = transformers.CLIPTokenizer.from_pretrained( "openai/clip-vit-large-patch14" )
textEncoder = transformers.CLIPTextModel.from_pretrained( "openai/clip-vit-large-patch14" )

vae = vae.to( dev )
unet = unet.to( dev )
textEncoder = textEncoder.to( dev )


""" Prepare everything """
prompt = ["whatever"]   # Fill in your own prompt
hImg = 512  # Image height
wImg = 512  # Image width
num_inference_steps = 50  # You may decrease it for quick result, especially during debug
guidance_scale = 7.5
generator = torch.manual_seed( 42 )  # seed for generating random noise, it must be set for both training and eval mode, otherwise you get differernt noise each time
batchSz = 1

# Prep text
text_input = textTokenizer( prompt, padding="max_length", max_length=textTokenizer.model_max_length, truncation=True, return_tensors="pt" )
with torch.no_grad():
  text_embeddings = textEncoder(text_input.input_ids.to( dev ))[0]
max_length = text_input.input_ids.shape[-1]
uncond_input = textTokenizer(
    [""] * batchSz, padding="max_length", max_length=max_length, return_tensors="pt"
)
with torch.no_grad():
  uncond_embeddings = textEncoder(uncond_input.input_ids.to( dev ))[0]
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

# For following part, See https://huggingface.co/docs/diffusers/main/en/using-diffusers/write_own_pipeline
scheduler.set_timesteps( num_inference_steps )
latentSpace = torch.randn( ( batchSz, unet.config.in_channels, hImg // 8, wImg // 8 ), generator=generator ) # The height and width are divided by 8 because the vae model has 3 down-sampling layers.
latentSpace = latentSpace.to( dev )
latentSpace = latentSpace * scheduler.sigmas[0]  # "Need to scale to match k", what does it mean?

with torch.autocast( dev ):
    for t in tqdm( scheduler.timesteps ):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latentSpace] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latentSpace = scheduler.step(noise_pred, t, latentSpace).prev_sample

    # scale and decode the image latents with vae
    latentSpace = 1 / 0.18215 * latentSpace

    with torch.no_grad():
        image = vae.decode( latentSpace )

    # Display
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0]
