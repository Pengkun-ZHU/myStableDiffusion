# myStableDiffusion
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
```diff
@@ Introducing stable diffusion!!! @@
```
Compared with traditional diffusion model where noising and denoising is applied to all pixels, this very version applies only to latent space, hence much more efficient with arguably minor performance degradation. It takes astounishing 1 min to generate a 512x512 image with 50 iterations by CUDA on my RTX3060Ti.\
Noteworthilly, runing on CPU or MacOS is supported but not encouraged. It takes roughly 5 hours on my 2021 Macbook pro to generate 512x512 image with 25 iterations... the heat is also unacceptable :(\
Feel free to modify and try various prompts to guide your image generation, powered by openai's pretrained CLIP model clip-vit-large-patch14.

**Reference**
-----
https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#quick-summary \
https://huggingface.co/docs/diffusers/main/en/using-diffusers/write_own_pipeline \
https://nn.labml.ai/diffusion/stable_diffusion/model/unet.html 

**Demo**
-----
Prompt:"Chinese grandparents with their grandson, in cartoon style" \
Iterations: 50\
![2](https://github.com/Pengkun-ZHU/myStableDiffusion/assets/56779575/d04d0ad3-3c7d-4d49-9c8a-8798f64cfd63) 

Prompt:""Beautiful girl with long hair and glasses"\
Iterations: 75\
![6](https://github.com/Pengkun-ZHU/myStableDiffusion/assets/56779575/cbe15854-1847-41ab-89cb-2864205cd8cd)



**Note**
-----
For user in China Mainland, the direct access to huggingface.co may be blocked no matter whether VPN applied. \
A remedy towards this is to download the model to your local path, and change the arg named pretrained_model_name_or_path of function from_pretrained() to the very absolute path.
