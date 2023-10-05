# myStableDiffusion
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) \
Introducing stable diffusion with prompt applied!!! \
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
