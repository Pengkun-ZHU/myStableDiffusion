# myStableDiffusion
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) \
Introducing stable diffusion with prompt applied!!! \
Feel free to modify and try various prompts to guide your image generation, powered by CLIP by openai's pretrained model clip-vit-large-patch14.

**Reference**
-----
https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#quick-summary \
https://huggingface.co/docs/diffusers/main/en/using-diffusers/write_own_pipeline \
https://nn.labml.ai/diffusion/stable_diffusion/model/unet.html 

**Note**
-----
For user in China Mainland, the direct access to huggingface.co may be blocked no matter whether VPN applied. \
A remedy towards this is to download the model to your local path, and change the arg named pretrained_model_name_or_path of function from_pretrained() to the very absolute path.
