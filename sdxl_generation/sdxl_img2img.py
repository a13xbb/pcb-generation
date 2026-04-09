import os
import env
os.environ['HF_HOME'] = env.HF_HOME
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from peft import PeftModel
from PIL import Image

device = "cuda"

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to(device)

# pipe.enable_xformers_memory_efficient_attention()

# загружаем LoRA
lora_dir = "/mnt/ssdm2/users/alexblokh/pcb_generation/pcb-generation/pcb_lora"
lora_scale = 1.0

pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_dir)
for module in pipe.unet.modules():
    if hasattr(module, "scaling"):
        for key in module.scaling:
            module.scaling[key] = lora_scale
pipe.unet.eval()

image = Image.open('/mnt/ssdm2/users/alexblokh/pcb_generation/pcb-generation/images/layouts/layout_0.png')

result = pipe(
    prompt="""
ultra detailed macro photo of a real printed circuit board,
green solder mask, copper traces, vias, electronic components,
realistic shadows, reflections, photorealistic, high detail, 8k
""",
    image=image,
    strength=0.5,              # КЛЮЧЕВОЙ параметр
    guidance_scale=7,
    num_inference_steps=50
).images[0]

result.save("/mnt/ssdm2/users/alexblokh/pcb_generation/pcb-generation/images/generated_img2img/pcb.png")