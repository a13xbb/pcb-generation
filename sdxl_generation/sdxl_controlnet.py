import os
os.environ['HF_HOME'] = '/mnt/ssdm2/users/alexblokh/cache'

import torch
import cv2
import numpy as np
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
)
from PIL import Image
from pathlib import Path

from peft import PeftModel

device = "cuda:0"

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to(device)

lora_dir = "/mnt/ssdm2/users/alexblokh/pcb_generation/pcb-generation/pcb_lora"
lora_scale = 0.8

pipe.unet = PeftModel.from_pretrained(
    pipe.unet,
    lora_dir
)

# pipe.unet.set_adapter("default")
# pipe.unet.set_adapter_scale(lora_scale)

pipe.unet.to(device)
pipe.unet.eval()

prompt = """
macro photo of printed circuit board,
green solder mask,
silver plated pads,
thin green copper traces,
high resolution macro photography,
industrial PCB manufacturing
"""

edge_dir = Path("/mnt/ssdm2/users/alexblokh/pcb_generation/pcb-generation/images/edges")
out_dir = Path("/mnt/ssdm2/users/alexblokh/pcb_generation/pcb-generation/images/generated2")
out_dir.mkdir(exist_ok=True)

for p in edge_dir.glob("*.png"):

    edge = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    edge = cv2.resize(edge, (768, 768))

    edge = Image.fromarray(edge)

    image = pipe(
        prompt=prompt,
        image=edge,
        num_inference_steps=30,
        guidance_scale=6,
        height=768,
        width=768,
        controlnet_conditioning_scale = 1.0
    ).images[0]

    image.save(out_dir / p.name)