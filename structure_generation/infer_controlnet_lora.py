from __future__ import annotations
import os
os.environ['HF_HOME'] = '/mnt/ssdm2/users/alexblokh/cache'

import argparse
from pathlib import Path

import cv2
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from peft import PeftModel
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SDXL ControlNet inference with an existing PCB LoRA.")
    parser.add_argument("--structure_map", type=str, required=True, help="Path to the structure map image.")
    parser.add_argument("--controlnet_path", type=str, required=True, help="Path to a trained ControlNet checkpoint.")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the existing PCB LoRA.")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the generated PCB image.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="macro photo of printed circuit board, green solder mask, copper traces, realistic industrial PCB",
    )
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--lora_scale", type=float, default=0.8)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16 if device.type == "cuda" else torch.float32

    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_path,
        torch_dtype=weight_dtype,
    )
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet,
        torch_dtype=weight_dtype,
    ).to(device)

    pipe.unet = PeftModel.from_pretrained(pipe.unet, args.lora_path)
    for module in pipe.unet.modules():
        if hasattr(module, "scaling"):
            for key in module.scaling:
                module.scaling[key] = args.lora_scale
    pipe.unet.eval()

    structure = cv2.imread(args.structure_map, cv2.IMREAD_GRAYSCALE)
    if structure is None:
        raise FileNotFoundError(f"Failed to read structure map: {args.structure_map}")
    structure = cv2.resize(structure, (args.width, args.height))
    structure_image = Image.fromarray(structure)

    generator = torch.Generator(device=device.type).manual_seed(args.seed)
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or None,
        image=structure_image,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        height=args.height,
        width=args.width,
        generator=generator,
    ).images[0]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"Saved generated image to: {output_path}")


if __name__ == "__main__":
    main()
