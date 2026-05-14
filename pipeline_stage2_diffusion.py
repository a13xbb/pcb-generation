#!/usr/bin/env python3
"""Stage 2: ControlNet + LoRA diffusion on pre-generated structure maps.

Reads from --input_dir:
  structure_maps/layout_N.png — Canny edge map (from stage 1)
  canvases/layout_N.pkl.gz    — canvas for border application (from stage 1)

Writes to --output_dir:
  images/layout_N.png         — final refined PCB image (default)

  With --verbose:
    images/layout_N_raw.png      — after ControlNet generation
    images/layout_N_bordered.png — after border painting (unless --skip_borders)
    images/layout_N_final.png    — after img2img refinement

  With --skip_borders:
    Skips border painting, refines directly from raw ControlNet output.
    Useful for light denoising with low --refine_strength (e.g., 0.2).
    Refinement uses ControlNet to maintain structural fidelity.

Usage:
  python pipeline_stage2_diffusion.py \\
      --input_dir images/run1 --output_dir images/run1 --verbose
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "layout_generator"))  # required for unpickling CanvasState

import env  # noqa: E402
os.environ["HF_HOME"] = env.HF_HOME

import cv2
import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline,
)
from peft import PeftModel
from PIL import Image
from tqdm import tqdm

PROMPT = (
    "macro photo of printed circuit board, green solder mask, realistic, photorealistic"
)
NEGATIVE_PROMPT = "blurry, low quality, defects, damage, cracks, burned, cartoon, illustration, 3d render, flat colors, oversaturated, distorted"


def _apply_borders(img_bgr: np.ndarray, occupied_mask: np.ndarray, border_px: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    mask = cv2.resize(occupied_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_px, border_px))
    dilated = cv2.dilate(mask, kernel)
    ring = (dilated > 0) & (mask == 0)
    out = img_bgr.copy()
    out[ring] = (15, 55, 15)  # BGR dark green solder-mask border
    return out


def _sorted_layout_paths(folder: Path) -> list[tuple[int, Path]]:
    results = []
    for p in folder.iterdir():
        m = re.fullmatch(r"layout_(\d+)\.png", p.name)
        if m:
            results.append((int(m.group(1)), p))
    return sorted(results)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2: diffusion on pre-generated structure maps.")
    p.add_argument("--input_dir", type=str, default="images/pipeline")
    p.add_argument("--output_dir", type=str, default="images/pipeline")
    p.add_argument("--controlnet_path", type=str, default="trained/controlnet_aug_600x600")
    p.add_argument("--lora_path", type=str, default="trained/lora_aug_600x600")
    p.add_argument("--height", type=int, default=600)
    p.add_argument("--width", type=int, default=600)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=6.0)
    p.add_argument("--controlnet_scale", type=float, default=0.85)
    p.add_argument("--lora_scale", type=float, default=0.8)
    p.add_argument("--refine_strength", type=float, default=0.3)
    p.add_argument("--refine_controlnet_scale", type=float, default=0.6,
                   help="ControlNet scale for refinement (lower = more smoothing freedom)")
    p.add_argument("--border_px", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true",
                   help="Save intermediate images: raw, bordered, and final")
    p.add_argument("--skip_borders", action="store_true",
                   help="Skip border painting, refine directly from raw output")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    in_dir = _ROOT / args.input_dir
    out_images = _ROOT / args.output_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    structure_entries = _sorted_layout_paths(in_dir / "structure_maps")
    if not structure_entries:
        print(f"No layout_*.png files found in {in_dir / 'structure_maps'}")
        return

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    weight_dtype = torch.float16 if device.type == "cuda" else torch.float32
    gen_device = "cpu" if device.type == "mps" else device.type
    print(f"Device: {device}  |  dtype: {weight_dtype}")

    print("Loading ControlNet...")
    controlnet = ControlNetModel.from_pretrained(
        str(_ROOT / args.controlnet_path), torch_dtype=weight_dtype
    )

    print("Loading SDXL + LoRA...")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
    )
    pipe.unet = PeftModel.from_pretrained(pipe.unet, str(_ROOT / args.lora_path))
    for module in pipe.unet.modules():
        if hasattr(module, "scaling"):
            for key in module.scaling:
                module.scaling[key] = args.lora_scale
    pipe.unet.eval()

    pipe.to(device)
    pipe.vae.to(dtype=torch.float32)

    _original_decode = pipe.vae.decode
    def _decode_fp32(latents, **kwargs):
        return _original_decode(latents.to(torch.float32), **kwargs)
    pipe.vae.decode = _decode_fp32

    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()

    print("Building img2img refinement pipeline (with ControlNet)...")
    pipe_i2i = StableDiffusionXLControlNetImg2ImgPipeline(**pipe.components)

    skipped = 0
    for i, structure_path in tqdm(structure_entries, desc="Diffusion"):
        canvas_path = in_dir / "canvases" / f"layout_{i}.pkl.gz"
        if not canvas_path.exists():
            print(f"  [{i}] WARNING: canvas not found at {canvas_path}, skipping")
            skipped += 1
            continue

        try:
            with gzip.open(canvas_path, "rb") as f:
                canvas = pickle.load(f)
        except Exception as e:
            print(f"  [{i}] WARNING: failed to load canvas ({e}), skipping")
            skipped += 1
            continue

        structure_np = cv2.imread(str(structure_path), cv2.IMREAD_GRAYSCALE)
        resized_np = cv2.resize(structure_np, (args.width, args.height))
        structure_pil = Image.fromarray(np.stack([resized_np, resized_np, resized_np], axis=-1))

        gen = torch.Generator(device=gen_device).manual_seed(args.seed + i)
        raw_pil = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=structure_pil,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_scale,
            height=args.height,
            width=args.width,
            generator=gen,
        ).images[0]

        if args.verbose:
            raw_pil.save(out_images / f"layout_{i}_raw.png")

        if args.skip_borders:
            refine_input = raw_pil
        else:
            raw_bgr = cv2.cvtColor(np.array(raw_pil), cv2.COLOR_RGB2BGR)
            bordered_bgr = _apply_borders(raw_bgr, canvas.occupied_mask, args.border_px)
            refine_input = Image.fromarray(cv2.cvtColor(bordered_bgr, cv2.COLOR_BGR2RGB))
            if args.verbose:
                refine_input.save(out_images / f"layout_{i}_bordered.png")

        gen2 = torch.Generator(device=gen_device).manual_seed(args.seed + i + 10000)
        final_pil = pipe_i2i(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=refine_input,
            control_image=structure_pil,
            strength=args.refine_strength,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.refine_controlnet_scale,
            generator=gen2,
        ).images[0]

        if args.verbose:
            final_pil.save(out_images / f"layout_{i}_final.png")
        else:
            final_pil.save(out_images / f"layout_{i}.png")

    total = len(structure_entries)
    print(f"\nDone. Processed {total - skipped}/{total} layouts. Images saved to: {out_images}")


if __name__ == "__main__":
    main()
