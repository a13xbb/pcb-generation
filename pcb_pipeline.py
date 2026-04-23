#!/usr/bin/env python3
"""End-to-end PCB synthesis pipeline.

Steps per layout:
  1. Generate graph layout  →  images/layouts/layout_N.png
  2. Derive structure map   →  images/structure_maps/layout_N.png
  3. ControlNet + LoRA      →  images/generated/layout_N_raw.png
  4. Add dark borders       →  images/generated/layout_N_bordered.png
  5. ControlNet img2img     →  images/generated/layout_N_final.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import env  # sets HF_HOME
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

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "layout_generator"))
sys.path.insert(0, str(_ROOT / "structure_generation"))

from classes import CanvasState, PadAsset, TraceAsset  # noqa: E402
from generator import canvas_coverage, generate_layout_motif_based  # noqa: E402
from image_to_structure import image_to_structure  # noqa: E402
from placement_engine import visualize_canvas_real  # noqa: E402
from utils import (  # noqa: E402
    compute_bbox_and_centroid,
    find_endpoints,
    skeleton_path_length_between_endpoints,
    skeletonize,
)

PROMPT = (
    "macro photo of printed circuit board, green solder mask, "
    "copper traces, vias, electronic components, realistic, photorealistic"
)
NEGATIVE_PROMPT = "blurry, low quality, defects, damage, cracks, burned"


def _load_pad_assets(folder: Path) -> list[PadAsset]:
    assets = []
    for name in os.listdir(folder):
        if not (name.startswith("mask") and name.endswith(".png")):
            continue
        mask = cv2.imread(str(folder / name), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = (mask > 0).astype(np.uint8)
        obj = cv2.imread(str(folder / name.replace("mask", "object", 1)))
        if obj is None:
            obj = np.full((*mask.shape, 3), 160, dtype=np.uint8)
        bbox, centroid = compute_bbox_and_centroid(mask)
        assets.append(PadAsset(mask, obj, centroid, bbox))
    return assets


def _load_trace_assets(folder: Path) -> list[TraceAsset]:
    assets = []
    for name in sorted(os.listdir(folder)):
        if not (name.startswith("mask") and name.endswith(".png")):
            continue
        mask = cv2.imread(str(folder / name), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = (mask > 0).astype(np.uint8)
        skel = (skeletonize(mask) > 0).astype(np.uint8)
        endpoints = find_endpoints(mask, skel)
        if len(endpoints) != 2:
            continue
        length = skeleton_path_length_between_endpoints(skel, endpoints[0], endpoints[1]) or 0.0
        bbox, centroid = compute_bbox_and_centroid(mask)
        obj = cv2.imread(str(folder / name.replace("mask", "object", 1)))
        if obj is None:
            obj = np.full((*mask.shape, 3), 160, dtype=np.uint8)
        assets.append(TraceAsset(mask, obj, skel, endpoints, centroid, length))
    return assets


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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end PCB synthesis pipeline.")
    p.add_argument("--n_layouts", type=int, default=5)
    p.add_argument("--controlnet_path", type=str, default="trained/controlnet_600x600")
    p.add_argument("--lora_path", type=str, default="trained/lora_600x600")
    p.add_argument("--output_dir", type=str, default="images/generated")
    p.add_argument("--layout_dir", type=str, default="images/layouts")
    p.add_argument("--structure_dir", type=str, default="images/structure_maps")
    p.add_argument("--height", type=int, default=600)
    p.add_argument("--width", type=int, default=600)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=8.0)
    p.add_argument("--controlnet_scale", type=float, default=0.8)
    p.add_argument("--lora_scale", type=float, default=0.8)
    p.add_argument("--refine_strength", type=float, default=0.25)
    p.add_argument("--border_px", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    out_dir = _ROOT / args.output_dir
    layout_dir = _ROOT / args.layout_dir
    structure_dir = _ROOT / args.structure_dir
    for d in (out_dir, layout_dir, structure_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("Loading pad and trace assets...")
    assets_root = _ROOT / "assets"
    pad_assets = _load_pad_assets(assets_root / "PADS")
    trace_assets = _load_trace_assets(assets_root / "COPPER_TRACES")
    assert pad_assets and trace_assets, "No assets found"
    print(f"  {len(pad_assets)} pads, {len(trace_assets)} traces")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16 if device.type == "cuda" else torch.float32

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
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()

    # Refinement pipeline shares the same loaded components — no extra VRAM.
    print("Building refinement (img2img) pipeline from shared components...")
    pipe_i2i = StableDiffusionXLControlNetImg2ImgPipeline(**pipe.components)

    for i in tqdm(range(args.n_layouts), desc="Layouts"):
        # --- Step 1: generate layout ---
        canvas = CanvasState(600, 600, pad_keepout_radius=8)
        ok = generate_layout_motif_based(
            canvas=canvas,
            pad_assets=pad_assets,
            trace_assets=trace_assets,
            n_cols=5,
            n_rows=4,
            edge_padding=10,
            max_motif_attempts=8,
        )
        layout_path = layout_dir / f"layout_{i}.png"
        visualize_canvas_real(canvas, str(layout_path))
        print(f"  [{i}] layout ok={ok}  coverage={canvas_coverage(canvas):.3f}")

        # --- Step 2: structure map ---
        layout_bgr = cv2.imread(str(layout_path))
        structure_np = image_to_structure(layout_bgr, return_rgb=False)
        structure_path = structure_dir / f"layout_{i}.png"
        cv2.imwrite(str(structure_path), structure_np)
        structure_pil = Image.fromarray(
            cv2.resize(structure_np, (args.width, args.height))
        )

        gen = torch.Generator(device=device.type).manual_seed(args.seed + i)

        # --- Step 3: ControlNet + LoRA text→image ---
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
        raw_pil.save(out_dir / f"layout_{i}_raw.png")

        # --- Step 4: paint dark borders from the exact occupied mask ---
        raw_bgr = cv2.cvtColor(np.array(raw_pil), cv2.COLOR_RGB2BGR)
        bordered_bgr = _apply_borders(raw_bgr, canvas.occupied_mask, args.border_px)
        bordered_pil = Image.fromarray(cv2.cvtColor(bordered_bgr, cv2.COLOR_BGR2RGB))
        bordered_pil.save(out_dir / f"layout_{i}_bordered.png")

        # --- Step 5: ControlNet img2img refinement to blend borders ---
        gen2 = torch.Generator(device=device.type).manual_seed(args.seed + i)
        final_pil = pipe_i2i(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=bordered_pil,
            control_image=structure_pil,
            strength=args.refine_strength,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_scale,
            generator=gen2,
        ).images[0]
        final_pil.save(out_dir / f"layout_{i}_final.png")
        print(f"  [{i}] saved raw / bordered / final")

    print("Done.")


if __name__ == "__main__":
    main()
