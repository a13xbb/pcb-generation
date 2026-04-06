from __future__ import annotations
import os
os.environ['HF_HOME'] = '/mnt/ssdm2/users/alexblokh/cache'

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from controlnet_dataset import PairedPCBControlNetDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SDXL ControlNet on paired structure/image data.")
    parser.add_argument("--images_dir", type=str, default="pcb-defect-dataset/train/images")
    parser.add_argument("--structures_dir", type=str, default="pcb-defect-dataset/train/structure_maps")
    parser.add_argument("--output_dir", type=str, default="structure_generation/pcb_controlnet")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default="diffusers/controlnet-canny-sdxl-1.0",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="macro photo of printed circuit board",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_train_pairs", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=20)
    return parser.parse_args()


def to_serializable_args(args: argparse.Namespace) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def is_finite_tensor(tensor: torch.Tensor) -> bool:
    return torch.isfinite(tensor).all().item()


def encode_prompt(
    tokenizer_one: CLIPTokenizer,
    tokenizer_two: CLIPTokenizer,
    text_encoder_one: CLIPTextModel,
    text_encoder_two: CLIPTextModelWithProjection,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
):
    text_inputs_1 = tokenizer_one(
        prompt,
        padding="max_length",
        max_length=tokenizer_one.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_inputs_2 = tokenizer_two(
        prompt,
        padding="max_length",
        max_length=tokenizer_two.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids_1 = text_inputs_1.input_ids.to(device)
    input_ids_2 = text_inputs_2.input_ids.to(device)

    with torch.no_grad():
        enc_1 = text_encoder_one(input_ids_1, output_hidden_states=True, return_dict=True)
        enc_2 = text_encoder_two(input_ids_2, output_hidden_states=True, return_dict=True)
        prompt_embeds = torch.cat([enc_1.hidden_states[-2], enc_2.hidden_states[-2]], dim=-1)
        pooled_prompt_embeds = enc_2.text_embeds if hasattr(enc_2, "text_embeds") else enc_2[0]

    return prompt_embeds.to(dtype=dtype), pooled_prompt_embeds.to(dtype=dtype)


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = PairedPCBControlNetDataset(
        images_dir=args.images_dir,
        structures_dir=args.structures_dir,
        resolution=args.resolution,
        max_pairs=args.max_train_pairs,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    (output_dir / "pair_stats.json").write_text(
        json.dumps(dataset.stats, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (output_dir / "training_args.json").write_text(
        json.dumps(to_serializable_args(args), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print(f"Paired training samples: {len(dataset)}")
    if dataset.stats["unmatched_images_count"] or dataset.stats["unmatched_structures_count"]:
        print(
            "Skipped unmatched files:",
            dataset.stats["unmatched_images_count"],
            "images,",
            dataset.stats["unmatched_structures_count"],
            "structure maps",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: CUDA is not available. SDXL ControlNet training on CPU will be very slow.")
    weight_dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
    )
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
    ).to(device)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        torch_dtype=weight_dtype,
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=weight_dtype,
    ).to(device)
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_model_name_or_path,
        torch_dtype=weight_dtype,
    ).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    unet.enable_gradient_checkpointing()
    controlnet.enable_gradient_checkpointing()
    try:
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()
        print("Enabled xformers memory-efficient attention")
    except Exception as exc:
        print(f"Could not enable xformers memory-efficient attention: {exc}")

    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.train()

    trainable_params = [parameter for parameter in controlnet.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable ControlNet parameters were found.")
    for parameter in trainable_params:
        parameter.data = parameter.data.float()

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and weight_dtype == torch.float16),
    )

    prompt_embeds, pooled_prompt_embeds = encode_prompt(
        tokenizer_one,
        tokenizer_two,
        text_encoder_one,
        text_encoder_two,
        args.prompt,
        device,
        weight_dtype,
    )
    add_time_ids = torch.tensor(
        [[args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]],
        dtype=weight_dtype,
        device=device,
    )

    global_step = 0
    skipped_steps = 0
    for epoch in range(args.epochs):
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            pixel_values = batch["pixel_values"].to(device=device, dtype=torch.float32, non_blocking=True)
            conditioning_pixel_values = batch["conditioning_pixel_values"].to(
                device=device,
                dtype=weight_dtype,
                non_blocking=True,
            )
            batch_size = pixel_values.shape[0]

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            if not is_finite_tensor(latents):
                skipped_steps += 1
                print(f"Skipping step {global_step + 1}: non-finite latents")
                continue

            latents = latents.to(dtype=weight_dtype)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            if noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                target = noise

            prompt_batch = prompt_embeds.expand(batch_size, -1, -1)
            pooled_batch = pooled_prompt_embeds.expand(batch_size, -1)
            time_ids_batch = add_time_ids.expand(batch_size, -1)
            added_cond_kwargs = {"text_embeds": pooled_batch, "time_ids": time_ids_batch}

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=weight_dtype, enabled=(device.type == "cuda")):
                down_block_residuals, mid_block_residual = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_batch,
                    controlnet_cond=conditioning_pixel_values,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_batch,
                    down_block_additional_residuals=down_block_residuals,
                    mid_block_additional_residual=mid_block_residual,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

            if not torch.isfinite(loss):
                skipped_steps += 1
                print(f"Skipping step {global_step + 1}: non-finite loss")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            if global_step % args.log_every == 0:
                print(f"step {global_step} loss {loss.item():.6f} skipped {skipped_steps}")
            progress.set_postfix(loss=f"{loss.item():.4f}", skipped=skipped_steps)

    controlnet.save_pretrained(output_dir)
    print(f"ControlNet saved to: {output_dir}")


if __name__ == "__main__":
    main()
