import os
import env
os.environ['HF_HOME'] = env.HF_HOME
import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDPMScheduler, StableDiffusionXLPipeline
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class PCBData(Dataset):
    def __init__(self, folder: str, resolution: int, max_images: int, seed: int) -> None:
        root = Path(folder)
        if not root.exists():
            raise FileNotFoundError(f"Dataset folder does not exist: {root}")

        image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        files = [str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in image_suffixes]
        if not files:
            raise RuntimeError(f"No images found in dataset folder: {root}")

        rng = random.Random(seed)
        if max_images > 0 and len(files) > max_images:
            files = rng.sample(files, max_images)
        files.sort()
        self.files = files

        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SDXL LoRA on a folder of PCB images.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Folder containing training images.")
    parser.add_argument("--output_dir", type=str, default="pcb_lora", help="Where to save LoRA weights.")
    parser.add_argument("--prompt", type=str, default="macro photo of printed circuit board", help="Training prompt.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base SDXL model id/path.",
    )
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_train_images", type=int, default=500, help="0 means use all images.")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=20)
    return parser.parse_args()


def is_finite_tensor(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item()


def encode_prompt(pipe: StableDiffusionXLPipeline, prompt: str, device: torch.device, dtype: torch.dtype):
    text_inputs_1 = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_inputs_2 = pipe.tokenizer_2(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids_1 = text_inputs_1.input_ids.to(device)
    input_ids_2 = text_inputs_2.input_ids.to(device)

    with torch.no_grad():
        enc_1 = pipe.text_encoder(input_ids_1, output_hidden_states=True, return_dict=True)
        enc_2 = pipe.text_encoder_2(input_ids_2, output_hidden_states=True, return_dict=True)
        prompt_embeds = torch.cat([enc_1.hidden_states[-2], enc_2.hidden_states[-2]], dim=-1)
        pooled_prompt_embeds = enc_2.text_embeds if hasattr(enc_2, "text_embeds") else enc_2[0]

    return prompt_embeds.to(dtype=dtype), pooled_prompt_embeds.to(dtype=dtype)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: CUDA is not available. SDXL LoRA training on CPU will be very slow.")
    weight_dtype = torch.float16 if device.type == "cuda" else torch.float32

    dataset = PCBData(args.dataset_dir, args.resolution, args.max_train_images, args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    print(f"Training images: {len(dataset)}")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
    )
    pipe.to(device)

    unet = pipe.unet
    vae = pipe.vae
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # SDXL VAE can produce NaNs in fp16 encode; run VAE in fp32 for stability.
    vae.to(dtype=torch.float32)
    vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)

    unet.enable_gradient_checkpointing()
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    unet.train()

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable LoRA parameters were found in UNet.")
    for param in trainable_params:
        param.data = param.data.float()
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"Trainable LoRA parameters: {trainable_count:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and weight_dtype == torch.float16))

    prompt_embeds, pooled_prompt_embeds = encode_prompt(pipe, args.prompt, device, weight_dtype)
    add_time_ids = torch.tensor(
        [[args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]],
        dtype=weight_dtype,
        device=device,
    )

    global_step = 0
    skipped_steps = 0

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for images in pbar:
            bsz = images.shape[0]

            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
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
                (bsz,),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            if noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                target = noise

            prompt_batch = prompt_embeds.expand(bsz, -1, -1)
            pooled_batch = pooled_prompt_embeds.expand(bsz, -1)
            time_ids_batch = add_time_ids.expand(bsz, -1)
            added_cond_kwargs = {"text_embeds": pooled_batch, "time_ids": time_ids_batch}

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=weight_dtype, enabled=(device.type == "cuda")):
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_batch,
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
            pbar.set_postfix(loss=f"{loss.item():.4f}", skipped=skipped_steps)

    os.makedirs(args.output_dir, exist_ok=True)
    unet.save_pretrained(args.output_dir)
    print(f"LoRA saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
