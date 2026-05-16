#!/usr/bin/env python3
"""Train YOLOv8 with on-the-fly grayscale/binarization augmentation.

This training script applies augmentations during training (not preprocessing):
1. Converts images to grayscale OR binarized (DeepPCB-style) on-the-fly
2. Uses YOLO's native scale augmentation for zoom variance
3. No disk preprocessing - starts training immediately

Usage:
    python train_grayscale_synth_otf.py --epochs 150
    python train_grayscale_synth_otf.py --binarize_p 0.3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from augmentations import to_grayscale, binarize_synth

ROOT = Path(__file__).parent.parent
DEFAULT_DATA = ROOT / "datasets/generated_dataset/v2/defects/data.yaml"
OUT = ROOT / "yolo_training/results/synth_grayscale_otf"


class GrayscaleAugDataset(YOLODataset):
    """Custom YOLO dataset with on-the-fly grayscale/binarize augmentation."""

    # Class-level config (set before instantiation)
    binarize_prob = 0.25

    def load_image(self, i):
        """Load image and apply grayscale/binarize augmentation."""
        im, hw_orig, hw_resized = super().load_image(i)

        if self.augment:  # Training mode
            if np.random.random() < self.binarize_prob:
                im = binarize_synth(im)
            else:
                im = to_grayscale(im)
        else:  # Validation mode - always grayscale
            im = to_grayscale(im)

        return im, hw_orig, hw_resized


class GrayscaleDetectionTrainer(DetectionTrainer):
    """Custom trainer that uses GrayscaleAugDataset."""

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset with custom grayscale augmentation."""
        # Get the standard dataset
        dataset = super().build_dataset(img_path, mode, batch)

        # Replace with our custom dataset class
        # We need to copy attributes and change the class
        dataset.__class__ = GrayscaleAugDataset

        return dataset


def parse_args():
    p = argparse.ArgumentParser(
        description="Train YOLOv8 with on-the-fly grayscale augmentation"
    )
    p.add_argument(
        "--model",
        default=str(ROOT / "models/yolov8m.pt"),
        help="Base weights (e.g. yolov8m.pt)",
    )
    p.add_argument(
        "--data", default=str(DEFAULT_DATA), help="Path to dataset data.yaml"
    )
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--binarize_p",
        type=float,
        default=0.25,
        help="Probability of binarization augmentation (vs grayscale)",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=0.9,
        help="YOLO scale augmentation factor (0.9 = zoom range 0.1x to 1.9x)",
    )
    p.add_argument("--name", default="grayscale_otf", help="Run name")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Grayscale Synthetic PCB Training (On-The-Fly)")
    print("=" * 60)
    print(f"Dataset: {args.data}")
    print(f"Binarization probability: {args.binarize_p}")
    print(f"Scale augmentation: {args.scale} (range {1-args.scale:.1f}x to {1+args.scale:.1f}x)")
    print("=" * 60)

    # Set binarization probability for custom dataset
    GrayscaleAugDataset.binarize_prob = args.binarize_p

    # Initialize model with custom trainer
    model = YOLO(args.model)

    # Override the trainer class
    model.trainer_class = GrayscaleDetectionTrainer

    # Train with YOLO's built-in augmentations
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        lr0=args.lr,
        project=str(OUT),
        name=args.name,
        exist_ok=True,
        patience=20,
        save=True,
        plots=True,
        # YOLO augmentations
        hsv_h=0.0,  # Disable hue (grayscale)
        hsv_s=0.0,  # Disable saturation (grayscale)
        hsv_v=0.4,  # Keep value augmentation
        degrees=0.0,
        translate=0.1,
        scale=args.scale,  # More aggressive zoom for scale variance
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
    )

    best = OUT / args.name / "weights/best.pt"
    print(f"\nBest weights: {best}")


if __name__ == "__main__":
    main()
