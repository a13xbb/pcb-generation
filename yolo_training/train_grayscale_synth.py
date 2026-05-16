#!/usr/bin/env python3
"""Train YOLOv8 on grayscaled synthetic PCB dataset with custom augmentations.

This training script:
1. Converts all images to grayscale (base representation)
2. Applies binarization augmentation (25% probability) for DeepPCB-style
3. Applies zoom-in crop augmentation (25% probability) for scale variance
4. Uses YOLO's built-in augmentations (flip, mosaic, scale)

Usage:
    python train_grayscale_synth.py --epochs 150
    python train_grayscale_synth.py --binarize_p 0.3 --crop_p 0.2
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from augmentations import GrayscaleAugmentor, to_grayscale, binarize_synth

ROOT = Path(__file__).parent.parent
DEFAULT_DATA = ROOT / "datasets/generated_dataset/v2_demo/defects_v3/data.yaml"
OUT = ROOT / "trained/yolo_grayscale_synth"


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 on grayscaled synthetic dataset")
    p.add_argument("--model", default=str(ROOT / "yolov8m.pt"),
                   help="Base weights (e.g. yolov8m.pt)")
    p.add_argument("--data", default=str(DEFAULT_DATA),
                   help="Path to dataset data.yaml")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--binarize_p", type=float, default=0.25,
                   help="Probability of binarization augmentation")
    p.add_argument("--crop_p", type=float, default=0.25,
                   help="Probability of zoom-in crop augmentation")
    p.add_argument("--crop_scale_min", type=float, default=0.3,
                   help="Minimum crop scale (zoom-in)")
    p.add_argument("--crop_scale_max", type=float, default=0.7,
                   help="Maximum crop scale (zoom-in)")
    p.add_argument("--name", default="grayscale_synth",
                   help="Run name")
    p.add_argument("--preprocess", action="store_true",
                   help="Preprocess dataset to grayscale (saves to temp dir)")
    return p.parse_args()


def preprocess_dataset_to_grayscale(data_yaml: Path, augmentor: GrayscaleAugmentor) -> Path:
    """Preprocess dataset: convert images to grayscale and save to temp directory.

    This approach applies augmentations during preprocessing rather than training.
    For a cleaner approach, YOLO's callback system could be used, but this is simpler.
    """
    import yaml

    with open(data_yaml) as f:
        data_config = yaml.safe_load(f)

    src_path = Path(data_config["path"])
    temp_dir = Path(tempfile.mkdtemp(prefix="yolo_grayscale_"))

    print(f"Preprocessing dataset to: {temp_dir}")

    # Process train and val splits
    for split in ["train", "val"]:
        if split not in data_config:
            continue

        split_dir = data_config[split]
        src_images = src_path / split_dir
        src_labels = src_path / "labels"

        dst_images = temp_dir / "images" / split
        dst_labels = temp_dir / "labels" / split
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        for img_path in src_images.glob("*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            # Load and convert to grayscale
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            gray_img = to_grayscale(img)

            # Save grayscale image
            dst_img_path = dst_images / img_path.name
            cv2.imwrite(str(dst_img_path), gray_img)

            # Copy label file
            label_path = src_labels / (img_path.stem + ".txt")
            if label_path.exists():
                shutil.copy(label_path, dst_labels / label_path.name)

    # Create new data.yaml
    new_data_yaml = temp_dir / "data.yaml"
    new_config = {
        "path": str(temp_dir),
        "train": "images/train" if "train" in data_config else "images",
        "val": "images/val" if "val" in data_config else "images",
        "names": data_config["names"]
    }

    with open(new_data_yaml, "w") as f:
        yaml.dump(new_config, f)

    return new_data_yaml


def create_augmented_dataset(
    data_yaml: Path,
    binarize_prob: float = 0.25,
    crop_prob: float = 0.25,
    crop_scale_range: tuple = (0.3, 0.7),
    augment_multiplier: int = 3
) -> Path:
    """Create augmented grayscale dataset with binarization and crop.

    For each original image, creates:
    - 1 grayscale version (always)
    - Additional augmented versions based on multiplier

    This pre-generates augmented samples so YOLO trains on them directly.
    """
    import yaml
    from augmentations import to_grayscale, binarize_synth, random_crop_zoom

    with open(data_yaml) as f:
        data_config = yaml.safe_load(f)

    src_path = Path(data_config["path"])
    temp_dir = Path(tempfile.mkdtemp(prefix="yolo_grayscale_aug_"))

    print(f"Creating augmented grayscale dataset at: {temp_dir}")
    print(f"  Binarization prob: {binarize_prob}")
    print(f"  Crop prob: {crop_prob}")
    print(f"  Augment multiplier: {augment_multiplier}")

    # Get image and label directories
    images_dir = src_path / data_config.get("train", "images")
    labels_dir = src_path / "labels"

    dst_images = temp_dir / "images"
    dst_labels = temp_dir / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    img_count = 0
    aug_count = 0

    for img_path in sorted(images_dir.glob("*")):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Load labels
        label_path = labels_dir / (img_path.stem + ".txt")
        labels = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        labels.append([float(x) for x in parts[:5]])
        labels = np.array(labels) if labels else np.array([])

        # 1. Always save grayscale version
        gray_img = to_grayscale(img)
        dst_path = dst_images / f"{img_path.stem}_gray.jpg"
        cv2.imwrite(str(dst_path), gray_img)
        if len(labels) > 0:
            with open(dst_labels / f"{img_path.stem}_gray.txt", "w") as f:
                for lbl in labels:
                    f.write(f"{int(lbl[0])} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")
        img_count += 1

        # 2. Generate augmented versions
        for aug_idx in range(augment_multiplier):
            aug_img = img.copy()
            aug_labels = labels.copy() if len(labels) > 0 else np.array([])
            suffix = f"_aug{aug_idx}"

            # Random crop
            if np.random.random() < crop_prob and len(aug_labels) > 0:
                aug_img, aug_labels = random_crop_zoom(aug_img, aug_labels, crop_scale_range)
                suffix += "_crop"

            # Binarize or grayscale
            if np.random.random() < binarize_prob:
                aug_img = binarize_synth(aug_img)
                suffix += "_bin"
            else:
                aug_img = to_grayscale(aug_img)

            # Save augmented version
            dst_path = dst_images / f"{img_path.stem}{suffix}.jpg"
            cv2.imwrite(str(dst_path), aug_img)

            if len(aug_labels) > 0:
                with open(dst_labels / f"{img_path.stem}{suffix}.txt", "w") as f:
                    for lbl in aug_labels:
                        f.write(f"{int(lbl[0])} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

            aug_count += 1

    print(f"Created {img_count} grayscale + {aug_count} augmented images")
    print(f"Total training images: {img_count + aug_count}")

    # Create data.yaml
    new_data_yaml = temp_dir / "data.yaml"
    new_config = {
        "path": str(temp_dir),
        "train": "images",
        "val": "images",
        "names": data_config["names"]
    }

    with open(new_data_yaml, "w") as f:
        yaml.dump(new_config, f)

    return new_data_yaml


def main():
    args = parse_args()

    print("=" * 60)
    print("Grayscale Synthetic PCB Training")
    print("=" * 60)
    print(f"Dataset: {args.data}")
    print(f"Binarization probability: {args.binarize_p}")
    print(f"Crop probability: {args.crop_p}")
    print(f"Crop scale range: {args.crop_scale_min}-{args.crop_scale_max}")
    print("=" * 60)

    # Create augmented grayscale dataset
    data_yaml = Path(args.data)
    grayscale_data = create_augmented_dataset(
        data_yaml,
        binarize_prob=args.binarize_p,
        crop_prob=args.crop_p,
        crop_scale_range=(args.crop_scale_min, args.crop_scale_max),
        augment_multiplier=3  # 3 augmented versions per image
    )

    print(f"\nUsing grayscale dataset: {grayscale_data}")

    # Initialize model
    model = YOLO(args.model)

    # Train with YOLO's built-in augmentations
    # Note: Custom binarization/crop would need a callback or custom dataset
    # For now, we train on grayscale with YOLO's standard augmentations
    model.train(
        data=str(grayscale_data),
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
        scale=0.5,
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

    # Cleanup temp directory
    print(f"\nNote: Temporary grayscale dataset at {grayscale_data.parent}")
    print("You may want to delete it manually after training.")


if __name__ == "__main__":
    main()
