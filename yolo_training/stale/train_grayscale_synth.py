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
import tempfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from augmentations import to_grayscale, binarize_synth

ROOT = Path(__file__).parent.parent
DEFAULT_DATA = ROOT / "datasets/generated_dataset/v2/defects/data.yaml"
OUT = ROOT / "yolo_training/results/synth_grayscale_aug"


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 on grayscaled synthetic dataset")
    p.add_argument("--model", default=str(ROOT / "models/yolov8m.pt"),
                   help="Base weights (e.g. yolov8m.pt)")
    p.add_argument("--data", default=str(DEFAULT_DATA),
                   help="Path to dataset data.yaml")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
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
    p.add_argument("--aug_mult", type=int, default=3,
                   help="Augmentation multiplier per image")
    p.add_argument("--name", default="grayscale_synth",
                   help="Run name")
    p.add_argument("--tmp_dir", default=str(ROOT),
                   help="Directory for temporary augmented dataset")
    return p.parse_args()

def process_split(
    image_paths: list,
    src_labels_dir: Path,
    dst_images_dir: Path,
    dst_labels_dir: Path,
    binarize_prob: float,
    crop_prob: float,
    crop_scale_range: tuple,
    augment_multiplier: int,
    apply_augmentation: bool = True,
    desc: str = "Processing"
) -> tuple:
    """Process a single split (train or val)."""
    from augmentations import to_grayscale, binarize_synth, random_crop_zoom

    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    img_count = 0
    aug_count = 0

    for img_path in tqdm(sorted(image_paths), desc=desc):
        img_path = Path(img_path)
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Load labels
        label_path = src_labels_dir / (img_path.stem + ".txt")
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
        dst_path = dst_images_dir / f"{img_path.stem}_gray.jpg"
        cv2.imwrite(str(dst_path), gray_img)
        if len(labels) > 0:
            with open(dst_labels_dir / f"{img_path.stem}_gray.txt", "w") as f:
                for lbl in labels:
                    f.write(f"{int(lbl[0])} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")
        img_count += 1

        # 2. Generate augmented versions (only for training, not validation)
        if apply_augmentation:
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
                dst_path = dst_images_dir / f"{img_path.stem}{suffix}.jpg"
                cv2.imwrite(str(dst_path), aug_img)

                if len(aug_labels) > 0:
                    with open(dst_labels_dir / f"{img_path.stem}{suffix}.txt", "w") as f:
                        for lbl in aug_labels:
                            f.write(f"{int(lbl[0])} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

                aug_count += 1

    return img_count, aug_count


def create_augmented_dataset(
    data_yaml: Path,
    binarize_prob: float = 0.25,
    crop_prob: float = 0.25,
    crop_scale_range: tuple = (0.3, 0.7),
    augment_multiplier: int = 3,
    tmp_dir: Path = None
) -> Path:
    """Create augmented grayscale dataset with binarization and crop.

    For each training image, creates:
    - 1 grayscale version (always)
    - Additional augmented versions based on multiplier

    Validation images are only converted to grayscale (no augmentation).
    """
    import yaml

    with open(data_yaml) as f:
        data_config = yaml.safe_load(f)

    src_path = Path(data_config["path"])
    temp_dir = Path(tempfile.mkdtemp(prefix="yolo_grayscale_aug_", dir=tmp_dir))

    print(f"Creating augmented grayscale dataset at: {temp_dir}")
    print(f"  Binarization prob: {binarize_prob}")
    print(f"  Crop prob: {crop_prob}")
    print(f"  Augment multiplier: {augment_multiplier}")

    # Determine dataset structure
    train_split = data_config.get("train", "images")
    val_split = data_config.get("val", train_split)

    # Helper to get image paths from either txt file list or directory
    def get_image_paths(split_value: str) -> list:
        if split_value.endswith(".txt"):
            txt_path = src_path / split_value
            if txt_path.exists():
                paths = []
                with open(txt_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        p = Path(line)
                        if p.exists():
                            paths.append(str(p))
                        else:
                            # Try relative to src_path/images using just filename
                            relative_p = src_path / "images" / p.name
                            if relative_p.exists():
                                paths.append(str(relative_p))
                return paths
        # Fall back to directory globbing
        split_dir = src_path / split_value
        if not split_dir.exists():
            split_dir = src_path / "images" / split_value
        if split_dir.exists() and split_dir.is_dir():
            return list(split_dir.glob("*"))
        return []

    train_image_paths = get_image_paths(train_split)
    val_image_paths = get_image_paths(val_split)

    # Labels directory - check common structures
    if (src_path / "labels" / "train").exists():
        train_labels_dir = src_path / "labels" / "train"
        val_labels_dir = src_path / "labels" / "val"
    elif (src_path / "train" / "labels").exists():
        train_labels_dir = src_path / "train" / "labels"
        val_labels_dir = src_path / "val" / "labels"
    else:
        train_labels_dir = src_path / "labels"
        val_labels_dir = src_path / "labels"

    print(f"  Train images: {len(train_image_paths)} files")
    print(f"  Train labels: {train_labels_dir}")
    print(f"  Val images: {len(val_image_paths)} files")
    print(f"  Val labels: {val_labels_dir}")

    # Process training set (with augmentation)
    train_img_count, train_aug_count = process_split(
        train_image_paths, train_labels_dir,
        temp_dir / "images" / "train", temp_dir / "labels" / "train",
        binarize_prob, crop_prob, crop_scale_range, augment_multiplier,
        apply_augmentation=True,
        desc="Train (gray+aug)"
    )
    print(f"  Train: {train_img_count} grayscale + {train_aug_count} augmented = {train_img_count + train_aug_count} total")

    # Process validation set (grayscale only, no augmentation)
    val_img_count, _ = process_split(
        val_image_paths, val_labels_dir,
        temp_dir / "images" / "val", temp_dir / "labels" / "val",
        binarize_prob, crop_prob, crop_scale_range, augment_multiplier,
        apply_augmentation=False,
        desc="Val (gray only)"
    )
    print(f"  Val: {val_img_count} grayscale images")

    # Create data.yaml
    new_data_yaml = temp_dir / "data.yaml"
    new_config = {
        "path": str(temp_dir),
        "train": "images/train",
        "val": "images/val",
        "names": data_config["names"]
    }

    with open(new_data_yaml, "w") as f:
        yaml.dump(new_config, f)

    print(f"\nDataset ready: {new_data_yaml}")
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
        augment_multiplier=args.aug_mult,
        tmp_dir=Path(args.tmp_dir)
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
