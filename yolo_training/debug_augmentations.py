#!/usr/bin/env python3
"""Debug script to visualize augmentations with bboxes.

Generates example images showing what the training script produces:
- Original image
- Grayscale version
- Binarized version
- Cropped versions (with adjusted bboxes)

All with bounding boxes drawn.

Usage:
    python yolo_training/debug_augmentations.py
    python yolo_training/debug_augmentations.py --n_images 20
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from augmentations import to_grayscale, binarize_synth, random_crop_zoom

ROOT = Path(__file__).parent.parent
DEFAULT_DATA = ROOT / "datasets/generated_dataset/v2/defects/data.yaml"
OUTPUT_DIR = ROOT / "debug_output/grayscale_aug_examples"

CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "open_circuit",
    4: "spurious_copper",
}

COLORS = {
    0: (255, 0, 0),      # mouse_bite - blue
    1: (0, 255, 0),      # spur - green
    2: (0, 0, 255),      # missing_hole - red
    3: (255, 255, 0),    # open_circuit - cyan
    4: (255, 0, 255),    # spurious_copper - magenta
}


def draw_bboxes(img: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Draw bounding boxes on image.

    Args:
        img: BGR image (H, W, 3)
        labels: YOLO format (N, 5) - [class_id, cx, cy, w, h] normalized

    Returns:
        Image with bboxes drawn
    """
    img = img.copy()
    h, w = img.shape[:2]

    for label in labels:
        cls_id = int(label[0])
        cx, cy, bw, bh = label[1:5]

        # Convert to pixel coordinates
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)

        color = COLORS.get(cls_id, (128, 128, 128))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label_text = CLASS_NAMES.get(cls_id, str(cls_id))
        cv2.putText(img, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img


def load_labels(label_path: Path) -> np.ndarray:
    """Load YOLO format labels."""
    labels = []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels.append([float(x) for x in parts[:5]])
    return np.array(labels) if labels else np.array([])


def generate_examples(
    data_yaml: Path,
    output_dir: Path,
    n_images: int = 10,
    binarize_prob: float = 0.25,
    crop_prob: float = 0.25,
    crop_scale_range: tuple = (0.3, 0.7),
):
    """Generate example augmented images with bboxes."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data config
    with open(data_yaml) as f:
        data_config = yaml.safe_load(f)

    src_path = Path(data_config["path"])
    train_split = data_config.get("train", "images")

    # Get image paths
    if train_split.endswith(".txt"):
        txt_path = src_path / train_split
        with open(txt_path) as f:
            image_paths = [line.strip() for line in f if line.strip()]
    else:
        images_dir = src_path / train_split
        if not images_dir.exists():
            images_dir = src_path / "images" / train_split
        image_paths = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
        image_paths = [str(p) for p in image_paths]

    # Labels directory
    labels_dir = src_path / "labels"

    # Select random images
    np.random.seed(42)
    selected_indices = np.random.choice(len(image_paths), min(n_images, len(image_paths)), replace=False)

    print(f"Generating {len(selected_indices)} example images to {output_dir}")

    for idx, i in enumerate(selected_indices):
        img_path = Path(image_paths[i])
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        labels = load_labels(labels_dir / (img_path.stem + ".txt"))

        base_name = f"{idx:02d}_{img_path.stem}"

        # 1. Original with bboxes
        orig_with_boxes = draw_bboxes(img, labels)
        cv2.imwrite(str(output_dir / f"{base_name}_0_original.jpg"), orig_with_boxes)

        # 2. Grayscale with bboxes
        gray = to_grayscale(img)
        gray_with_boxes = draw_bboxes(gray, labels)
        cv2.imwrite(str(output_dir / f"{base_name}_1_grayscale.jpg"), gray_with_boxes)

        # 3. Binarized with bboxes
        binary = binarize_synth(img)
        binary_with_boxes = draw_bboxes(binary, labels)
        cv2.imwrite(str(output_dir / f"{base_name}_2_binarized.jpg"), binary_with_boxes)

        # 4. Cropped grayscale (if has labels)
        if len(labels) > 0:
            crop_img, crop_labels = random_crop_zoom(img.copy(), labels.copy(), crop_scale_range)
            crop_gray = to_grayscale(crop_img)
            crop_with_boxes = draw_bboxes(crop_gray, crop_labels)
            cv2.imwrite(str(output_dir / f"{base_name}_3_crop_gray.jpg"), crop_with_boxes)

            # 5. Cropped binarized
            crop_img2, crop_labels2 = random_crop_zoom(img.copy(), labels.copy(), crop_scale_range)
            crop_bin = binarize_synth(crop_img2)
            crop_bin_with_boxes = draw_bboxes(crop_bin, crop_labels2)
            cv2.imwrite(str(output_dir / f"{base_name}_4_crop_bin.jpg"), crop_bin_with_boxes)

        print(f"  [{idx+1}/{len(selected_indices)}] {img_path.name}")

    print(f"\nDone! Examples saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Debug augmentations visualization")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Path to data.yaml")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--n_images", type=int, default=10, help="Number of images to generate")
    parser.add_argument("--binarize_p", type=float, default=0.25, help="Binarization probability")
    parser.add_argument("--crop_p", type=float, default=0.25, help="Crop probability")
    args = parser.parse_args()

    generate_examples(
        Path(args.data),
        Path(args.output),
        n_images=args.n_images,
        binarize_prob=args.binarize_p,
        crop_prob=args.crop_p,
    )


if __name__ == "__main__":
    main()
