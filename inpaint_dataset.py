#!/usr/bin/env python3
"""
Inpaint defective pads in PCB dataset by replacing them with clean pad assets.

For each image:
1. Parse YOLO annotations to find pad-related defects (mouse_bite, missing_hole, spurious_copper)
2. Classify defective pad shape (square/circle/oval)
3. Paste matching clean pad asset over the defect
4. Save cleaned image to output directory
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "layout_generator"))
from pad_classifier import classify_pad_shape


# Defect classes that affect pads (from data.yaml)
PAD_DEFECT_CLASSES = {
    0,  # mouse_bite - nibbled pad edges
    2,  # missing_hole - holes in pads
    5,  # spurious_copper - extra copper on/near pads
}


def load_replacement_pads(folder: Path) -> dict:
    """Load replacement pad images with alpha channels, cropped to content."""
    pads = {}
    for shape in ["circle", "oval", "square"]:
        path = folder / f"{shape}.png"
        if not path.exists():
            print(f"Warning: {path} not found")
            continue

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[2] != 4:
            print(f"Warning: {path} must be RGBA image")
            continue

        # Find bounding box of non-transparent pixels
        alpha = img[:, :, 3]
        coords = np.argwhere(alpha > 10)
        if len(coords) == 0:
            print(f"Warning: {path} has no visible content")
            continue

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)

        # Crop to content with small padding
        pad = 2
        y0 = max(0, y0 - pad)
        x0 = max(0, x0 - pad)
        y1 = min(img.shape[0] - 1, y1 + pad)
        x1 = min(img.shape[1] - 1, x1 + pad)

        cropped = img[y0:y1+1, x0:x1+1]
        pads[shape] = cropped
        print(f"Loaded {shape} pad: {cropped.shape[:2]}")

    return pads


def parse_yolo_label(label_path: Path, img_w: int, img_h: int) -> list:
    """Parse YOLO format label file, return list of (class_id, x1, y1, x2, y2)."""
    if not label_path.exists():
        return []

    detections = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1]) * img_w
            y_center = float(parts[2]) * img_h
            width = float(parts[3]) * img_w
            height = float(parts[4]) * img_h

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            detections.append((class_id, x1, y1, x2, y2))

    return detections


def expand_bbox(x1: int, y1: int, x2: int, y2: int,
                factor: float, img_w: int, img_h: int) -> tuple:
    """Expand bounding box by factor, clamped to image bounds."""
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    new_w = w * factor
    new_h = h * factor

    x1 = int(max(0, cx - new_w / 2))
    y1 = int(max(0, cy - new_h / 2))
    x2 = int(min(img_w - 1, cx + new_w / 2))
    y2 = int(min(img_h - 1, cy + new_h / 2))

    return x1, y1, x2, y2


def classify_region_shape(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> str:
    """Classify the shape of a pad in the given region."""
    region = img[y1:y2, x1:x2]
    if region.size == 0:
        return "circle"  # default fallback

    # Convert to grayscale
    if len(region.shape) == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = region

    # Threshold to get binary mask (pads are typically lighter than background)
    # Try adaptive thresholding to handle varying lighting
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use pad classifier
    shape, _ = classify_pad_shape(binary)

    if shape == "unknown":
        # Fallback: use aspect ratio
        h, w = region.shape[:2]
        aspect = w / h if h > 0 else 1.0
        if aspect < 0.8 or aspect > 1.25:
            return "oval"
        return "circle"

    return shape


def paste_pad(img: np.ndarray, pad_rgba: np.ndarray,
              x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Paste replacement pad onto image using alpha blending."""
    target_w = x2 - x1
    target_h = y2 - y1

    if target_w <= 0 or target_h <= 0:
        return img

    # Resize pad to target size
    resized = cv2.resize(pad_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # Split into BGR and alpha
    bgr = resized[:, :, :3]
    alpha = resized[:, :, 3:4].astype(np.float32) / 255.0

    # Get the target region
    roi = img[y1:y2, x1:x2].astype(np.float32)

    # Alpha blend
    blended = roi * (1 - alpha) + bgr.astype(np.float32) * alpha
    img[y1:y2, x1:x2] = blended.astype(np.uint8)

    return img


def process_image(img_path: Path, label_path: Path,
                  replacement_pads: dict, expand_factor: float = 1.3) -> tuple:
    """
    Process a single image, replacing defective pads.
    Returns (processed_image, num_replacements).
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None, 0

    img_h, img_w = img.shape[:2]
    detections = parse_yolo_label(label_path, img_w, img_h)

    num_replaced = 0
    for class_id, x1, y1, x2, y2 in detections:
        if class_id not in PAD_DEFECT_CLASSES:
            continue

        # Expand bbox to capture full pad (defect bbox might be smaller)
        x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, expand_factor, img_w, img_h)

        # Classify pad shape
        shape = classify_region_shape(img, x1, y1, x2, y2)

        # Get replacement pad
        if shape not in replacement_pads:
            shape = "circle"  # fallback
        if shape not in replacement_pads:
            continue

        pad = replacement_pads[shape]

        # Paste replacement
        img = paste_pad(img, pad, x1, y1, x2, y2)
        num_replaced += 1

    return img, num_replaced


def main():
    parser = argparse.ArgumentParser(description="Inpaint defective pads in PCB dataset")
    parser.add_argument("--images_dir", type=str,
                        default="pcb-defect-dataset/train/images",
                        help="Input images directory")
    parser.add_argument("--labels_dir", type=str,
                        default="pcb-defect-dataset/train/labels",
                        help="YOLO labels directory")
    parser.add_argument("--replacement_pads_dir", type=str,
                        default="images/replacement_pads",
                        help="Directory with replacement pad assets")
    parser.add_argument("--output_dir", type=str,
                        default="pcb-defect-dataset/train/images_clean",
                        help="Output directory for cleaned images")
    parser.add_argument("--expand_factor", type=float, default=1.3,
                        help="Factor to expand defect bbox to capture full pad")
    args = parser.parse_args()

    # Resolve paths relative to script location
    root = Path(__file__).parent
    images_dir = root / args.images_dir
    labels_dir = root / args.labels_dir
    pads_dir = root / args.replacement_pads_dir
    output_dir = root / args.output_dir

    # Load replacement pads
    print("Loading replacement pad assets...")
    replacement_pads = load_replacement_pads(pads_dir)
    if not replacement_pads:
        print("Error: No replacement pads loaded")
        return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all images
    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ])
    print(f"Found {len(image_files)} images to process")

    # Process each image
    total_replacements = 0
    images_with_replacements = 0

    for img_path in tqdm(image_files, desc="Processing"):
        # Find corresponding label
        label_path = labels_dir / (img_path.stem + ".txt")

        # Process
        result, num_replaced = process_image(
            img_path, label_path, replacement_pads, args.expand_factor
        )

        if result is None:
            print(f"Warning: Could not read {img_path}")
            continue

        # Save
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), result)

        total_replacements += num_replaced
        if num_replaced > 0:
            images_with_replacements += 1

    print(f"\nDone!")
    print(f"  Total images: {len(image_files)}")
    print(f"  Images with pad replacements: {images_with_replacements}")
    print(f"  Total pads replaced: {total_replacements}")
    print(f"  Output saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
