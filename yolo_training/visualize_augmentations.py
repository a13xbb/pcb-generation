#!/usr/bin/env python3
"""Visualize augmentations for grayscale PCB training.

Generates sample images showing:
- Original (color)
- Grayscale
- Binarized (DeepPCB style)
- Zoomed crops at different scales

Saves to debug_output/training_augmentations/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np

from augmentations import to_grayscale, binarize_synth, random_crop_zoom

ROOT = Path(__file__).parent.parent
DATASET_DIR = ROOT / "datasets/generated_dataset/v2_demo/defects_v3"
OUTPUT_DIR = ROOT / "debug_output/training_augmentations"


def draw_boxes(img: np.ndarray, labels: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    """Draw bounding boxes on image."""
    h, w = img.shape[:2]
    result = img.copy()

    for label in labels:
        cls_id, cx, cy, bw, bh = label

        # Convert to pixel coordinates
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

    return result


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


def visualize_single_image(img_path: Path, output_prefix: str):
    """Generate all augmentation variations for a single image."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Could not load: {img_path}")
        return

    # Load labels
    label_path = DATASET_DIR / "labels" / (img_path.stem + ".txt")
    labels = load_labels(label_path)

    # 1. Original (color) with boxes
    original_boxes = draw_boxes(img, labels, (0, 255, 0))
    cv2.imwrite(str(OUTPUT_DIR / f"{output_prefix}_1_original.jpg"), original_boxes)

    # 2. Grayscale
    gray = to_grayscale(img)
    gray_boxes = draw_boxes(gray, labels, (0, 255, 0))
    cv2.imwrite(str(OUTPUT_DIR / f"{output_prefix}_2_grayscale.jpg"), gray_boxes)

    # 3. Binarized
    binary = binarize_synth(img)
    binary_boxes = draw_boxes(binary, labels, (128, 128, 128))
    cv2.imwrite(str(OUTPUT_DIR / f"{output_prefix}_3_binarized.jpg"), binary_boxes)

    # 4. Zoom crop 50%
    if len(labels) > 0:
        np.random.seed(42)  # Reproducible
        crop50, labels50 = random_crop_zoom(img, labels.copy(), (0.5, 0.5))
        crop50_gray = to_grayscale(crop50)
        crop50_boxes = draw_boxes(crop50_gray, labels50, (0, 255, 0))
        cv2.imwrite(str(OUTPUT_DIR / f"{output_prefix}_4_crop50_gray.jpg"), crop50_boxes)

    # 5. Zoom crop 30%
    if len(labels) > 0:
        np.random.seed(123)
        crop30, labels30 = random_crop_zoom(img, labels.copy(), (0.3, 0.3))
        crop30_gray = to_grayscale(crop30)
        crop30_boxes = draw_boxes(crop30_gray, labels30, (0, 255, 0))
        cv2.imwrite(str(OUTPUT_DIR / f"{output_prefix}_5_crop30_gray.jpg"), crop30_boxes)

    # 6. Binarized + crop
    if len(labels) > 0:
        np.random.seed(456)
        crop_bin, labels_bin = random_crop_zoom(img, labels.copy(), (0.4, 0.4))
        crop_bin = binarize_synth(crop_bin)
        crop_bin_boxes = draw_boxes(crop_bin, labels_bin, (128, 128, 128))
        cv2.imwrite(str(OUTPUT_DIR / f"{output_prefix}_6_crop_binarized.jpg"), crop_bin_boxes)


def create_comparison_grid(images: list, output_path: Path, cols: int = 3):
    """Create a grid of images for comparison."""
    if not images:
        return

    # Resize all to same size
    size = (300, 300)
    resized = [cv2.resize(img, size) for img in images]

    rows = (len(resized) + cols - 1) // cols
    grid_h = rows * size[1]
    grid_w = cols * size[0]
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

    for i, img in enumerate(resized):
        r, c = i // cols, i % cols
        y, x = r * size[1], c * size[0]
        grid[y:y+size[1], x:x+size[0]] = img

    cv2.imwrite(str(output_path), grid)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get sample images
    images_dir = DATASET_DIR / "images"
    image_files = sorted(images_dir.glob("*.png"))[:5]  # First 5 images

    if not image_files:
        print(f"No images found in {images_dir}")
        return

    print(f"Generating augmentation visualizations for {len(image_files)} images...")
    print(f"Output directory: {OUTPUT_DIR}")

    for i, img_path in enumerate(image_files):
        print(f"  Processing: {img_path.name}")
        visualize_single_image(img_path, f"img{i:02d}")

    # Create summary grid
    print("\nCreating comparison grids...")

    # Grid 1: All grayscale versions
    gray_images = []
    for i in range(len(image_files)):
        path = OUTPUT_DIR / f"img{i:02d}_2_grayscale.jpg"
        if path.exists():
            gray_images.append(cv2.imread(str(path)))
    create_comparison_grid(gray_images, OUTPUT_DIR / "grid_grayscale.jpg", cols=5)

    # Grid 2: All binarized versions
    bin_images = []
    for i in range(len(image_files)):
        path = OUTPUT_DIR / f"img{i:02d}_3_binarized.jpg"
        if path.exists():
            bin_images.append(cv2.imread(str(path)))
    create_comparison_grid(bin_images, OUTPUT_DIR / "grid_binarized.jpg", cols=5)

    # Grid 3: All cropped versions
    crop_images = []
    for i in range(len(image_files)):
        path = OUTPUT_DIR / f"img{i:02d}_4_crop50_gray.jpg"
        if path.exists():
            crop_images.append(cv2.imread(str(path)))
    create_comparison_grid(crop_images, OUTPUT_DIR / "grid_cropped.jpg", cols=5)

    print(f"\nDone! Generated {len(image_files) * 6} augmentation samples.")
    print(f"View results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
