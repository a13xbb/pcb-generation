#!/usr/bin/env python3
"""Debug canvas refinement EXCEPT defect regions.

Algorithm:
1. Baseline binarization (from augmentations.py)
2. Canvas refinement EXCEPT defect bboxes (defects keep baseline)

Usage:
    python yolo_training/debug_canvas_except_defects.py
"""

import gzip
import pickle
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "layout_generator"))

from augmentations import binarize_synth

ROOT = Path(__file__).parent.parent
DEFAULT_DATA = ROOT / "datasets/generated_dataset/v2/defects/data.yaml"
OUTPUT_DIR = ROOT / "debug_output/canvas_except_defects"
REFERENCE_DIR = ROOT / "debug_output/grayscale_aug_examples"

CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "open_circuit",
    4: "spurious_copper",
}

COLORS = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (255, 0, 255),
}


def extract_layout_id(img_path: str) -> int:
    match = re.search(r'layout_(\d+)(?:_rep|\.)', str(img_path))
    return int(match.group(1)) if match else -1


def load_yolo_labels(label_path: Path):
    labels = []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels.append([float(x) for x in parts[:5]])
    return labels


def draw_bboxes(img: np.ndarray, labels: list) -> np.ndarray:
    img = img.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]

    for label in labels:
        cls_id = int(label[0])
        cx, cy, bw, bh = label[1:5]

        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)

        color = COLORS.get(cls_id, (128, 128, 128))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label_text = CLASS_NAMES.get(cls_id, str(cls_id))
        cv2.putText(img, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return img


def load_canvas_mask(layout_id: int, target_size: tuple, dataset_root: Path) -> np.ndarray:
    canvas_path = dataset_root / "canvases" / f"layout_{layout_id}.pkl.gz"
    if not canvas_path.exists():
        return None

    with gzip.open(canvas_path, 'rb') as f:
        canvas = pickle.load(f)

    mask = (canvas.occupied_mask > 0).astype(np.uint8) * 255
    return cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)


def canvas_except_defects_binarize(
    img: np.ndarray,
    img_path: str,
    dataset_root: Path,
    defect_bbox_scale: float = 1.5
) -> np.ndarray:
    """
    1. Baseline binarization
    2. Canvas refinement EXCEPT defect bboxes

    Returns: grayscale binary (copper=0, bg=255)
    """
    h, w = img.shape[:2]

    # Step 1: Baseline binarization
    baseline = binarize_synth(img)  # BGR
    baseline_gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)

    # Step 2: Load canvas
    layout_id = extract_layout_id(img_path)
    canvas_copper = load_canvas_mask(layout_id, (w, h), dataset_root)

    if canvas_copper is None:
        return baseline_gray

    # Create mask of defect regions (to EXCLUDE from canvas refinement)
    defect_mask = np.zeros((h, w), dtype=np.uint8)

    label_path = Path(str(img_path).replace("/images/", "/labels/").replace(".png", ".txt"))
    defect_bboxes = load_yolo_labels(label_path)

    for label in defect_bboxes:
        cx, cy, bw, bh = label[1:5]

        # Extend bbox slightly
        bw_ext = bw * defect_bbox_scale
        bh_ext = bh * defect_bbox_scale

        x1 = max(0, int((cx - bw_ext / 2) * w))
        y1 = max(0, int((cy - bh_ext / 2) * h))
        x2 = min(w, int((cx + bw_ext / 2) * w))
        y2 = min(h, int((cy + bh_ext / 2) * h))

        defect_mask[y1:y2, x1:x2] = 255

    # Canvas refinement: fill missed copper EXCEPT in defect regions
    # canvas_copper: copper=255, bg=0
    # baseline_gray: copper=0, bg=255

    result = baseline_gray.copy()

    # Where canvas says copper (255) AND baseline missed it (255) AND NOT in defect region
    missed_copper = (canvas_copper == 255) & (baseline_gray == 255) & (defect_mask == 0)
    result[missed_copper] = 0  # Set to copper (black)

    return result


def get_reference_images():
    images = []
    for f in sorted(REFERENCE_DIR.glob("*_0_original.jpg")):
        parts = f.stem.split("_")
        if len(parts) >= 4:
            img_name = f"{parts[1]}_{parts[2]}_{parts[3]}"
            images.append((f.stem.split("_")[0], img_name))
    return images


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DEFAULT_DATA) as f:
        data_config = yaml.safe_load(f)

    src_path = Path(data_config["path"])
    dataset_root = src_path.parent
    images_dir = src_path / "images"
    labels_dir = src_path / "labels"

    reference_images = get_reference_images()

    print(f"Processing {len(reference_images)} images")
    print(f"Output: {OUTPUT_DIR}")
    print()

    for idx_str, img_name in reference_images:
        img_path = images_dir / f"{img_name}.png"
        if not img_path.exists():
            print(f"  Skipping {img_name} - not found")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        labels = load_yolo_labels(labels_dir / f"{img_name}.txt")
        base_name = f"{idx_str}_{img_name}"

        # Baseline
        baseline = binarize_synth(img)
        baseline_gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)

        # Canvas
        layout_id = extract_layout_id(img_name)
        canvas_mask = load_canvas_mask(layout_id, (img.shape[1], img.shape[0]), dataset_root)

        # Final: canvas except defects
        final = canvas_except_defects_binarize(img, str(img_path), dataset_root)

        # Save images
        orig_with_boxes = draw_bboxes(img, labels)
        cv2.imwrite(str(OUTPUT_DIR / f"{base_name}_0_original.jpg"), orig_with_boxes)

        baseline_with_boxes = draw_bboxes(baseline, labels)
        cv2.imwrite(str(OUTPUT_DIR / f"{base_name}_1_baseline.jpg"), baseline_with_boxes)

        if canvas_mask is not None:
            canvas_vis = cv2.bitwise_not(canvas_mask)
            canvas_bgr = cv2.cvtColor(canvas_vis, cv2.COLOR_GRAY2BGR)
            canvas_with_boxes = draw_bboxes(canvas_bgr, labels)
            cv2.imwrite(str(OUTPUT_DIR / f"{base_name}_2_canvas.jpg"), canvas_with_boxes)

        final_bgr = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
        final_with_boxes = draw_bboxes(final_bgr, labels)
        cv2.imwrite(str(OUTPUT_DIR / f"{base_name}_3_final.jpg"), final_with_boxes)

        # Comparison grid
        h, w = img.shape[:2]
        row1 = np.hstack([orig_with_boxes, baseline_with_boxes])
        row2 = np.hstack([
            canvas_with_boxes if canvas_mask is not None else np.ones_like(img) * 128,
            final_with_boxes
        ])
        comparison = np.vstack([row1, row2])

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 25), font, 0.6, (0, 255, 0), 2)
        cv2.putText(comparison, "Baseline", (w + 10, 25), font, 0.6, (0, 255, 0), 2)
        cv2.putText(comparison, "Canvas", (10, h + 25), font, 0.6, (0, 255, 0), 2)
        cv2.putText(comparison, "Final (canvas except defects)", (w + 10, h + 25), font, 0.6, (0, 255, 0), 2)

        cv2.imwrite(str(OUTPUT_DIR / f"{base_name}_comparison.jpg"), comparison)

        print(f"  [{idx_str}] {img_name}")

    print(f"\nDone! Results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
