#!/usr/bin/env python3
"""Debug script — apply defects to a pre-generated image and visualize results.

Reads the outputs of generate.py and runs only the defect generation step,
so you can iterate on defect quality without re-running the diffusion model.

Outputs (written to --out_dir, defaults to same folder as --image):
  defected.png   — image with defects, no overlays
  annotated.png  — same image with colored bboxes and class name labels
  labels.txt     — YOLO-format annotations

Usage:
  python defects_generation/debug/add_defects.py \\
      --image debug_output/image.png \\
      --canvas debug_output/canvas.pkl

  python defects_generation/debug/add_defects.py \\
      --image debug_output/image.png \\
      --canvas debug_output/canvas.pkl \\
      --n_defects 6 --seed 123 --out_dir debug_output/run2
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "layout_generator"))

import cv2
import numpy as np
from PIL import Image

from defects_generation.orchestrator import add_defects, save_yolo_labels  # noqa: E402
from defects_generation.types import CLASS_NAMES  # noqa: E402


# One distinct BGR color per defect class
_CLASS_COLORS = {
    0: (0,   200, 255),   # mouse_bite    — yellow-orange
    1: (255, 100,   0),   # spur          — blue
    2: (0,    80, 255),   # missing_hole  — red
    3: (255,   0, 180),   # short         — magenta
    4: (0,   255, 100),   # open_circuit  — green
    5: (200,   0, 255),   # spurious_copper — purple
}


def _draw_annotations(image_bgr: np.ndarray, annotations: list) -> np.ndarray:
    vis = image_bgr.copy()
    h, w = vis.shape[:2]

    for ann in annotations:
        color = _CLASS_COLORS.get(ann.class_id, (255, 255, 255))
        label = CLASS_NAMES[ann.class_id]

        cx = int(ann.center_x * w)
        cy = int(ann.center_y * h)
        bw = int(ann.width * w)
        bh = int(ann.height * h)

        x1 = cx - bw // 2
        y1 = cy - bh // 2
        x2 = cx + bw // 2
        y2 = cy + bh // 2

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Background chip behind the label
        lx = x1
        ly = y1 - th - baseline - 2
        if ly < 0:
            ly = y2 + 2
        cv2.rectangle(vis, (lx, ly), (lx + tw + 4, ly + th + baseline + 2), color, -1)
        cv2.putText(
            vis, label,
            (lx + 2, ly + th),
            font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA,
        )

    return vis


def _parse_args() -> argparse.Namespace:
    valid_types = list(CLASS_NAMES.values())
    p = argparse.ArgumentParser(description="Add defects to a pre-generated PCB image.")
    p.add_argument("--image",   type=str, required=True, help="Path to image.png from generate.py")
    p.add_argument("--canvas",  type=str, required=True, help="Path to canvas.pkl from generate.py")
    p.add_argument("--out_dir", type=str, default=None,  help="Output folder (default: same as --image)")
    p.add_argument("--n_defects", type=int, default=None, help="Number of defects (default: random 1-4)")
    p.add_argument("--defect_type", type=str, default=None,
                   choices=valid_types, metavar="TYPE",
                   help=f"Force a single defect class. One of: {', '.join(valid_types)}. Default: random.")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    image_path = Path(args.image)
    canvas_path = Path(args.canvas)

    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}")
        sys.exit(1)
    if not canvas_path.exists():
        print(f"ERROR: canvas not found: {canvas_path}")
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else image_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"ERROR: could not read image: {image_path}")
        sys.exit(1)
    print(f"Loaded image {image_bgr.shape[1]}×{image_bgr.shape[0]} from {image_path}")

    # Load canvas
    sys.path.insert(0, str(_ROOT / "layout_generator"))
    with open(canvas_path, "rb") as f:
        canvas = pickle.load(f)
    print(f"Loaded canvas: {len(canvas.pad_instances)} pads, {len(canvas.trace_instances)} traces")

    # Determine defect count and type
    rng = np.random.default_rng(args.seed)
    n_defects = args.n_defects if args.n_defects is not None else int(rng.integers(1, 5))

    defect_weights = None
    if args.defect_type is not None:
        class_id = next(k for k, v in CLASS_NAMES.items() if v == args.defect_type)
        defect_weights = {i: (1.0 if i == class_id else 0.0) for i in range(6)}

    type_label = args.defect_type or "random"
    print(f"Adding {n_defects}× {type_label} (seed={args.seed})...")
    defected_bgr, annotations = add_defects(
        image=image_bgr,
        canvas=canvas,
        n_defects=n_defects,
        defect_weights=defect_weights,
        seed=args.seed,
    )

    placed = len(annotations)
    print(f"  {placed} defect(s) placed:")
    for ann in annotations:
        print(f"    [{ann.class_id}] {CLASS_NAMES[ann.class_id]:20s}  "
              f"cx={ann.center_x:.3f} cy={ann.center_y:.3f}  "
              f"w={ann.width:.3f} h={ann.height:.3f}")

    # Save defected image
    defected_path = out_dir / "defected.png"
    cv2.imwrite(str(defected_path), defected_bgr)

    # Save annotated visualization
    annotated_bgr = _draw_annotations(defected_bgr, annotations)
    annotated_path = out_dir / "annotated.png"
    cv2.imwrite(str(annotated_path), annotated_bgr)

    # Save YOLO labels
    labels_path = out_dir / "labels.txt"
    save_yolo_labels(annotations, labels_path)

    print(f"\nOutputs saved to: {out_dir}")
    print(f"  defected.png   — image with defects")
    print(f"  annotated.png  — with colored bboxes and class labels")
    print(f"  labels.txt     — YOLO annotations")


if __name__ == "__main__":
    main()
