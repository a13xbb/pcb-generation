#!/usr/bin/env python3
"""Stage 3: add synthetic defects to pre-generated PCB images.

Reads from --input_dir:
  images/layout_N.png   — final PCB image (from stage 2)
  canvases/layout_N.pkl — canvas for defect placement (from stage 1)

Writes to --output_dir:
  defects/images/layout_N_rep{R}.png  — image with defects
  defects/labels/layout_N_rep{R}.txt  — YOLO-format annotations

The --repeats_per_image flag controls how many defect variations are produced
per source image, each with a different seed derived from the base seed.

Usage:
  python pipeline_stage3_defects.py \\
      --input_dir images/run1 --output_dir images/run1 --repeats_per_image 5
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "layout_generator"))  # required for unpickling CanvasState

import cv2
import numpy as np
from tqdm import tqdm

from defects_generation.orchestrator import add_defects, save_yolo_labels  # noqa: E402


def _sorted_layout_paths(folder: Path) -> list[tuple[int, Path]]:
    results = []
    for p in folder.iterdir():
        m = re.fullmatch(r"layout_(\d+)\.png", p.name)
        if m:
            results.append((int(m.group(1)), p))
    return sorted(results)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 3: add synthetic defects to PCB images.")
    p.add_argument("--input_dir", type=str, default="images/pipeline")
    p.add_argument("--output_dir", type=str, default="images/pipeline")
    p.add_argument("--repeats_per_image", type=int, default=1,
                   help="Number of defect variations per source image (different seeds)")
    p.add_argument("--min_defects", type=int, default=4)
    p.add_argument("--max_defects", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    in_dir = _ROOT / args.input_dir
    out_images = _ROOT / args.output_dir / "defects" / "images"
    out_labels = _ROOT / args.output_dir / "defects" / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_entries = _sorted_layout_paths(in_dir / "images")
    if not image_entries:
        print(f"No layout_*.png files found in {in_dir / 'images'}")
        return

    skipped = 0
    for i, image_path in tqdm(image_entries, desc="Defects"):
        canvas_path = in_dir / "canvases" / f"layout_{i}.pkl"
        if not canvas_path.exists():
            print(f"  [{i}] WARNING: canvas not found at {canvas_path}, skipping")
            skipped += 1
            continue

        try:
            with open(canvas_path, "rb") as f:
                canvas = pickle.load(f)
        except Exception as e:
            print(f"  [{i}] WARNING: failed to load canvas ({e}), skipping")
            skipped += 1
            continue

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"  [{i}] WARNING: could not read image {image_path}, skipping")
            skipped += 1
            continue

        for rep in range(args.repeats_per_image):
            defect_seed = args.seed + i * 10000 + rep
            n_defects = int(
                np.random.default_rng(defect_seed).integers(args.min_defects, args.max_defects + 1)
            )
            defected_bgr, annotations = add_defects(
                image=image_bgr.copy(),
                canvas=canvas,
                n_defects=n_defects,
                seed=defect_seed,
            )
            stem = f"layout_{i}_rep{rep}"
            cv2.imwrite(str(out_images / f"{stem}.png"), defected_bgr)
            save_yolo_labels(annotations, out_labels / f"{stem}.txt")

    total = len(image_entries)
    total_out = (total - skipped) * args.repeats_per_image
    print(f"\nDone. Processed {total - skipped}/{total} images → {total_out} defected images.")
    print(f"  Images: {out_images}")
    print(f"  Labels: {out_labels}")


if __name__ == "__main__":
    main()
