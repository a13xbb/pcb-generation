#!/usr/bin/env python3
"""Stage 1: generate PCB layouts, structure maps, and canvas pickles.

Outputs (written to --output_dir):
  layouts/layout_N.png        — procedural layout visualization
  canvases/layout_N.pkl       — pickled CanvasState (pad/trace locations)
  structure_maps/layout_N.png — Canny edge map for ControlNet conditioning

No model loading; this stage runs on CPU only.

Usage:
  python pipeline_stage1_layouts.py --n_layouts 10 --output_dir images/run1
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "layout_generator"))
sys.path.insert(0, str(_ROOT / "structure_generation"))

import cv2
import numpy as np
from tqdm import tqdm

from classes import CanvasState, PadAsset, TraceAsset  # noqa: E402
from generator import canvas_coverage, generate_layout_motif_based  # noqa: E402
from image_to_structure import image_to_structure  # noqa: E402
from placement_engine import visualize_canvas_real  # noqa: E402
from utils import (  # noqa: E402
    compute_bbox_and_centroid,
    find_endpoints,
    skeleton_path_length_between_endpoints,
    skeletonize,
)


def _load_pad_assets(folder: Path) -> list[PadAsset]:
    assets = []
    for name in os.listdir(folder):
        if not (name.startswith("mask") and name.endswith(".png")):
            continue
        mask = cv2.imread(str(folder / name), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = (mask > 0).astype(np.uint8)
        obj = cv2.imread(str(folder / name.replace("mask", "object", 1)))
        if obj is None:
            obj = np.full((*mask.shape, 3), 160, dtype=np.uint8)
        bbox, centroid = compute_bbox_and_centroid(mask)
        assets.append(PadAsset(mask, obj, centroid, bbox))
    return assets


def _load_trace_assets(folder: Path) -> list[TraceAsset]:
    assets = []
    for name in sorted(os.listdir(folder)):
        if not (name.startswith("mask") and name.endswith(".png")):
            continue
        mask = cv2.imread(str(folder / name), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = (mask > 0).astype(np.uint8)
        skel = (skeletonize(mask) > 0).astype(np.uint8)
        endpoints = find_endpoints(mask, skel)
        if len(endpoints) != 2:
            continue
        length = skeleton_path_length_between_endpoints(skel, endpoints[0], endpoints[1]) or 0.0
        bbox, centroid = compute_bbox_and_centroid(mask)
        obj = cv2.imread(str(folder / name.replace("mask", "object", 1)))
        if obj is None:
            obj = np.full((*mask.shape, 3), 160, dtype=np.uint8)
        assets.append(TraceAsset(mask, obj, skel, endpoints, centroid, length))
    return assets


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1: generate layouts, structure maps, and canvases.")
    p.add_argument("--n_layouts", type=int, default=10)
    p.add_argument("--output_dir", type=str, default="images/pipeline")
    p.add_argument("--height", type=int, default=600)
    p.add_argument("--width", type=int, default=600)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    root_dir = _ROOT / args.output_dir
    out_layouts = root_dir / "layouts"
    out_canvases = root_dir / "canvases"
    out_structures = root_dir / "structure_maps"
    for d in (out_layouts, out_canvases, out_structures):
        d.mkdir(parents=True, exist_ok=True)

    print("Loading pad and trace assets...")
    assets_root = _ROOT / "assets"
    pad_assets = _load_pad_assets(assets_root / "PADS")
    trace_assets = _load_trace_assets(assets_root / "COPPER_TRACES")
    assert pad_assets and trace_assets, "No assets found in assets/PADS or assets/COPPER_TRACES"
    print(f"  {len(pad_assets)} pads, {len(trace_assets)} traces")

    for i in tqdm(range(args.n_layouts), desc="Layouts"):
        canvas = CanvasState(600, 600, pad_keepout_radius=8)
        ok = generate_layout_motif_based(
            canvas=canvas,
            pad_assets=pad_assets,
            trace_assets=trace_assets,
            n_cols=5,
            n_rows=4,
            edge_padding=10,
            max_motif_attempts=8,
        )

        layout_path = out_layouts / f"layout_{i}.png"
        visualize_canvas_real(canvas, str(layout_path))
        print(f"  [{i}] layout ok={ok}  coverage={canvas_coverage(canvas):.3f}"
              f"  pads={len(canvas.pad_instances)}  traces={len(canvas.trace_instances)}")

        with open(out_canvases / f"layout_{i}.pkl", "wb") as f:
            pickle.dump(canvas, f)

        layout_bgr = cv2.imread(str(layout_path))
        structure_np = image_to_structure(layout_bgr, return_rgb=False)
        cv2.imwrite(str(out_structures / f"layout_{i}.png"), structure_np)

    print(f"\nDone. {args.n_layouts} layouts saved to: {root_dir}")


if __name__ == "__main__":
    main()
