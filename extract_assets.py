#!/usr/bin/env python3
"""Extract pad and copper trace asset masks from a real PCB photo using SAM 3.

Pipeline:
  1. Segment contact pads from the original image   → assets/PADS/
  2. Build a binary copper mask (pads blanked out, binarized via Otsu+KMeans)
  3. Segment copper traces from the binary mask     → assets/COPPER_TRACES/
  4. Filter pads  — remove border/margin-touching masks
  5. Split traces — split multi-component masks into individual instances

Usage:
  python extract_assets.py --image images/example.jpg
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).parent.resolve()


# ---------------------------------------------------------------------------
# SAM extraction
# ---------------------------------------------------------------------------

def extract_and_save_masked_objects(
    image_path: str,
    results,
    output_dir: str,
    *,
    exclude_pad_dir: str | None = None,
    pad_overlap_threshold: float = 0.5,
    dedup_iou_threshold: float | None = None,
) -> int:
    """Save individual mask + object crops from SAM results.

    Writes mask_XXXX.png (grayscale binary) and object_XXXX.png (BGRA with
    original pixel colours) for each detected instance, plus a combined layer.

    If exclude_pad_dir is given, masks that overlap pad masks by more than
    pad_overlap_threshold (fraction of mask area) are dropped. Among remaining
    masks, duplicates are removed keeping the largest (fullest) instance per
    overlapping group, using dedup_iou_threshold as the IoU cutoff.

    Returns the number of instances saved.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    original = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if original is None:
        raise ValueError(f"Cannot load image: {image_path}")

    bgr = original[:, :, :3] if original.shape[2] == 4 else original
    height, width = bgr.shape[:2]

    if results[0].masks is None:
        print("  No masks returned by SAM.")
        return 0

    masks_tensor = results[0].masks.data
    n = len(masks_tensor)
    print(f"  Found {n} raw masks")

    # Convert to bool arrays for fast set operations
    raw = [m.cpu().numpy().astype(bool) for m in masks_tensor]
    areas = [int(m.sum()) for m in raw]

    # --- Step 1: drop masks that are mostly pads ---
    combined_pad: np.ndarray | None = None
    if exclude_pad_dir is not None:
        combined_pad = np.zeros((height, width), dtype=bool)
        for f in sorted(Path(exclude_pad_dir).glob("mask_*.png")):
            pm = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if pm is not None:
                if pm.shape != (height, width):
                    pm = cv2.resize(pm, (width, height), interpolation=cv2.INTER_NEAREST)
                combined_pad |= (pm > 0)

    candidates: list[int] = []
    pad_removed = 0
    for i, mask in enumerate(raw):
        if combined_pad is not None and areas[i] > 0:
            overlap = int((mask & combined_pad).sum()) / areas[i]
            if overlap > pad_overlap_threshold:
                pad_removed += 1
                continue
        candidates.append(i)

    if pad_removed:
        print(f"  Removed {pad_removed} masks overlapping with pads (>{pad_overlap_threshold:.0%})")

    # --- Step 2: deduplicate — keep the largest mask per overlapping group ---
    if dedup_iou_threshold is not None:
        # Sort by area descending so the first accepted mask in each group is the biggest.
        candidates.sort(key=lambda i: areas[i], reverse=True)

        keep: list[int] = []
        for i in candidates:
            for j in keep:
                inter = int((raw[i] & raw[j]).sum())
                union = int((raw[i] | raw[j]).sum())
                if union > 0 and inter / union > dedup_iou_threshold:
                    break  # duplicate of an already-kept (larger) mask
            else:
                keep.append(i)

        dedup_removed = len(candidates) - len(keep)
        if dedup_removed:
            print(f"  Removed {dedup_removed} duplicate masks (IoU>{dedup_iou_threshold})")
    else:
        keep = candidates

    print(f"  Saving {len(keep)} unique masks")

    # --- Save ---
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    for save_idx, orig_idx in enumerate(keep):
        mask_np = raw[orig_idx].astype(np.uint8) * 255

        obj = np.zeros((height, width, 4), dtype=np.uint8)
        obj[:, :, :3] = bgr
        obj[:, :, 3] = mask_np

        cv2.imwrite(f"{output_dir}/mask_{save_idx:04d}.png", mask_np)
        cv2.imwrite(
            f"{output_dir}/object_{save_idx:04d}.png", obj,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )

        combined_mask = cv2.bitwise_or(combined_mask, mask_np)

    combined_obj = np.zeros((height, width, 4), dtype=np.uint8)
    combined_obj[:, :, :3] = bgr
    combined_obj[:, :, 3] = combined_mask
    cv2.imwrite(
        f"{output_dir}/all_objects_combined.png", combined_obj,
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )

    print(f"  Saved {len(keep)} instances → {output_dir}/")
    return len(keep)


# ---------------------------------------------------------------------------
# Binarization helper
# ---------------------------------------------------------------------------

def _binarize_without_pads(image_path: str, pads_dir: str) -> np.ndarray:
    """Return a binary dark-copper mask with pad regions blanked out.

    Applies combined Otsu + K-means (2-cluster) thresholding to the
    original image after masking out already-extracted pad pixels.
    Returns dark_mask (uint8, 0/255).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    height, width = img.shape[:2]

    combined_pad_mask = np.zeros((height, width), dtype=np.uint8)
    for f in sorted(Path(pads_dir).glob("mask_*.png")):
        m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            combined_pad_mask = cv2.bitwise_or(combined_pad_mask, m)

    # Blank out pad pixels with white so they don't skew the threshold
    masked_img = img.copy()
    masked_img[combined_pad_mask > 0] = 255

    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    non_pad_mask = (combined_pad_mask == 0).astype(np.uint8) * 255
    opaque_pixels = gray[non_pad_mask > 0]

    if len(opaque_pixels) == 0:
        raise RuntimeError("No non-pad pixels found — cannot binarize.")

    # Otsu on opaque pixels
    otsu_thresh, _ = cv2.threshold(
        opaque_pixels.reshape(1, -1), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # K-means (2 clusters) on opaque pixels
    pixels_f = opaque_pixels.astype(np.float32).reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(
        pixels_f, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS,
    )
    kmeans_thresh = float(np.mean(centers))

    final_thresh = int((float(otsu_thresh) + kmeans_thresh) / 2)
    print(
        f"  Binarization threshold: {final_thresh} "
        f"(Otsu={otsu_thresh:.0f}, KMeans={kmeans_thresh:.0f})"
    )

    _, binary = cv2.threshold(gray, final_thresh, 255, cv2.THRESH_BINARY)

    dark_mask = cv2.bitwise_not(binary)
    dark_mask = cv2.bitwise_and(dark_mask, non_pad_mask)
    return dark_mask


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_pads(
    folder: str,
    trash_folder: str = "trash",
    margin: int = 10,
    mask_prefix: str = "mask",
    object_prefix: str = "object",
) -> None:
    """Move invalid pad masks to trash: empty, border-touching, within margin."""
    os.makedirs(os.path.join(folder, trash_folder), exist_ok=True)

    mask_files = sorted(
        f for f in os.listdir(folder)
        if f.startswith(mask_prefix) and f.endswith(".png")
    )

    removed = 0
    for mask_name in mask_files:
        suffix = mask_name[len(mask_prefix):]
        object_name = object_prefix + suffix

        mask_path = os.path.join(folder, mask_name)
        object_path = os.path.join(folder, object_name)

        if not os.path.exists(object_path):
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        h, w = mask.shape
        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            invalid = True
        else:
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()
            invalid = (
                xmin == 0 or ymin == 0 or xmax == w - 1 or ymax == h - 1
                or xmin < margin or ymin < margin
                or xmax > w - margin - 1 or ymax > h - margin - 1
            )

        if invalid:
            shutil.move(mask_path, os.path.join(folder, trash_folder, mask_name))
            shutil.move(object_path, os.path.join(folder, trash_folder, object_name))
            removed += 1

    kept = len(mask_files) - removed
    print(f"  Pads: {kept} kept, {removed} moved to trash/")


def split_trace_instances(
    folder: str,
    trash_folder: str = "trash",
    min_area: int = 20,
    mask_prefix: str = "mask",
    object_prefix: str = "object",
) -> None:
    """Split multi-component trace masks; discard instances smaller than min_area."""
    os.makedirs(os.path.join(folder, trash_folder), exist_ok=True)

    mask_files = sorted(
        f for f in os.listdir(folder)
        if f.startswith(mask_prefix) and f.endswith(".png")
    )

    split_count = removed_count = 0
    for mask_name in mask_files:
        suffix = mask_name[len(mask_prefix):]
        object_name = object_prefix + suffix

        mask_path = os.path.join(folder, mask_name)
        object_path = os.path.join(folder, object_name)

        if not os.path.exists(object_path):
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        obj = cv2.imread(object_path, cv2.IMREAD_COLOR)
        if mask is None or obj is None:
            continue

        bin_mask = (mask > 0).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(bin_mask, connectivity=8)

        if num_labels <= 2:  # single component
            area = int((labels == 1).sum())
            if area < min_area:
                shutil.move(mask_path, os.path.join(folder, trash_folder, mask_name))
                shutil.move(object_path, os.path.join(folder, trash_folder, object_name))
                removed_count += 1
            continue

        _, ext = os.path.splitext(mask_name)
        comp_idx = 0
        for label in range(1, num_labels):
            comp_mask = (labels == label).astype(np.uint8) * 255
            if int(comp_mask.sum()) // 255 < min_area:
                continue

            new_suffix = suffix.replace(ext, f"_{comp_idx}{ext}")
            comp_obj = np.zeros_like(obj)
            comp_obj[comp_mask > 0] = obj[comp_mask > 0]

            cv2.imwrite(os.path.join(folder, f"{mask_prefix}{new_suffix}"), comp_mask)
            cv2.imwrite(os.path.join(folder, f"{object_prefix}{new_suffix}"), comp_obj)
            comp_idx += 1

        shutil.move(mask_path, os.path.join(folder, trash_folder, mask_name))
        shutil.move(object_path, os.path.join(folder, trash_folder, object_name))
        split_count += 1

    print(f"  Traces: {split_count} masks split, {removed_count} tiny instances removed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    root = str(_REPO_ROOT)
    parser = argparse.ArgumentParser(
        description="Segment pads and traces from a PCB image and save asset masks."
    )
    parser.add_argument(
        "--image", default=os.path.join(root, "images", "example.jpg"),
        help="Input PCB image (default: images/example.jpg)",
    )
    parser.add_argument(
        "--pads_dir", default=os.path.join(root, "assets", "PADS"),
        help="Output directory for pad assets (default: assets/PADS)",
    )
    parser.add_argument(
        "--traces_dir", default=os.path.join(root, "assets", "COPPER_TRACES"),
        help="Output directory for trace assets (default: assets/COPPER_TRACES)",
    )
    parser.add_argument(
        "--model", default=os.path.join(root, "models", "sam3.pt"),
        help="SAM 3 model path (default: models/sam3.pt)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="SAM confidence threshold")
    parser.add_argument("--pad_margin", type=int, default=10, help="Margin (px) for filter_pads")
    parser.add_argument("--min_trace_area", type=int, default=200,
                        help="Min pixel area for trace instance filtering")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from ultralytics.models.sam import SAM3SemanticPredictor  # noqa: E402 (heavy import)

    print("Initialising SAM3...")
    predictor = SAM3SemanticPredictor(overrides=dict(
        conf=args.conf,
        task="segment",
        mode="predict",
        model=args.model,
        half=False,
        save=False,
    ))

    # 1. Segment pads
    print("\n[1/5] Segmenting pads...")
    predictor.set_image(args.image)
    results_pads = predictor(text=["small silver element"])
    extract_and_save_masked_objects(args.image, results_pads, args.pads_dir)

    # 2. Build binary copper mask
    print("\n[2/5] Building binary copper mask...")
    dark_mask = _binarize_without_pads(args.image, args.pads_dir)
    tmp_mask_path = os.path.join(args.traces_dir, "_tmp_dark_mask.png")
    Path(args.traces_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(tmp_mask_path, dark_mask)

    # 3. Segment traces from binary mask
    print("\n[3/5] Segmenting copper traces...")
    predictor.set_image(tmp_mask_path)
    results_traces = predictor(text=[
        "black thick line",
        "black thin line",
        "black short line that connects black circles",
        "black short line that connects black elements",
        "black short line that connects black squares",
    ])
    extract_and_save_masked_objects(
        args.image, results_traces, args.traces_dir,
        exclude_pad_dir=args.pads_dir,
        dedup_iou_threshold=0.3,
    )
    os.remove(tmp_mask_path)

    # 4. Filter pads
    print("\n[4/5] Filtering pads...")
    filter_pads(args.pads_dir, margin=args.pad_margin)

    # 5. Split / filter traces
    print("\n[5/5] Splitting trace instances...")
    split_trace_instances(args.traces_dir, min_area=args.min_trace_area)

    pads_kept = sum(1 for _ in Path(args.pads_dir).glob("mask_*.png"))
    traces_kept = sum(1 for _ in Path(args.traces_dir).glob("mask_*.png"))
    print(f"\nDone.")
    print(f"  Pads:   {pads_kept} masks in {args.pads_dir}")
    print(f"  Traces: {traces_kept} masks in {args.traces_dir}")


if __name__ == "__main__":
    main()
