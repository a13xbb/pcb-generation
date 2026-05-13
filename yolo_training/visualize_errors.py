#!/usr/bin/env python3
"""Visualize YOLO predictions vs ground truth using FiftyOne.

Loads test images with ground truth labels and model predictions.
Computes evaluation metrics (TP/FP/FN) with per-class breakdown.

Usage:
  # Analyze synthetic model (default: conf=0.5, iou=0.5)
  python yolo_training/visualize_errors.py --model yolo_training/results/synth_train/weights/best.pt --name synth_analysis

  # Analyze real model with custom thresholds
  python yolo_training/visualize_errors.py --model yolo_training/results/real_train/weights/best.pt --name real_analysis --conf 0.25 --iou 0.5

Filtering in FiftyOne UI:
  - By evaluation: F("eval") == "fn" (false negatives / missed)
  - By class (GT): F("ground_truth.detections.label") == "mouse_bite"
  - By class (pred): F("predictions.detections.label") == "spur"
  - By confidence: F("predictions.detections.confidence") > 0.5
"""
import argparse
from pathlib import Path

import cv2
import fiftyone as fo
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
_DEFAULT_IMAGES = ROOT / "pcb-defect-dataset/test/images"
_DEFAULT_LABELS = ROOT / "pcb-defect-dataset/test/labels"

CLASS_NAMES = ["mouse_bite", "spur", "missing_hole", "open_circuit", "spurious_copper"]


def yolo_to_fo_detections(results, class_names):
    """Convert YOLO results to FiftyOne Detections."""
    detections = []
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return fo.Detections(detections=[])

    img_h, img_w = results.orig_shape

    for i in range(len(boxes)):
        box = boxes.xyxy[i].cpu().numpy()
        conf = float(boxes.conf[i].cpu().numpy())
        cls_id = int(boxes.cls[i].cpu().numpy())

        x1, y1, x2, y2 = box
        rel_x = x1 / img_w
        rel_y = y1 / img_h
        rel_w = (x2 - x1) / img_w
        rel_h = (y2 - y1) / img_h

        label = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"

        detections.append(fo.Detection(
            label=label,
            bounding_box=[rel_x, rel_y, rel_w, rel_h],
            confidence=conf,
        ))

    return fo.Detections(detections=detections)


def load_yolo_labels(label_path, class_names):
    """Load YOLO format labels as FiftyOne Detections."""
    detections = []
    if not label_path.exists():
        return fo.Detections(detections=[])

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            x = cx - w / 2
            y = cy - h / 2
            label = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            detections.append(fo.Detection(
                label=label,
                bounding_box=[x, y, w, h],
            ))

    return fo.Detections(detections=detections)


def _save_confusion_matrix(dataset, class_names, out_path, iou, conf):
    """Build and save a row-normalised confusion matrix from eval-tagged detections."""
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(class_names)
    # rows = GT class, cols = predicted class; last col = FN (background), last row = FP (background)
    labels = class_names + ["background"]
    matrix = np.zeros((n + 1, n + 1), dtype=int)

    for ci, cls in enumerate(class_names):
        view = dataset.filter_labels("predictions", fo.ViewField("label") == cls)
        for other_ci, other_cls in enumerate(class_names):
            tp = (
                view
                .filter_labels("predictions", fo.ViewField("eval") == "tp")
                .filter_labels("ground_truth", fo.ViewField("label") == other_cls)
                .count("predictions.detections")
            )
            matrix[other_ci, ci] += tp
        fp = view.filter_labels("predictions", fo.ViewField("eval") == "fp").count("predictions.detections")
        matrix[n, ci] += fp

    for ci, cls in enumerate(class_names):
        fn = (
            dataset
            .filter_labels("ground_truth", fo.ViewField("label") == cls)
            .filter_labels("ground_truth", fo.ViewField("eval") == "fn")
            .count("ground_truth.detections")
        )
        matrix[ci, n] += fn

    # Normalise each row by its sum so values are fractions 0..1
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm = np.where(row_sums > 0, matrix / row_sums.astype(float), 0.0)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Fraction of row total")

    ax.set_xticks(range(n + 1))
    ax.set_yticks(range(n + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"Confusion Matrix  |  conf={conf}  IoU={iou}")

    for i in range(n + 1):
        for j in range(n + 1):
            frac = norm[i, j]
            if frac > 0:
                color = "white" if frac > 0.5 else "black"
                ax.text(j, i, f"{frac:.2f}", ha="center", va="center", fontsize=8, color=color)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Visualize YOLO errors with FiftyOne")
    p.add_argument("--model", type=str, required=True, help="Path to YOLO model weights")
    p.add_argument("--images", type=str, default=str(_DEFAULT_IMAGES), help="Path to images folder")
    p.add_argument("--labels", type=str, default=str(_DEFAULT_LABELS), help="Path to YOLO labels folder")
    p.add_argument("--split-file", type=str, default=None, help="Text file listing image paths (e.g. val.txt). Overrides --images.")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for evaluation")
    p.add_argument("--port", type=int, default=5150, help="FiftyOne app port")
    p.add_argument("--name", type=str, default="pcb_analysis", help="Dataset name")
    p.add_argument("--delete", action="store_true", help="Delete existing dataset and recreate")
    return p.parse_args()


def main():
    args = parse_args()

    if args.delete and args.name in fo.list_datasets():
        print(f"Deleting existing dataset '{args.name}'...")
        fo.delete_dataset(args.name)

    if args.name in fo.list_datasets():
        print(f"Loading existing dataset '{args.name}'...")
        dataset = fo.load_dataset(args.name)
    else:
        print("Creating new dataset...")
        dataset = fo.Dataset(args.name, persistent=True)

        labels_dir = Path(args.labels)
        if args.split_file:
            with open(args.split_file) as f:
                image_paths = [Path(line.strip()) for line in f if line.strip()]
            print(f"Found {len(image_paths)} images from split file {args.split_file}")
        else:
            images_dir = Path(args.images)
            image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
            print(f"Found {len(image_paths)} images in {images_dir}")

        print(f"Loading model: {args.model}")
        model = YOLO(args.model)

        samples = []
        for i, img_path in enumerate(image_paths):
            if (i + 1) % 100 == 0:
                print(f"  Processing {i + 1}/{len(image_paths)}...")

            sample = fo.Sample(filepath=str(img_path))

            # Ground truth
            label_path = labels_dir / f"{img_path.stem}.txt"
            sample["ground_truth"] = load_yolo_labels(label_path, CLASS_NAMES)

            # Predictions
            results = model(str(img_path), conf=args.conf, verbose=False)[0]
            sample["predictions"] = yolo_to_fo_detections(results, CLASS_NAMES)

            samples.append(sample)

        dataset.add_samples(samples)
        print(f"Added {len(samples)} samples")

        print(f"Evaluating predictions (IoU={args.iou})...")
        results = dataset.evaluate_detections(
            "predictions",
            gt_field="ground_truth",
            eval_key="eval",
            iou=args.iou,
        )
        dataset.info["eval_results"] = True
        dataset.info["conf"] = args.conf
        dataset.info["iou"] = args.iou
        dataset.save()

    # Compute and print metrics
    print("\n" + "=" * 60)
    print(f"EVALUATION METRICS (conf={args.conf}, IoU={args.iou})")
    print("=" * 60)

    # Re-run evaluation to get results object (needed if loading existing dataset)
    results = dataset.evaluate_detections(
        "predictions",
        gt_field="ground_truth",
        eval_key="eval",
        iou=args.iou,
    )

    # Print the full report
    print("\n--- Classification Report ---")
    results.print_report(classes=CLASS_NAMES)

    # Confusion matrix
    cm_path = Path(args.name + "_confusion_matrix.png")
    _save_confusion_matrix(dataset, CLASS_NAMES, cm_path, iou=args.iou, conf=args.conf)
    print(f"\nConfusion matrix saved to: {cm_path.resolve()}")

    # Manual per-class TP/FP/FN counts
    print("\n--- Per-Class Breakdown ---")
    print(f"{'Class':<20} {'TP':>6} {'FP':>6} {'FN':>6} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 74)

    for cls in CLASS_NAMES:
        # Filter to this class
        gt_view = dataset.filter_labels("ground_truth", fo.ViewField("label") == cls)
        pred_view = dataset.filter_labels("predictions", fo.ViewField("label") == cls)

        # Count TP, FP, FN using eval field
        tp = pred_view.filter_labels("predictions", fo.ViewField("eval") == "tp").count("predictions.detections")
        fp = pred_view.filter_labels("predictions", fo.ViewField("eval") == "fp").count("predictions.detections")
        fn = gt_view.filter_labels("ground_truth", fo.ViewField("eval") == "fn").count("ground_truth.detections")

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"{cls:<20} {tp:>6} {fp:>6} {fn:>6} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f}")

    # Totals
    total_tp = dataset.filter_labels("predictions", fo.ViewField("eval") == "tp").count("predictions.detections")
    total_fp = dataset.filter_labels("predictions", fo.ViewField("eval") == "fp").count("predictions.detections")
    total_fn = dataset.filter_labels("ground_truth", fo.ViewField("eval") == "fn").count("ground_truth.detections")

    total_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    total_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    total_f1 = 2 * total_p * total_r / (total_p + total_r) if (total_p + total_r) > 0 else 0.0

    print("-" * 74)
    print(f"{'TOTAL':<20} {total_tp:>6} {total_fp:>6} {total_fn:>6} {total_p:>10.3f} {total_r:>10.3f} {total_f1:>10.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total images: {len(dataset)}")

    gt_count = dataset.count("ground_truth.detections")
    pred_count = dataset.count("predictions.detections")
    print(f"\nGround truth total: {gt_count}")
    print(f"Predictions total: {pred_count}")

    # Per-class breakdown
    print("\n--- Ground Truth by Class ---")
    for cls in CLASS_NAMES:
        count = dataset.filter_labels("ground_truth", fo.ViewField("label") == cls).count("ground_truth.detections")
        print(f"  {cls:<20} {count}")

    print("\n--- Predictions by Class ---")
    for cls in CLASS_NAMES:
        count = dataset.filter_labels("predictions", fo.ViewField("label") == cls).count("predictions.detections")
        print(f"  {cls:<20} {count}")

    # Launch app
    print(f"\n" + "=" * 60)
    print("FIFTYONE UI")
    print("=" * 60)
    print(f"Open http://localhost:{args.port} in your browser")
    print("\nFiltering examples (paste in filter bar):")
    print("  False negatives (missed GT):     F('eval') == 'fn'")
    print("  False positives (wrong pred):    F('eval') == 'fp'")
    print("  True positives:                  F('eval') == 'tp'")
    print("  GT class filter:                 F('ground_truth.detections.label') == 'mouse_bite'")
    print("  Pred class filter:               F('predictions.detections.label') == 'spur'")
    print("  High confidence only:            F('predictions.detections.confidence') > 0.7")
    print("\nPress Ctrl+C to stop")

    session = fo.launch_app(dataset, port=args.port)
    session.wait()


if __name__ == "__main__":
    main()
