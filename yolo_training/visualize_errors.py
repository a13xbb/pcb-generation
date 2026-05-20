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
import numpy as np
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


def compute_iou_boxes(box1, box2):
    """Compute IoU between two boxes in [x, y, w, h] relative format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter = max(0, xb - xa) * max(0, yb - ya)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def compute_ap(precisions, recalls):
    """Compute Average Precision using 101-point interpolation (COCO style)."""
    recalls = np.array(recalls)
    precisions = np.array(precisions)

    indices = np.argsort(recalls)
    recalls = recalls[indices]
    precisions = precisions[indices]

    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[0], precisions, [0]])

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    recall_levels = np.linspace(0, 1, 101)
    precision_at_recalls = np.zeros(101)
    for i, r in enumerate(recall_levels):
        idx = np.where(recalls >= r)[0]
        if len(idx) > 0:
            precision_at_recalls[i] = precisions[idx[0]]

    return np.mean(precision_at_recalls)


def compute_class_ap_from_dataset(dataset, cls_name, iou_thresh):
    """Compute AP for a single class at a given IoU threshold from FiftyOne dataset."""
    all_detections = []
    total_gt = 0

    for sample in dataset:
        gt_dets = sample["ground_truth"].detections if sample["ground_truth"] else []
        pred_dets = sample["predictions"].detections if sample["predictions"] else []

        class_gts = [d for d in gt_dets if d.label == cls_name]
        total_gt += len(class_gts)

        for pred in pred_dets:
            if pred.label == cls_name:
                all_detections.append((sample.id, pred.bounding_box, pred.confidence, class_gts))

    if total_gt == 0:
        return 0.0

    all_detections.sort(key=lambda x: -x[2])

    gt_matched = {}
    tps = []
    fps = []

    for sample_id, pred_box, conf, class_gts in all_detections:
        if sample_id not in gt_matched:
            gt_matched[sample_id] = set()

        matched = False
        best_iou = 0
        best_gt_idx = -1

        for gi, gt_det in enumerate(class_gts):
            if gi in gt_matched[sample_id]:
                continue
            iou = compute_iou_boxes(pred_box, gt_det.bounding_box)
            if iou >= iou_thresh and iou > best_iou:
                best_iou = iou
                best_gt_idx = gi
                matched = True

        if matched:
            gt_matched[sample_id].add(best_gt_idx)
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)

    tps = np.cumsum(tps)
    fps = np.cumsum(fps)

    if len(tps) == 0:
        return 0.0

    recalls = tps / total_gt
    precisions = tps / (tps + fps)

    return compute_ap(precisions, recalls)


def compute_and_print_map(dataset, class_names, iou_thresholds=[0.25, 0.50, 0.75]):
    """Compute and print mAP at multiple IoU thresholds."""
    print(f"\n── Per-class AP at Different IoU Thresholds ──────────────────────────")
    header = f"{'Class':<20}"
    for iou in iou_thresholds:
        header += f" {'AP@' + str(int(iou*100)):>10}"
    print(header)
    print("-" * (20 + 11 * len(iou_thresholds)))

    all_aps = {iou: [] for iou in iou_thresholds}

    for cls_name in class_names:
        row = f"  {cls_name:<20}"
        for iou in iou_thresholds:
            ap = compute_class_ap_from_dataset(dataset, cls_name, iou)
            all_aps[iou].append(ap)
            row += f" {ap:>9.4f}"
        print(row)

    print("-" * (20 + 11 * len(iou_thresholds)))
    row = f"  {'mAP':<20}"
    for iou in iou_thresholds:
        map_val = np.mean(all_aps[iou]) if all_aps[iou] else 0
        row += f" {map_val:>9.4f}"
    print(row)

    print(f"\n── mAP Summary ─────────────────────────────────")
    for iou in iou_thresholds:
        map_val = np.mean(all_aps[iou]) if all_aps[iou] else 0
        print(f"  mAP@{int(iou*100):<6} {map_val:.4f}")

    return all_aps


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
    p.add_argument("--grayscale", action="store_true", help="Convert images to grayscale before inference (for models trained on grayscale)")
    p.add_argument("--no_launch", action="store_true", help="Skip FiftyOne UI, just print metrics and save confusion matrix")
    p.add_argument("--compute_map", action="store_true", help="Compute mAP at multiple IoU thresholds (0.25, 0.50, 0.75)")
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
            if args.grayscale:
                bgr = cv2.imread(str(img_path))
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                img_input = np.stack([gray, gray, gray], axis=-1)
            else:
                img_input = str(img_path)
            results = model(img_input, conf=args.conf, verbose=False)[0]
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

    # Compute mAP at multiple IoU thresholds if requested
    if args.compute_map:
        compute_and_print_map(dataset, CLASS_NAMES, iou_thresholds=[0.25, 0.50, 0.75])

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

    if args.no_launch:
        print(f"\n(Skipping FiftyOne UI launch due to --no_launch flag)")
    else:
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
