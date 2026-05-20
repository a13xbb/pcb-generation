import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.data.build import build_dataloader
from ultralytics.data.dataset import YOLODataset

ROOT = Path(__file__).parent.parent

CLASS_NAMES = ["mouse_bite", "spur", "missing_hole", "open_circuit", "spurious_copper"]


class GrayscaleYOLODataset(YOLODataset):
    """YOLO dataset that converts images to grayscale on load."""

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if "img" in sample and sample["img"] is not None:
            img = sample["img"]
            if img.ndim == 3 and img.shape[0] == 3:
                gray = cv2.cvtColor(img.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
                sample["img"] = np.stack([gray, gray, gray], axis=0)
        return sample


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a YOLOv8 model on PCB test sets")
    p.add_argument("weights", help="Path to model weights (.pt)")
    p.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--grayscale", action="store_true", help="Convert images to grayscale")
    p.add_argument("--imgsz",  type=int,   default=640)
    p.add_argument("--batch",  type=int,   default=16)
    p.add_argument("--conf",   type=float, default=0.25)
    p.add_argument("--iou",    type=float, default=0.5)
    p.add_argument("--workers",type=int,   default=0)
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    if args.grayscale:
        print(f"Note: --grayscale requires preprocessing. Using custom validation loop.")
        run_grayscale_eval(model, args)
    else:
        metrics = model.val(
            data=args.data,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.conf,
            iou=args.iou,
            workers=args.workers,
            plots=True,
            save_json=True,
        )
        print_metrics(metrics)


def run_grayscale_eval(model, args):
    """Run evaluation with grayscale preprocessing."""
    import yaml
    from collections import defaultdict

    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)

    base_path = Path(data_cfg["path"])
    split_path = data_cfg.get(args.split, data_cfg.get("val"))
    images_dir = base_path / split_path
    labels_dir = images_dir.parent / "labels" if "images" in str(images_dir) else base_path / "labels"

    if not labels_dir.exists():
        labels_dir = images_dir.with_name("labels")

    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    print(f"Evaluating {len(image_files)} images from {images_dir}")

    all_preds = []
    all_gts = []

    for img_path in image_files:
        bgr = cv2.imread(str(img_path))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        img_input = np.stack([gray, gray, gray], axis=-1)

        results = model(img_input, conf=args.conf, iou=args.iou, verbose=False)[0]

        preds = []
        if results.boxes is not None and len(results.boxes) > 0:
            for i in range(len(results.boxes)):
                box = results.boxes.xyxy[i].cpu().numpy()
                conf = float(results.boxes.conf[i].cpu().numpy())
                cls_id = int(results.boxes.cls[i].cpu().numpy())
                preds.append((cls_id, box, conf))
        all_preds.append(preds)

        label_path = labels_dir / f"{img_path.stem}.txt"
        gts = []
        if label_path.exists():
            h, w = bgr.shape[:2]
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        x2 = (cx + bw / 2) * w
                        y2 = (cy + bh / 2) * h
                        gts.append((cls_id, np.array([x1, y1, x2, y2])))
        all_gts.append(gts)

    compute_and_print_metrics(all_preds, all_gts, args.iou)


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
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


def compute_class_ap(all_preds, all_gts, cls_id, iou_thresh):
    """Compute AP for a single class at a given IoU threshold."""
    all_detections = []
    total_gt = 0

    for img_idx, (preds, gts) in enumerate(zip(all_preds, all_gts)):
        class_gts = [(i, gt_box) for i, (gt_cls, gt_box) in enumerate(gts) if gt_cls == cls_id]
        total_gt += len(class_gts)

        for pred_cls, box, conf in preds:
            if pred_cls == cls_id:
                all_detections.append((img_idx, box, conf))

    if total_gt == 0:
        return 0.0

    all_detections.sort(key=lambda x: -x[2])

    gt_matched = {}
    for img_idx, (preds, gts) in enumerate(zip(all_preds, all_gts)):
        gt_matched[img_idx] = [False] * len(gts)

    tps = []
    fps = []

    for img_idx, box, conf in all_detections:
        gts = all_gts[img_idx]
        matched = False
        best_iou = 0
        best_gt_idx = -1

        for gi, (gt_cls, gt_box) in enumerate(gts):
            if gt_cls != cls_id or gt_matched[img_idx][gi]:
                continue
            iou = compute_iou(box, gt_box)
            if iou >= iou_thresh and iou > best_iou:
                best_iou = iou
                best_gt_idx = gi
                matched = True

        if matched:
            gt_matched[img_idx][best_gt_idx] = True
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)

    tps = np.cumsum(tps)
    fps = np.cumsum(fps)

    recalls = tps / total_gt
    precisions = tps / (tps + fps)

    if len(recalls) == 0:
        return 0.0

    return compute_ap(precisions, recalls)


def compute_and_print_metrics(all_preds, all_gts, iou_thresh):
    """Compute and print evaluation metrics including mAP at multiple IoU thresholds."""
    from collections import defaultdict

    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_fn = defaultdict(int)

    for preds, gts in zip(all_preds, all_gts):
        gt_matched = [False] * len(gts)

        preds_sorted = sorted(preds, key=lambda x: -x[2])

        for cls_id, box, conf in preds_sorted:
            matched = False
            for gi, (gt_cls, gt_box) in enumerate(gts):
                if gt_matched[gi]:
                    continue
                if gt_cls == cls_id and compute_iou(box, gt_box) >= iou_thresh:
                    per_class_tp[cls_id] += 1
                    gt_matched[gi] = True
                    matched = True
                    break
            if not matched:
                per_class_fp[cls_id] += 1

        for gi, (gt_cls, gt_box) in enumerate(gts):
            if not gt_matched[gi]:
                per_class_fn[gt_cls] += 1

    print(f"\n── Per-class Metrics (IoU={iou_thresh}) ──────────────────────────")
    print(f"{'Class':<20} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 70)

    total_tp, total_fp, total_fn = 0, 0, 0
    for i, name in enumerate(CLASS_NAMES):
        tp = per_class_tp[i]
        fp = per_class_fp[i]
        fn = per_class_fn[i]
        total_tp += tp
        total_fp += fp
        total_fn += fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
        print(f"  {name:<20} {tp:>4} {fp:>6} {fn:>6} {prec:>8.4f} {recall:>8.4f} {f1:>8.4f}")

    print("-" * 70)
    total_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1 = 2 * total_prec * total_recall / (total_prec + total_recall) if (total_prec + total_recall) > 0 else 0
    print(f"  {'TOTAL':<20} {total_tp:>4} {total_fp:>6} {total_fn:>6} {total_prec:>8.4f} {total_recall:>8.4f} {total_f1:>8.4f}")

    iou_thresholds = [0.25, 0.50, 0.75]

    print(f"\n── Per-class AP at Different IoU Thresholds ──────────────────────────")
    print(f"{'Class':<20} {'AP@25':>10} {'AP@50':>10} {'AP@75':>10}")
    print("-" * 55)

    all_aps = {iou: [] for iou in iou_thresholds}

    for i, name in enumerate(CLASS_NAMES):
        row = f"  {name:<20}"
        for iou in iou_thresholds:
            ap = compute_class_ap(all_preds, all_gts, i, iou)
            all_aps[iou].append(ap)
            row += f" {ap:>9.4f}"
        print(row)

    print("-" * 55)
    row = f"  {'mAP':<20}"
    for iou in iou_thresholds:
        map_val = np.mean(all_aps[iou]) if all_aps[iou] else 0
        row += f" {map_val:>9.4f}"
    print(row)

    print(f"\n── Summary ─────────────────────────────────")
    print(f"  mAP@25      {np.mean(all_aps[0.25]):.4f}")
    print(f"  mAP@50      {np.mean(all_aps[0.50]):.4f}")
    print(f"  mAP@75      {np.mean(all_aps[0.75]):.4f}")
    print(f"  Precision   {total_prec:.4f}")
    print(f"  Recall      {total_recall:.4f}")
    print(f"  F1          {total_f1:.4f}")


def print_metrics(metrics):
    """Print metrics from YOLO val() results."""
    print("\n── Per-class AP50 ──────────────────────────")
    for i, name in enumerate(CLASS_NAMES):
        ap = metrics.box.ap50[i] if i < len(metrics.box.ap50) else float("nan")
        print(f"  {name:<20} {ap:.4f}")

    print("\n── Summary ─────────────────────────────────")
    print(f"  mAP@50       {metrics.box.map50:.4f}")
    print(f"  mAP@50-95    {metrics.box.map:.4f}")
    print(f"  Precision    {metrics.box.mp:.4f}")
    print(f"  Recall       {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
