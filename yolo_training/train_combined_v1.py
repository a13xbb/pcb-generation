#!/usr/bin/env python3
"""Train YOLOv8 on combined real + synthetic PCB dataset with on-the-fly augmentations.

Per image, per epoch (training only):
  1. With prob binarize_p → binarize (real images use binarize_real, synthetic use binarize_synth)
     else → grayscale
  2. With prob crop_p → random zoom-in crop

Val images are always converted to grayscale with no other transforms.
Image source (real vs synthetic) is detected by checking if path contains 'generated_dataset'.

Usage:
    python train_combined_v1.py
    python train_combined_v1.py --binarize_p 0.3 --crop_p 0.3 --epochs 100
    python train_combined_v1.py --preview
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import is_parallel

sys.path.insert(0, str(Path(__file__).parent))
from augmentations import (
    to_grayscale, binarize_real, binarize_synth,
    sample_crop_params, apply_crop_zoom
)

ROOT = Path(__file__).parent.parent
DEFAULT_DATA = ROOT / "datasets/combined/data.yaml"
OUT = ROOT / "yolo_training/results"


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 on combined real+synthetic PCB with gray/binary/crop augmentation")
    p.add_argument("--model", default=str(ROOT / "models/yolov8m.pt"))
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--binarize_p", type=float, default=0.3,
                   help="Per-image probability of binarizing instead of grayscaling")
    p.add_argument("--crop_p", type=float, default=0.3,
                   help="Per-image probability of applying a zoom-in crop")
    p.add_argument("--crop_scale_min", type=float, default=0.3)
    p.add_argument("--crop_scale_max", type=float, default=0.7)
    p.add_argument("--mosaic", type=float, default=0.1)
    p.add_argument("--name", default="combined_v1")
    p.add_argument("--preview", action="store_true",
                   help="Save augmentation examples to debug_output/combined_v1_preview and exit")
    p.add_argument("--preview_n", type=int, default=8)
    return p.parse_args()


def _is_synthetic(img_path: str) -> bool:
    """Check if image is from synthetic dataset based on path."""
    return "generated_dataset" in img_path


def _draw_boxes(img: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    for lbl in labels:
        cls, cx, cy, bw, bh = lbl
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, str(int(cls)), (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def save_preview(data_yaml: Path, out_dir: Path, n_images: int,
                 crop_scale_range: tuple) -> None:
    """Save augmentation examples: original, gray, binary, gray+crop, binary+crop.

    Samples from both real and synthetic images to show different binarization.
    """
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    cfg_path = Path(cfg["path"])
    train_txt = cfg_path / cfg.get("train", "train.txt")

    all_imgs = []
    if train_txt.exists():
        for line in train_txt.read_text().splitlines():
            line = line.strip()
            if line:
                all_imgs.append(Path(line))

    if not all_imgs:
        print(f"No images found in {train_txt}")
        return

    # Sample half from real, half from synthetic
    real_imgs = [p for p in all_imgs if not _is_synthetic(str(p))]
    synth_imgs = [p for p in all_imgs if _is_synthetic(str(p))]

    n_real = min(n_images // 2, len(real_imgs))
    n_synth = min(n_images - n_real, len(synth_imgs))

    chosen = real_imgs[:n_real] + synth_imgs[:n_synth]
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in chosen:
        color = cv2.imread(str(img_path))
        if color is None:
            continue

        # Find labels file
        lbl_file = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
        labels = []
        if lbl_file.exists():
            for line in lbl_file.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels.append([float(v) for v in parts[:5]])
        labels = np.array(labels) if labels else np.array([]).reshape(0, 5)

        h, w = color.shape[:2]
        stem = img_path.stem
        is_synth = _is_synthetic(str(img_path))
        prefix = "synth" if is_synth else "real"

        gray = to_grayscale(color)
        binary = binarize_synth(color) if is_synth else binarize_real(color)

        y, x, ch, cw = sample_crop_params(h, w, crop_scale_range)
        gray_crop, crop_labels = apply_crop_zoom(gray, labels, y, x, ch, cw)
        binary_crop, _ = apply_crop_zoom(binary, labels, y, x, ch, cw)

        variants = [
            (f"{prefix}_{stem}_1_original.jpg",  color,       labels),
            (f"{prefix}_{stem}_2_gray.jpg",      gray,        labels),
            (f"{prefix}_{stem}_3_binary.jpg",    binary,      labels),
            (f"{prefix}_{stem}_4_gray_crop.jpg", gray_crop,   crop_labels),
            (f"{prefix}_{stem}_5_bin_crop.jpg",  binary_crop, crop_labels),
        ]
        for fname, img, lbls in variants:
            cv2.imwrite(str(out_dir / fname), _draw_boxes(img, lbls))

    print(f"Saved {len(chosen) * 5} preview images ({n_real} real + {n_synth} synth sources) -> {out_dir}")


class CombinedPCBDataset(YOLODataset):
    """YOLODataset with on-the-fly grayscale/binarize and zoom-in crop augmentation.

    Detects image source by path to apply correct binarization algorithm:
    - Real images (pcb-defect-dataset): binarize_real
    - Synthetic images (generated_dataset): binarize_synth
    """

    def __init__(self, *args, binarize_p=0.3, crop_p=0.3,
                 crop_scale_range=(0.3, 0.7), pcb_augment=True, **kwargs):
        self.binarize_p = binarize_p
        self.crop_p = crop_p
        self.crop_scale_range = crop_scale_range
        self.pcb_augment = pcb_augment
        super().__init__(*args, **kwargs)

    def load_image(self, i):
        img, orig_shape, resized_shape = super().load_image(i)
        img_path = str(self.im_files[i])

        if self.pcb_augment and np.random.random() < self.binarize_p:
            if _is_synthetic(img_path):
                img = binarize_synth(img)
            else:
                img = binarize_real(img)
        else:
            img = to_grayscale(img)

        return img, orig_shape, resized_shape

    def get_image_and_label(self, index):
        label = super().get_image_and_label(index)
        if not self.pcb_augment:
            return label

        bboxes = label.get("bboxes", np.array([]))
        if len(bboxes) == 0 or np.random.random() >= self.crop_p:
            return label

        img = label["img"]
        h, w = img.shape[:2]
        y, x, crop_h, crop_w = sample_crop_params(h, w, self.crop_scale_range)

        combined = np.hstack([label["cls"], bboxes])
        img_crop, combined_crop = apply_crop_zoom(img, combined, y, x, crop_h, crop_w)

        if len(combined_crop) > 0:
            label["img"] = img_crop
            label["cls"] = combined_crop[:, :1]
            label["bboxes"] = combined_crop[:, 1:]

        return label


class CombinedPCBTrainer(DetectionTrainer):
    """DetectionTrainer using CombinedPCBDataset for on-the-fly PCB augmentations."""

    def __init__(self, *args, binarize_p=0.3, crop_p=0.3,
                 crop_scale_range=(0.3, 0.7), **kwargs):
        self.binarize_p = binarize_p
        self.crop_p = crop_p
        self.crop_scale_range = crop_scale_range
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path, mode="train", batch=None):
        model = self.model.module if is_parallel(self.model) else self.model if self.model else None
        gs = max(int(model.stride.max()) if model else 0, 32)
        is_train = mode == "train"
        return CombinedPCBDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=is_train,
            hyp=self.args,
            rect=self.args.rect or not is_train,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0 if is_train else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=getattr(self, "task", "detect"),
            classes=self.args.classes,
            data=self.data,
            fraction=getattr(self.args, "fraction", 1.0) if is_train else 1.0,
            binarize_p=self.binarize_p,
            crop_p=self.crop_p,
            crop_scale_range=self.crop_scale_range,
            pcb_augment=is_train,
        )


def main():
    args = parse_args()

    if args.preview:
        save_preview(
            Path(args.data),
            ROOT / "debug_output" / "combined_v1_preview",
            n_images=args.preview_n,
            crop_scale_range=(args.crop_scale_min, args.crop_scale_max),
        )
        return

    print("=" * 60)
    print("Combined PCB Training — real + synthetic with on-the-fly augmentation (v1)")
    print("=" * 60)
    print(f"  binarize_p : {args.binarize_p}  (else grayscale)")
    print(f"  crop_p     : {args.crop_p}  scale [{args.crop_scale_min}, {args.crop_scale_max}]")
    print(f"  mosaic     : {args.mosaic}")
    print(f"  data       : {args.data}")
    print("=" * 60)

    trainer = CombinedPCBTrainer(
        overrides=dict(
            model=args.model,
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            lr0=args.lr,
            project=str(OUT),
            name=args.name,
            exist_ok=True,
            patience=20,
            save=True,
            plots=True,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.0,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=args.mosaic,
            mixup=0.0,
            copy_paste=0.0,
        ),
        binarize_p=args.binarize_p,
        crop_p=args.crop_p,
        crop_scale_range=(args.crop_scale_min, args.crop_scale_max),
    )

    trainer.train()

    best = OUT / args.name / "weights/best.pt"
    print(f"\nBest weights: {best}")


if __name__ == "__main__":
    main()
