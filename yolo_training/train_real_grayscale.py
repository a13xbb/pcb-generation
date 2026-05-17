#!/usr/bin/env python3
"""Train YOLOv8 on real PCB dataset with grayscale + manual zoom-in crops.

Per image, per epoch (training only):
  1. Convert to grayscale
  2. With prob crop_p → random zoom-in crop

Val images are grayscale only, no crop augmentation.

Usage:
    python train_real_grayscale.py
    python train_real_grayscale.py --crop_p 0.3 --epochs 100
    python train_real_grayscale.py --preview  # visualize augmentations
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
from augmentations import to_grayscale, sample_crop_params, apply_crop_zoom

ROOT = Path(__file__).parent.parent
DATA = ROOT / "datasets/pcb-defect-dataset/data_synth_aligned.yaml"
OUT = ROOT / "yolo_training/results"


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 on real PCB with grayscale + zoom-in augmentation")
    p.add_argument("--model", default=str(ROOT / "models/yolov8m.pt"))
    p.add_argument("--data", default=str(DATA), help="data.yaml for the dataset")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--crop_p", type=float, default=0.3,
                   help="Per-image probability of applying a zoom-in crop")
    p.add_argument("--crop_scale_min", type=float, default=0.3)
    p.add_argument("--crop_scale_max", type=float, default=0.7)
    p.add_argument("--mosaic", type=float, default=0.1,
                   help="YOLO mosaic probability")
    p.add_argument("--name", default="real_grayscale_v1")
    p.add_argument("--preview", action="store_true",
                   help="Save augmentation examples to debug_output/real_grayscale_preview and exit")
    p.add_argument("--preview_n", type=int, default=8,
                   help="Number of source images to preview")
    return p.parse_args()


def _draw_boxes(img: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Draw YOLO-format bounding boxes [cls, cx, cy, w, h] normalised onto img."""
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
    """Save augmentation examples for visual inspection.

    For each sampled source image writes three variants:
      original, gray, gray+crop
    Bounding boxes are drawn in green on every variant.
    """
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    # Resolve path relative to yaml file location
    cfg_path = Path(cfg["path"])
    if not cfg_path.is_absolute():
        src_path = (data_yaml.parent / cfg_path).resolve()
    else:
        src_path = cfg_path

    train_val = cfg.get("train", "images")

    # Handle txt file vs directory
    all_imgs = []
    if train_val.endswith(".txt"):
        txt_path = src_path / train_val
        if txt_path.exists():
            for line in txt_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                p = Path(line)
                if p.exists():
                    all_imgs.append(p)
                else:
                    # Try to find image by name in train/images
                    alt_path = src_path / "train" / "images" / p.name
                    if alt_path.exists():
                        all_imgs.append(alt_path)
    else:
        img_dir = src_path / train_val
        if not img_dir.is_dir():
            img_dir = src_path / "images" / train_val
        if not img_dir.is_dir():
            img_dir = src_path / "images"
        all_imgs = sorted(p for p in img_dir.glob("*")
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    if not all_imgs:
        print(f"No images found for train={train_val} in {src_path}")
        return

    chosen = sorted(all_imgs)[:n_images] if len(all_imgs) >= n_images else all_imgs
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve labels dir
    lbl_dir = src_path / "train" / "labels"
    if not lbl_dir.is_dir():
        lbl_dir = src_path / "labels"

    for img_path in chosen:
        color = cv2.imread(str(img_path))
        if color is None:
            continue

        # Load labels
        lbl_file = lbl_dir / (img_path.stem + ".txt")
        labels = []
        if lbl_file.exists():
            for line in lbl_file.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels.append([float(v) for v in parts[:5]])
        labels = np.array(labels) if labels else np.array([]).reshape(0, 5)

        h, w = color.shape[:2]
        stem = img_path.stem

        # Prepare gray from full color image
        gray = to_grayscale(color)

        # Compute one crop region
        y, x, ch, cw = sample_crop_params(h, w, crop_scale_range)
        gray_crop, crop_labels = apply_crop_zoom(gray, labels, y, x, ch, cw)

        variants = [
            (f"{stem}_1_original.jpg", color, labels),
            (f"{stem}_2_gray.jpg", gray, labels),
            (f"{stem}_3_gray_crop.jpg", gray_crop, crop_labels),
        ]
        for fname, img, lbls in variants:
            cv2.imwrite(str(out_dir / fname), _draw_boxes(img, lbls))

    print(f"Saved {len(chosen) * 3} preview images -> {out_dir}")


class RealPCBDataset(YOLODataset):
    """YOLODataset with grayscale conversion and zoom-in crop augmentation.

    load_image applies grayscale conversion every time an image is loaded.
    get_image_and_label then optionally crops.
    Both augmentations are disabled for the val split (pcb_augment=False).
    """

    def __init__(self, *args, crop_p=0.3,
                 crop_scale_range=(0.3, 0.7), pcb_augment=True, **kwargs):
        self.crop_p = crop_p
        self.crop_scale_range = crop_scale_range
        self.pcb_augment = pcb_augment
        super().__init__(*args, **kwargs)

    def load_image(self, i):
        img, orig_shape, resized_shape = super().load_image(i)
        return to_grayscale(img), orig_shape, resized_shape

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


class RealPCBTrainer(DetectionTrainer):
    """DetectionTrainer using RealPCBDataset for grayscale + zoom-in augmentation."""

    def __init__(self, *args, crop_p=0.3, crop_scale_range=(0.3, 0.7), **kwargs):
        self.crop_p = crop_p
        self.crop_scale_range = crop_scale_range
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path, mode="train", batch=None):
        model = self.model.module if is_parallel(self.model) else self.model if self.model else None
        gs = max(int(model.stride.max()) if model else 0, 32)
        is_train = mode == "train"
        return RealPCBDataset(
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
            # PCB-specific
            crop_p=self.crop_p,
            crop_scale_range=self.crop_scale_range,
            pcb_augment=is_train,
        )


def main():
    args = parse_args()

    if args.preview:
        save_preview(
            Path(args.data),
            ROOT / "debug_output" / "real_grayscale_preview",
            n_images=args.preview_n,
            crop_scale_range=(args.crop_scale_min, args.crop_scale_max),
        )
        return

    print("=" * 60)
    print("Real PCB Training - grayscale + zoom-in augmentation")
    print("=" * 60)
    print(f"  crop_p     : {args.crop_p}  scale [{args.crop_scale_min}, {args.crop_scale_max}]")
    print(f"  mosaic     : {args.mosaic}")
    print(f"  data       : {args.data}")
    print("=" * 60)

    trainer = RealPCBTrainer(
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
            # Same augmentations as train_grayscale_synth_v2.py
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.0,       # zoom-out disabled; crop_p handles zoom-in
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=args.mosaic,
            mixup=0.0,
            copy_paste=0.0,
        ),
        crop_p=args.crop_p,
        crop_scale_range=(args.crop_scale_min, args.crop_scale_max),
    )

    trainer.train()

    best = OUT / args.name / "weights/best.pt"
    print(f"\nBest weights: {best}")


if __name__ == "__main__":
    main()
