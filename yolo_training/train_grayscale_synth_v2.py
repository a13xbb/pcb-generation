#!/usr/bin/env python3
"""Train YOLOv8 on synthetic PCB dataset with on-the-fly augmentations.

Per image, per epoch (training only):
  1. With prob binarize_p → binarize (on the color image); else → grayscale
  2. With prob crop_p     → random zoom-in crop

Val images are always converted to grayscale with no other transforms.
No preprocessed dataset is written to disk — original color images are used directly.

Usage:
    python train_grayscale_synth_v2.py --data path/to/data.yaml
    python train_grayscale_synth_v2.py --binarize_p 0.3 --crop_p 0.3 --epochs 150
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import is_parallel

sys.path.insert(0, str(Path(__file__).parent))
from augmentations import to_grayscale, binarize_synth, sample_crop_params, apply_crop_zoom

ROOT = Path(__file__).parent.parent
DEFAULT_DATA = ROOT / "datasets/generated_dataset/v2/defects/data.yaml"
OUT = ROOT / "yolo_training/results/synth_grayscale_aug_v2"


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 with on-the-fly gray/binary/crop augmentation")
    p.add_argument("--model", default=str(ROOT / "models/yolov8m.pt"))
    p.add_argument("--data", default=str(DEFAULT_DATA), help="data.yaml for the source (color) dataset")
    p.add_argument("--epochs", type=int, default=150)
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
    p.add_argument("--mosaic", type=float, default=0.0,
                   help="YOLO mosaic probability (default 0 — avoids implicit zoom-out)")
    p.add_argument("--name", default="grayscale_synth_v2")
    return p.parse_args()


class PCBDataset(YOLODataset):
    """YOLODataset with on-the-fly grayscale/binary and zoom-in crop augmentation.

    load_image applies gray or binary conversion every time an image is loaded,
    giving fresh randomness each epoch. get_image_and_label then optionally crops.
    Both augmentations are disabled for the val split (pcb_augment=False).
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
        # Binarize on the color image before any further processing
        if self.pcb_augment and np.random.random() < self.binarize_p:
            img = binarize_synth(img)
        else:
            img = to_grayscale(img)
        return img, orig_shape, resized_shape

    def get_image_and_label(self, index):
        label = super().get_image_and_label(index)  # calls load_image → gray/binary already applied
        if not self.pcb_augment:
            return label

        bboxes = label.get("bboxes", np.array([]))
        if len(bboxes) == 0 or np.random.random() >= self.crop_p:
            return label

        img = label["img"]
        h, w = img.shape[:2]
        y, x, crop_h, crop_w = sample_crop_params(h, w, self.crop_scale_range)

        # apply_crop_zoom expects (N, 5) with [cls, cx, cy, w, h]
        combined = np.hstack([label["cls"], bboxes])
        img_crop, combined_crop = apply_crop_zoom(img, combined, y, x, crop_h, crop_w)

        if len(combined_crop) > 0:
            label["img"] = img_crop
            label["cls"] = combined_crop[:, :1]
            label["bboxes"] = combined_crop[:, 1:]

        return label


class PCBTrainer(DetectionTrainer):
    """DetectionTrainer that uses PCBDataset for on-the-fly PCB augmentations."""

    def __init__(self, *args, binarize_p=0.3, crop_p=0.3,
                 crop_scale_range=(0.3, 0.7), **kwargs):
        # Store before super().__init__ in case build_dataset is called during init
        self.binarize_p = binarize_p
        self.crop_p = crop_p
        self.crop_scale_range = crop_scale_range
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path, mode="train", batch=None):
        model = self.model.module if is_parallel(self.model) else self.model if self.model else None
        gs = max(int(model.stride.max()) if model else 0, 32)
        is_train = mode == "train"
        return PCBDataset(
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
            task=self.task,
            classes=self.args.classes,
            data=self.data,
            fraction=getattr(self.args, "fraction", 1.0) if is_train else 1.0,
            # PCB-specific
            binarize_p=self.binarize_p,
            crop_p=self.crop_p,
            crop_scale_range=self.crop_scale_range,
            pcb_augment=is_train,
        )


def main():
    args = parse_args()

    print("=" * 60)
    print("Synthetic PCB Training — on-the-fly augmentation (v2)")
    print("=" * 60)
    print(f"  binarize_p : {args.binarize_p}  (else grayscale)")
    print(f"  crop_p     : {args.crop_p}  scale [{args.crop_scale_min}, {args.crop_scale_max}]")
    print(f"  mosaic     : {args.mosaic}")
    print(f"  data       : {args.data}")
    print("=" * 60)

    trainer = PCBTrainer(
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
            # Disable color-space YOLO augmentations (images are gray/binary)
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
        binarize_p=args.binarize_p,
        crop_p=args.crop_p,
        crop_scale_range=(args.crop_scale_min, args.crop_scale_max),
    )

    trainer.train()

    best = OUT / args.name / "weights/best.pt"
    print(f"\nBest weights: {best}")


if __name__ == "__main__":
    main()
