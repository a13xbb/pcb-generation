import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
DATA = ROOT / "pcb-defect-dataset/data_synth_aligned.yaml"

CLASS_NAMES = ["mouse_bite", "spur", "missing_hole", "open_circuit", "spurious_copper"]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a YOLOv8 model on the real PCB test set")
    p.add_argument("weights", help="Path to model weights (.pt)")
    p.add_argument("--imgsz",  type=int,   default=640)
    p.add_argument("--batch",  type=int,   default=16)
    p.add_argument("--conf",   type=float, default=0.25)
    p.add_argument("--iou",    type=float, default=0.5)
    p.add_argument("--workers",type=int,   default=0)
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    metrics = model.val(
        data=str(DATA),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        workers=args.workers,
        plots=True,
        save_json=True,
    )

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
