import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
DATA = ROOT / "pcb-defect-dataset/data_synth_aligned.yaml"
OUT  = ROOT / "trained/yolo_real"


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 on real PCB defect dataset")
    p.add_argument("--model",   default=str(ROOT / "models/yolov8m.pt"),
                   help="Base weights — pass path to synth best.pt to fine-tune from synthetic")
    p.add_argument("--epochs",  type=int,   default=100)
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--batch",   type=int,   default=8)
    p.add_argument("--workers", type=int,   default=0)
    p.add_argument("--lr",      type=float, default=1e-3,
                   help="Initial LR — use ~1e-4 when fine-tuning from synthetic weights")
    p.add_argument("--name",    default="real", help="Run name under trained/real/")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=str(DATA),
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
    )
    best = OUT / args.name / "weights/best.pt"
    print(f"\nBest weights: {best}")


if __name__ == "__main__":
    main()
