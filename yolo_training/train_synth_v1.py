import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
DATA = ROOT / "generated_dataset/dataset_full_pipe_cluster/data.yaml"
OUT  = ROOT / "trained/yolo_synth"


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 on synthetic PCB defect dataset")
    p.add_argument("--model",   default=str(ROOT / "models/yolov8m.pt"), help="Base weights (e.g. yolov8s.pt / yolov8m.pt)")
    p.add_argument("--epochs",  type=int,   default=100)
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--batch",   type=int,   default=8)
    p.add_argument("--workers", type=int,   default=0)
    p.add_argument("--name",    default="synth", help="Run name under trained/synth/")
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
