"""
YOLOv8 Fine-tuning Script
--------------------------
Fine-tunes a YOLOv8 model on your industrial defect dataset.

Dataset format expected (Ultralytics YOLO format):
    datasets/
      defects/
        images/
          train/  *.jpg
          val/    *.jpg
        labels/
          train/  *.txt   (YOLO format: class cx cy w h)
          val/    *.txt
        data.yaml

Usage:
    python scripts/train_model.py --data datasets/defects/data.yaml --epochs 100
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
from loguru import logger


DATA_YAML_TEMPLATE = """
path: {dataset_root}
train: images/train
val: images/val

nc: 5
names:
  0: crack
  1: scratch
  2: dent
  3: corrosion
  4: hole
"""


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for defect detection")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8m.pt", help="Base model (yolov8n/s/m/l/x)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="GPU index or 'cpu'")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="defect_detector")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(f"data.yaml not found: {args.data}")

    logger.info(f"Loading base model: {args.model}")
    model = YOLO(args.model)

    logger.info(f"Starting training for {args.epochs} epochs...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        patience=20,          # Early stopping
        save=True,
        plots=True,
        val=True,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
    )

    # Export best weights
    best = Path(args.project) / args.name / "weights" / "best.pt"
    if best.exists():
        import shutil
        shutil.copy(best, "models/defect_yolov8m.pt")
        logger.success(f"Best weights saved to models/defect_yolov8m.pt")

    logger.info("Training complete. Evaluate with:")
    logger.info(f"  yolo val model=models/defect_yolov8m.pt data={args.data}")


if __name__ == "__main__":
    main()
