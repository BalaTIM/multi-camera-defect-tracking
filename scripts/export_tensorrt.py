"""
TensorRT Export Script
-----------------------
Exports YOLOv8 model to TensorRT for 3-4x speedup on Jetson / NVIDIA GPUs.

Requirements:
  - NVIDIA GPU with CUDA
  - TensorRT installed (comes with Jetson JetPack or TensorRT SDK)

Usage:
    python scripts/export_tensorrt.py --weights models/defect_yolov8m.pt
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 to TensorRT")
    parser.add_argument("--weights", default="models/defect_yolov8m.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1, help="Batch size (1 for edge inference)")
    parser.add_argument("--int8", action="store_true", help="INT8 quantization (requires calibration data)")
    parser.add_argument("--fp16", action="store_true", default=True, help="FP16 half precision (default: on)")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    logger.info(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    precision = "int8" if args.int8 else ("fp16" if args.fp16 else "fp32")
    logger.info(f"Exporting to TensorRT [{precision}] — this may take several minutes...")

    model.export(
        format="engine",
        imgsz=args.imgsz,
        batch=args.batch,
        half=args.fp16 and not args.int8,
        int8=args.int8,
        device=args.device,
        simplify=True,
        workspace=4,    # GB
    )

    engine_path = Path(args.weights).with_suffix(".engine")
    if engine_path.exists():
        logger.success(f"TensorRT engine saved: {engine_path}")
        logger.info("Update configs/config.yaml: weights_path → models/defect_yolov8m.engine")
    else:
        logger.error("Export may have failed — check logs above.")


if __name__ == "__main__":
    main()
