"""
YOLO Detector
-------------
Wraps YOLOv8 (Ultralytics) for defect detection.
Returns structured Detection objects with bounding boxes, labels, and confidence.
"""

from dataclasses import dataclass
import numpy as np
import torch
from ultralytics import YOLO
from loguru import logger


@dataclass
class Detection:
    bbox: list[float]       # [x1, y1, x2, y2] in pixel coords
    label: str
    confidence: float
    class_id: int


class YOLODetector:
    def __init__(
        self,
        weights: str = "models/defect_yolov8m.pt",
        conf: float = 0.5,
        iou: float = 0.45,
        device: str = "auto",
    ):
        self.conf = conf
        self.iou = iou

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading YOLO model from {weights} on {self.device}")
        self.model = YOLO(weights)
        self.model.to(self.device)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a single frame.

        Args:
            frame: BGR numpy array (HxWx3).

        Returns:
            List of Detection objects. Empty list if nothing found.
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device=self.device,
        )

        detections = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())
                label = self.model.names[cls_id]

                detections.append(Detection(
                    bbox=xyxy,
                    label=label,
                    confidence=conf,
                    class_id=cls_id,
                ))

        return detections

    def crop_detections(self, frame: np.ndarray, detections: list[Detection]) -> list[np.ndarray]:
        """
        Crop bounding box regions from the frame for embedding extraction.

        Returns:
            List of cropped BGR numpy arrays.
        """
        crops = []
        h, w = frame.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
        return crops
