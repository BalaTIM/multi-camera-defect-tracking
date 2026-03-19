"""
DeepSORT Local Tracker
-----------------------
Maintains per-camera object identities across frames.
Combines IoU-based matching with appearance embeddings.
"""

from dataclasses import dataclass
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from loguru import logger

from inference.yolo_detector import Detection


@dataclass
class Track:
    track_id: int           # Local (per-camera) track ID
    bbox: list[float]       # [x1, y1, x2, y2]
    label: str
    confidence: float
    embedding: list[float]  # Normalized feature vector
    is_defect: bool


class DeepSORTTracker:
    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_distance: float = 0.4,
        nn_budget: int = 100,
    ):
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            override_track_class=None,
            embedder=None,       # We supply our own embeddings
        )
        logger.info("DeepSORT tracker initialized.")

    # --- Defect class names that trigger alert ---
    DEFECT_LABELS = {"crack", "scratch", "dent", "corrosion", "hole", "stain"}

    def update(
        self,
        detections: list[Detection],
        embeddings: list[np.ndarray],
        frame: np.ndarray,
    ) -> list[Track]:
        """
        Update tracker state with new detections.

        Args:
            detections: List of Detection from YOLO.
            embeddings: Parallel list of feature vectors (one per detection).
            frame: Current BGR frame (used internally by DeepSort if needed).

        Returns:
            List of confirmed Track objects.
        """
        if not detections:
            self.tracker.update_tracks([], frame=frame)
            return []

        # Format for deep_sort_realtime: ([x1,y1,w,h], conf, class_label, embedding)
        raw_detections = []
        for det, emb in zip(detections, embeddings):
            x1, y1, x2, y2 = det.bbox
            w, h = x2 - x1, y2 - y1
            raw_detections.append(([x1, y1, w, h], det.confidence, det.label, emb))

        tracks_raw = self.tracker.update_tracks(raw_detections, frame=frame)

        tracks = []
        for i, t in enumerate(tracks_raw):
            if not t.is_confirmed():
                continue

            ltrb = t.to_ltrb()
            embedding = embeddings[i] if i < len(embeddings) else []

            tracks.append(Track(
                track_id=int(t.track_id),
                bbox=ltrb.tolist(),
                label=t.det_class or "unknown",
                confidence=t.det_conf or 0.0,
                embedding=embedding.tolist() if hasattr(embedding, "tolist") else embedding,
                is_defect=(t.det_class in self.DEFECT_LABELS),
            ))

        return tracks
