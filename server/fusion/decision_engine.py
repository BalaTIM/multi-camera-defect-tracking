"""
Multi-View Decision Fusion Engine
-----------------------------------
Aggregates defect signals from multiple cameras for the same global object
and produces a final decision.

Decision rules:
  - DEFECT_CONFIRMED : ≥ min_cameras confirmed defect within temporal window
  - UNCERTAIN        : exactly 1 camera flagged defect (below threshold)
  - OK               : no camera flagged defect
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from loguru import logger


Decision = str  # "DEFECT_CONFIRMED" | "UNCERTAIN" | "OK"


@dataclass
class ObjectEvidence:
    global_id: str
    observations: list[dict] = field(default_factory=list)


class DecisionEngine:
    def __init__(self, temporal_window_ms: float = 100.0, min_cameras: int = 2):
        """
        Args:
            temporal_window_ms: Time window to group observations across cameras.
            min_cameras: Minimum cameras agreeing for DEFECT_CONFIRMED.
        """
        self.temporal_window_s = temporal_window_ms / 1000.0
        self.min_cameras = min_cameras

        # global_id -> list of recent observations
        self._evidence: dict[str, list[dict]] = defaultdict(list)

    def decide(
        self,
        global_id: str,
        camera_id: str,
        defect_flag: bool,
        timestamp: float,
    ) -> Decision:
        """
        Record an observation and return updated decision for this global_id.
        """
        self._evidence[global_id].append({
            "camera_id": camera_id,
            "defect_flag": defect_flag,
            "timestamp": timestamp,
        })

        # Keep only observations within the temporal window
        cutoff = timestamp - self.temporal_window_s
        self._evidence[global_id] = [
            obs for obs in self._evidence[global_id]
            if obs["timestamp"] >= cutoff
        ]

        return self._compute_decision(global_id)

    def _compute_decision(self, global_id: str) -> Decision:
        observations = self._evidence.get(global_id, [])

        # Deduplicate: one vote per camera (most recent)
        latest_per_cam: dict[str, bool] = {}
        for obs in sorted(observations, key=lambda o: o["timestamp"]):
            latest_per_cam[obs["camera_id"]] = obs["defect_flag"]

        defect_cameras = [cam for cam, flag in latest_per_cam.items() if flag]
        total_cameras = len(latest_per_cam)

        if len(defect_cameras) >= self.min_cameras:
            return "DEFECT_CONFIRMED"
        elif len(defect_cameras) >= 1:
            return "UNCERTAIN"
        else:
            return "OK"

    def clear(self, global_id: str):
        """Remove evidence for a global object (e.g. after it leaves scene)."""
        self._evidence.pop(global_id, None)
