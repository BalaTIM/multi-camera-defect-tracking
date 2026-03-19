"""
Global Object Tracker
----------------------
Maintains the last known state (position, camera, timestamp) for every
globally identified object across all camera views.
"""

import time
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger


@dataclass
class GlobalObject:
    global_id: str
    camera_id: str
    bbox: list[float]
    last_seen: float
    history: list[dict] = field(default_factory=list)  # recent positions


class GlobalTracker:
    def __init__(self, max_history: int = 30, stale_timeout_s: float = 5.0):
        self._objects: dict[str, GlobalObject] = {}
        self.max_history = max_history
        self.stale_timeout_s = stale_timeout_s

    def update(
        self,
        global_id: str,
        camera_id: str,
        bbox: list[float],
        timestamp: float,
    ):
        """Update or create a global object state."""
        if global_id not in self._objects:
            self._objects[global_id] = GlobalObject(
                global_id=global_id,
                camera_id=camera_id,
                bbox=bbox,
                last_seen=timestamp,
            )
        else:
            obj = self._objects[global_id]
            obj.camera_id = camera_id
            obj.bbox = bbox
            obj.last_seen = timestamp
            obj.history.append({
                "camera_id": camera_id,
                "bbox": bbox,
                "timestamp": timestamp,
            })
            # Trim history
            if len(obj.history) > self.max_history:
                obj.history = obj.history[-self.max_history:]

        self._prune_stale(timestamp)

    def get(self, global_id: str) -> GlobalObject | None:
        return self._objects.get(global_id)

    def all_active(self) -> list[GlobalObject]:
        return list(self._objects.values())

    def _prune_stale(self, current_ts: float):
        stale = [
            gid for gid, obj in self._objects.items()
            if current_ts - obj.last_seen > self.stale_timeout_s
        ]
        for gid in stale:
            del self._objects[gid]
