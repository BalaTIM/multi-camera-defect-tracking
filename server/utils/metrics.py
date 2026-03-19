"""
Server Metrics
--------------
Lightweight in-process metrics counters.
Can be scraped by Prometheus if you add prometheus-fastapi-instrumentator.
"""

import time
from dataclasses import dataclass, field
from collections import deque
from threading import Lock


@dataclass
class LatencyTracker:
    """Rolling window latency tracker (thread-safe)."""
    window: int = 1000
    _samples: deque = field(default_factory=lambda: deque(maxlen=1000))
    _lock: Lock = field(default_factory=Lock)

    def record(self, latency_ms: float):
        with self._lock:
            self._samples.append(latency_ms)

    def p50(self) -> float:
        with self._lock:
            s = sorted(self._samples)
        return s[len(s) // 2] if s else 0.0

    def p99(self) -> float:
        with self._lock:
            s = sorted(self._samples)
        return s[int(len(s) * 0.99)] if s else 0.0

    def mean(self) -> float:
        with self._lock:
            s = list(self._samples)
        return sum(s) / len(s) if s else 0.0


class ServerMetrics:
    """Global singleton for server-side metrics."""

    def __init__(self):
        self.messages_received: int = 0
        self.defects_confirmed: int = 0
        self.defects_uncertain: int = 0
        self.reid_latency = LatencyTracker()
        self.fusion_latency = LatencyTracker()
        self._start_time = time.time()

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    @property
    def messages_per_second(self) -> float:
        elapsed = self.uptime_seconds
        return self.messages_received / elapsed if elapsed > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "uptime_s": round(self.uptime_seconds, 1),
            "messages_received": self.messages_received,
            "messages_per_second": round(self.messages_per_second, 2),
            "defects_confirmed": self.defects_confirmed,
            "defects_uncertain": self.defects_uncertain,
            "reid_latency_p50_ms": round(self.reid_latency.p50(), 3),
            "reid_latency_p99_ms": round(self.reid_latency.p99(), 3),
            "fusion_latency_p50_ms": round(self.fusion_latency.p50(), 3),
            "fusion_latency_p99_ms": round(self.fusion_latency.p99(), 3),
        }


# Module-level singleton
metrics = ServerMetrics()
