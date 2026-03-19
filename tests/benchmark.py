"""
Performance Benchmark
---------------------
Simulates N cameras streaming for a set duration and measures:
- Messages published per second
- Re-ID match latency
- Decision fusion latency
- End-to-end latency estimate

Usage:
    python tests/benchmark.py --cameras 4 --duration 60
"""

import argparse
import time
import uuid
import statistics
import numpy as np
from loguru import logger

from server.reid.reid_matcher import ReIDMatcher
from server.fusion.decision_engine import DecisionEngine
from server.tracking.global_tracker import GlobalTracker


def make_emb(dim: int = 256) -> list[float]:
    v = np.random.rand(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def run_benchmark(n_cameras: int, duration_s: float):
    reid = ReIDMatcher(threshold=0.75, embedding_dim=256, feature_ttl=30.0)
    fusion = DecisionEngine(temporal_window_ms=100, min_cameras=2)
    tracker = GlobalTracker()

    reid_latencies = []
    fusion_latencies = []
    total_messages = 0

    logger.info(f"Starting benchmark: {n_cameras} cameras, {duration_s}s duration")
    start = time.perf_counter()

    while time.perf_counter() - start < duration_s:
        for cam_idx in range(n_cameras):
            cam_id = f"cam_{cam_idx:02d}"
            local_id = np.random.randint(1, 10)
            emb = make_emb()
            ts = time.time()
            defect = np.random.random() < 0.3  # 30% defect rate

            # Re-ID
            t0 = time.perf_counter()
            gid = reid.match_or_register(cam_id, local_id, emb, ts)
            reid_latencies.append((time.perf_counter() - t0) * 1000)

            # Tracker update
            tracker.update(gid, cam_id, [0, 0, 100, 100], ts)

            # Fusion
            t1 = time.perf_counter()
            decision = fusion.decide(gid, cam_id, defect, ts)
            fusion_latencies.append((time.perf_counter() - t1) * 1000)

            total_messages += 1

    elapsed = time.perf_counter() - start
    msgs_per_sec = total_messages / elapsed

    print("\n" + "=" * 50)
    print(f"  BENCHMARK RESULTS — {n_cameras} cameras, {duration_s:.0f}s")
    print("=" * 50)
    print(f"  Total messages processed : {total_messages:,}")
    print(f"  Throughput               : {msgs_per_sec:.1f} msg/s")
    print(f"  Re-ID latency (median)   : {statistics.median(reid_latencies):.3f} ms")
    print(f"  Re-ID latency (p99)      : {np.percentile(reid_latencies, 99):.3f} ms")
    print(f"  Fusion latency (median)  : {statistics.median(fusion_latencies):.3f} ms")
    print(f"  Fusion latency (p99)     : {np.percentile(fusion_latencies, 99):.3f} ms")
    print(f"  Active global IDs        : {len(reid.active_ids)}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", type=int, default=4)
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()
    run_benchmark(n_cameras=args.cameras, duration_s=args.duration)
