"""
Camera Simulator
----------------
Simulates N cameras publishing fake detection events to Kafka.
Use this to test the full pipeline without real hardware.

Usage:
    python scripts/simulate_cameras.py --cameras 4 --fps 10 --defect-rate 0.25
"""

import argparse
import time
import json
import threading
import numpy as np
from kafka import KafkaProducer
from loguru import logger


def make_embedding(dim: int = 256, seed: int | None = None) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


DEFECT_LABELS = ["crack", "scratch", "dent", "corrosion", "hole"]
OK_LABELS = ["surface_ok", "edge_ok", "weld_ok"]


def simulate_camera(
    camera_id: str,
    producer: KafkaProducer,
    topic: str,
    fps: float,
    defect_rate: float,
    n_objects: int = 5,
    duration: float | None = None,
):
    """Simulate one camera publishing detection events."""
    interval = 1.0 / fps
    start = time.time()
    frame = 0

    # Stable embeddings per "object" so Re-ID can match them
    object_embeddings = {i: make_embedding(seed=i * 1000) for i in range(n_objects)}

    logger.info(f"[{camera_id}] Starting simulation at {fps} fps, defect_rate={defect_rate}")

    while True:
        if duration and (time.time() - start) > duration:
            break

        # Simulate 1–3 visible objects per frame
        n_visible = np.random.randint(1, min(4, n_objects + 1))
        visible_ids = np.random.choice(list(range(n_objects)), size=n_visible, replace=False)

        for obj_id in visible_ids:
            is_defect = np.random.random() < defect_rate
            label = np.random.choice(DEFECT_LABELS) if is_defect else np.random.choice(OK_LABELS)

            # Slightly perturb embedding to simulate viewpoint change
            base = np.array(object_embeddings[obj_id])
            noise = np.random.randn(256).astype(np.float32) * 0.05
            emb = base + noise
            emb = (emb / np.linalg.norm(emb)).tolist()

            x1 = int(np.random.uniform(0, 500))
            y1 = int(np.random.uniform(0, 400))
            x2 = x1 + int(np.random.uniform(50, 150))
            y2 = y1 + int(np.random.uniform(50, 150))

            message = {
                "camera_id": camera_id,
                "timestamp": time.time(),
                "object_id": int(obj_id),
                "bbox": [x1, y1, x2, y2],
                "confidence": round(float(np.random.uniform(0.6, 0.99)), 3),
                "label": label,
                "embedding": [round(v, 6) for v in emb],
                "defect_flag": is_defect,
            }

            producer.send(
                topic=topic,
                key=camera_id.encode(),
                value=json.dumps(message).encode(),
            )

        frame += 1
        if frame % (fps * 10) == 0:
            elapsed = time.time() - start
            logger.info(f"[{camera_id}] {frame} frames in {elapsed:.0f}s")

        time.sleep(interval)

    logger.info(f"[{camera_id}] Simulation complete.")


def main():
    parser = argparse.ArgumentParser(description="Camera simulator for pipeline testing")
    parser.add_argument("--cameras", type=int, default=2, help="Number of simulated cameras")
    parser.add_argument("--fps", type=float, default=10.0, help="Frames per second per camera")
    parser.add_argument("--defect-rate", type=float, default=0.25, help="Fraction of defect detections")
    parser.add_argument("--duration", type=float, default=None, help="Run duration in seconds (default: forever)")
    parser.add_argument("--kafka", default="localhost:9092", help="Kafka bootstrap servers")
    parser.add_argument("--topic", default="defect_tracking", help="Kafka topic name")
    args = parser.parse_args()

    producer = KafkaProducer(bootstrap_servers=args.kafka)
    logger.info(f"Simulating {args.cameras} cameras → Kafka {args.kafka}/{args.topic}")

    threads = []
    for i in range(args.cameras):
        cam_id = f"sim_cam_{i:02d}"
        t = threading.Thread(
            target=simulate_camera,
            args=(cam_id, producer, args.topic, args.fps, args.defect_rate),
            kwargs={"duration": args.duration},
            daemon=True,
        )
        threads.append(t)
        t.start()

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        logger.info("Simulation stopped.")
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()
