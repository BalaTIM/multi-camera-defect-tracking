"""
Edge Node Entrypoint
--------------------
Runs per camera. Captures frames, detects defects, extracts embeddings,
tracks locally, and publishes results to Kafka.

Usage:
    python edge/main.py --camera-id cam_01 --source rtsp://192.168.1.10/stream
    python edge/main.py --camera-id cam_02 --source 0   # webcam index
"""

import argparse
import time
import asyncio
from loguru import logger

from capture.camera_stream import CameraStream
from inference.yolo_detector import YOLODetector
from tracking.deepsort_tracker import DeepSORTTracker
from embedding.feature_extractor import FeatureExtractor
from sync.timestamp_sync import TimestampSync
from publisher.kafka_producer import KafkaDefectProducer
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Edge Node - Defect Detection")
    parser.add_argument("--camera-id", required=True, help="Unique camera identifier (e.g. cam_01)")
    parser.add_argument("--source", required=True, help="Video source: RTSP URL, file path, or webcam index")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--show", action="store_true", help="Display local preview window")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    logger.info(f"[{args.camera_id}] Initializing edge node...")

    # --- Initialize modules ---
    stream = CameraStream(source=args.source, fps=cfg["system"]["fps_target"])
    detector = YOLODetector(
        weights=cfg["detection"]["weights_path"],
        conf=cfg["detection"]["confidence_threshold"],
        iou=cfg["detection"]["iou_threshold"],
        device=cfg["detection"]["device"],
    )
    tracker = DeepSORTTracker(
        max_age=cfg["tracking"]["max_age"],
        n_init=cfg["tracking"]["n_init"],
        max_cosine_distance=cfg["tracking"]["max_cosine_distance"],
    )
    extractor = FeatureExtractor(embedding_dim=cfg["reid"]["embedding_dim"])
    sync = TimestampSync(camera_id=args.camera_id)
    producer = KafkaDefectProducer(
        bootstrap_servers=cfg["kafka"]["bootstrap_servers"],
        topic=cfg["kafka"]["topic"],
    )

    logger.success(f"[{args.camera_id}] Edge node ready. Processing stream: {args.source}")

    frame_count = 0
    for frame, raw_ts in stream.read():
        ts = sync.align(raw_ts)

        # Detection
        detections = detector.detect(frame)
        if not detections:
            continue

        # Embedding extraction
        crops = detector.crop_detections(frame, detections)
        embeddings = extractor.extract(crops)

        # Local tracking
        tracks = tracker.update(detections, embeddings, frame)

        # Publish to Kafka
        for track in tracks:
            producer.publish(
                camera_id=args.camera_id,
                timestamp=ts,
                track=track,
            )

        frame_count += 1
        if frame_count % 100 == 0:
            logger.info(f"[{args.camera_id}] Processed {frame_count} frames")

    producer.close()
    logger.info(f"[{args.camera_id}] Edge node shut down cleanly.")


if __name__ == "__main__":
    main()
