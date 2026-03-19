"""
Kafka Producer
--------------
Publishes detection + tracking results from an edge node to Kafka.
Each message contains all fields needed for cross-camera Re-ID.
"""

import json
import time
from kafka import KafkaProducer
from kafka.errors import KafkaError
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from tracking.deepsort_tracker import Track


class KafkaDefectProducer:
    def __init__(self, bootstrap_servers: str, topic: str):
        self.topic = topic
        self._producer = self._connect(bootstrap_servers)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _connect(self, bootstrap_servers: str) -> KafkaProducer:
        logger.info(f"Connecting to Kafka at {bootstrap_servers}...")
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8"),
            acks="all",
            retries=3,
            linger_ms=5,
        )
        logger.success("Kafka producer connected.")
        return producer

    def publish(self, camera_id: str, timestamp: float, track: Track):
        """
        Publish a single track event to Kafka.

        Message schema:
            camera_id   (str)
            timestamp   (float) — NTP-aligned epoch seconds
            object_id   (int)   — local track ID
            bbox        ([x1,y1,x2,y2])
            confidence  (float)
            label       (str)
            embedding   ([float, ...])
            defect_flag (bool)
        """
        message = {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "object_id": track.track_id,
            "bbox": track.bbox,
            "confidence": round(track.confidence, 4),
            "label": track.label,
            "embedding": [round(v, 6) for v in track.embedding],
            "defect_flag": track.is_defect,
        }

        future = self._producer.send(
            topic=self.topic,
            key=camera_id,
            value=message,
        )
        try:
            future.get(timeout=2)
        except KafkaError as e:
            logger.error(f"Kafka publish error for {camera_id}: {e}")

    def close(self):
        self._producer.flush()
        self._producer.close()
        logger.info("Kafka producer closed.")
