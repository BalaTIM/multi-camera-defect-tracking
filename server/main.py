"""
Central Server Entrypoint
--------------------------
Consumes Kafka messages from all edge nodes, performs cross-camera Re-ID,
global tracking, and decision fusion. Writes results to PostgreSQL.
"""

import asyncio
from loguru import logger

from consumer.kafka_consumer import KafkaDefectConsumer
from reid.reid_matcher import ReIDMatcher
from tracking.global_tracker import GlobalTracker
from fusion.decision_engine import DecisionEngine
from storage.database import Database
from utils.config import load_config


async def process_messages(
    consumer: KafkaDefectConsumer,
    reid: ReIDMatcher,
    tracker: GlobalTracker,
    fusion: DecisionEngine,
    db: Database,
):
    async for message in consumer.stream():
        camera_id = message["camera_id"]
        timestamp = message["timestamp"]
        local_id = message["object_id"]
        embedding = message["embedding"]
        bbox = message["bbox"]
        label = message["label"]
        confidence = message["confidence"]
        defect_flag = message["defect_flag"]

        # --- Step 1: Cross-camera Re-ID ---
        global_id = reid.match_or_register(
            camera_id=camera_id,
            local_id=local_id,
            embedding=embedding,
            timestamp=timestamp,
        )

        # --- Step 2: Update global tracker ---
        tracker.update(
            global_id=global_id,
            camera_id=camera_id,
            bbox=bbox,
            timestamp=timestamp,
        )

        # --- Step 3: Decision fusion ---
        decision = fusion.decide(
            global_id=global_id,
            camera_id=camera_id,
            defect_flag=defect_flag,
            timestamp=timestamp,
        )

        if decision in ("DEFECT_CONFIRMED", "UNCERTAIN"):
            await db.insert_defect(
                global_id=global_id,
                camera_id=camera_id,
                label=label,
                confidence=confidence,
                bbox=bbox,
                decision=decision,
                timestamp=timestamp,
            )
            logger.warning(f"[{camera_id}] Global ID {global_id} → {decision}")


async def main():
    cfg = load_config("configs/config.yaml")
    logger.info("Starting central defect tracking server...")

    db = Database(url=cfg["database"]["url"])
    await db.connect()

    consumer = KafkaDefectConsumer(
        bootstrap_servers=cfg["kafka"]["bootstrap_servers"],
        topic=cfg["kafka"]["topic"],
        group_id=cfg["kafka"]["group_id"],
    )
    reid = ReIDMatcher(
        threshold=cfg["reid"]["similarity_threshold"],
        embedding_dim=cfg["reid"]["embedding_dim"],
        feature_ttl=cfg["reid"]["feature_ttl_seconds"],
    )
    tracker = GlobalTracker()
    fusion = DecisionEngine(
        temporal_window_ms=cfg["fusion"]["temporal_window_ms"],
        min_cameras=cfg["fusion"]["min_cameras_for_confirmation"],
    )

    try:
        await process_messages(consumer, reid, tracker, fusion, db)
    finally:
        consumer.close()
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
