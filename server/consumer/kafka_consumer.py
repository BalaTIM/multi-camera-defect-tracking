"""
Kafka Consumer
--------------
Async generator that streams messages from the defect tracking topic.
"""

import json
import asyncio
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class KafkaDefectConsumer:
    def __init__(self, bootstrap_servers: str, topic: str, group_id: str):
        self._consumer = self._connect(bootstrap_servers, topic, group_id)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=15))
    def _connect(self, bootstrap_servers, topic, group_id) -> KafkaConsumer:
        logger.info(f"Connecting consumer to Kafka topic: {topic}")
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )
        logger.success("Kafka consumer connected.")
        return consumer

    async def stream(self):
        """Async generator yielding deserialized message dicts."""
        loop = asyncio.get_event_loop()
        while True:
            # poll in executor to avoid blocking the event loop
            records = await loop.run_in_executor(
                None,
                lambda: self._consumer.poll(timeout_ms=100, max_records=50),
            )
            for _, messages in records.items():
                for msg in messages:
                    yield msg.value

    def close(self):
        self._consumer.close()
        logger.info("Kafka consumer closed.")
