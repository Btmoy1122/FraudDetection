"""Fire-and-forget Kafka producer for transaction events.

Design choices:
    - Lazy initialization: the producer is created on first publish, not
      at import time.  This avoids blocking app startup if Kafka is slow.
    - Key = user_id: Kafka partitions by key, so all events for a user
      land on the same partition.  This preserves per-user ordering, which
      matters when consumers rebuild user state.
    - Fire-and-forget: we don't wait for broker acknowledgement on the hot
      path.  If a publish fails we log and move on — the DB write is the
      durable fallback.  In a fully async architecture you'd use acks=all.
"""

import json
import logging

from kafka import KafkaProducer

from config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TRANSACTION_TOPIC

logger = logging.getLogger(__name__)

_producer: KafkaProducer | None = None


def get_producer() -> KafkaProducer | None:
    global _producer
    if _producer is not None:
        return _producer
    try:
        _producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
        )
        logger.info("Kafka producer connected")
    except Exception:
        logger.warning("Kafka unavailable — events will not be published")
    return _producer


def publish_transaction_event(event: dict) -> None:
    producer = get_producer()
    if producer is None:
        return
    try:
        producer.send(
            KAFKA_TRANSACTION_TOPIC,
            key=event.get("user_id"),
            value=event,
        )
    except Exception:
        logger.warning("Failed to publish event to Kafka", exc_info=True)


def flush() -> None:
    if _producer is not None:
        _producer.flush()


def close() -> None:
    global _producer
    if _producer is not None:
        _producer.close()
        _producer = None
