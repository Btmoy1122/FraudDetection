"""Kafka consumer that maintains running fraud-detection metrics.

Demonstrates:
    - Idempotent counting via a seen-set (bounded to prevent memory leaks)
    - Independent consumer group ("analytics") receiving the full stream
    - Periodic metric summaries

In production you'd push these to Prometheus/Grafana instead of logging.
"""

import json
import logging
from collections import deque

from kafka import KafkaConsumer

from config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TRANSACTION_TOPIC

logger = logging.getLogger(__name__)

SEEN_MAX = 100_000
REPORT_INTERVAL = 10


def run() -> None:
    consumer = KafkaConsumer(
        KAFKA_TRANSACTION_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id="analytics",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )

    logger.info("analytics consumer started")

    total = 0
    denials = 0
    seen: deque[str] = deque(maxlen=SEEN_MAX)

    for message in consumer:
        event = message.value
        tid = event.get("transaction_id")

        if tid in seen:
            continue
        seen.append(tid)

        total += 1
        if event.get("decision") == "DENY":
            denials += 1

        if total % REPORT_INTERVAL == 0:
            rate = (denials / total * 100) if total else 0
            logger.info(
                "METRICS | total=%d denied=%d denial_rate=%.1f%%",
                total, denials, rate,
            )
