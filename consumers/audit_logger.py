"""Kafka consumer that writes a structured audit trail.

Every transaction decision is logged with its full context.  In a
production system this might write to an append-only audit table, S3,
or a SIEM.  Here we write structured JSON to stdout so you can see
the events flowing through the system.

This consumer is in its own consumer group ("audit"), so it receives
every message independently of the db-writer or analytics consumers.
"""

import json
import logging

from kafka import KafkaConsumer

from config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TRANSACTION_TOPIC

logger = logging.getLogger(__name__)


def run() -> None:
    consumer = KafkaConsumer(
        KAFKA_TRANSACTION_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id="audit",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )

    logger.info("audit_logger consumer started")

    for message in consumer:
        event = message.value
        audit_record = {
            "audit_type": "transaction_decision",
            "transaction_id": event.get("transaction_id"),
            "user_id": event.get("user_id"),
            "amount": event.get("amount"),
            "decision": event.get("decision"),
            "reason": event.get("reason"),
            "timestamp": event.get("timestamp"),
            "kafka_partition": message.partition,
            "kafka_offset": message.offset,
        }
        logger.info("AUDIT | %s", json.dumps(audit_record))
