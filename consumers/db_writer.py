"""Kafka consumer that persists transaction events to PostgreSQL.

This is an *idempotent consumer*: if it receives the same event twice
(which Kafka's at-least-once delivery can cause), the INSERT will fail on
the unique transaction_id and the duplicate is silently skipped.

Interview takeaway — "at-least-once + idempotent consumer = effectively
exactly-once processing."
"""

import json
import logging

from kafka import KafkaConsumer
from sqlalchemy import create_engine, text

from config import DATABASE_URL, KAFKA_BOOTSTRAP_SERVERS, KAFKA_TRANSACTION_TOPIC

logger = logging.getLogger(__name__)

INSERT_SQL = text("""
    INSERT INTO transactions (transaction_id, user_id, amount, decision, reason, features, created_at)
    VALUES (:tid, :uid, :amount, :decision, :reason, CAST(:features AS jsonb), NOW())
    ON CONFLICT (transaction_id) DO NOTHING
""")


def run() -> None:
    engine = create_engine(DATABASE_URL)
    consumer = KafkaConsumer(
        KAFKA_TRANSACTION_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id="db-writers",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )

    logger.info("db_writer consumer started")

    for message in consumer:
        event = message.value
        try:
            with engine.connect() as conn:
                conn.execute(INSERT_SQL, {
                    "tid": event["transaction_id"],
                    "uid": event["user_id"],
                    "amount": event["amount"],
                    "decision": event["decision"],
                    "reason": event["reason"],
                    "features": json.dumps(event.get("features", {})),
                })
                conn.commit()
            logger.info("Persisted %s", event["transaction_id"])
        except Exception:
            logger.exception("Failed to persist %s", event.get("transaction_id"))
