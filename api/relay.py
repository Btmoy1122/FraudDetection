"""Outbox relay — publishes outbox rows to Kafka with at-least-once delivery.

Loop:
    1. SELECT a batch of unpublished rows FOR UPDATE SKIP LOCKED
    2. Send each to Kafka keyed by user_id, wait for acks=all
    3. Mark the batch published in the same DB transaction

If Kafka is down, step 2 raises, the DB transaction rolls back, and the
rows stay unpublished until the next attempt — nothing is lost.  If the
relay crashes after step 2 but before step 3 commits, the batch is sent
again on restart.  That duplicate is the "at-least" in at-least-once;
consumers are idempotent, so it's harmless.

Run exactly one relay.  Two relays would each grab different batches and
could publish one user's events out of order.
"""

import json
import logging
import time

from kafka import KafkaProducer
from kafka.errors import KafkaError
from prometheus_client import Counter, Gauge, start_http_server
from sqlalchemy import delete, func, literal_column, select, update

from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    RELAY_BATCH_SIZE,
    RELAY_METRICS_PORT,
    RELAY_POLL_INTERVAL_SECONDS,
)
from database import engine
from models import OutboxDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s — %(message)s")
logger = logging.getLogger("relay")

SEND_TIMEOUT_SECONDS = 10
CLEANUP_INTERVAL_SECONDS = 60
GAUGE_INTERVAL_SECONDS = 1

PUBLISHED = Counter("fraud_outbox_published_total", "Outbox events published to Kafka")
PUBLISH_FAILURES = Counter("fraud_outbox_publish_failures_total", "Failed publish batches")
PENDING = Gauge("fraud_outbox_pending", "Outbox events not yet published")
OLDEST_AGE = Gauge(
    "fraud_outbox_oldest_pending_age_seconds", "Age of the oldest unpublished event"
)


def make_producer() -> KafkaProducer:
    while True:
        try:
            return KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                # Wait for all in-sync replicas before a send counts as done.
                acks="all",
                retries=5,
                # With retries, >1 in-flight request could reorder messages.
                max_in_flight_requests_per_connection=1,
                linger_ms=5,
                key_serializer=lambda k: k.encode("utf-8"),
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
        except KafkaError:
            logger.warning("Kafka not reachable — retrying in 2s")
            time.sleep(2)


def publish_batch(producer: KafkaProducer) -> int:
    with engine.begin() as conn:
        rows = conn.execute(
            select(OutboxDB.id, OutboxDB.topic, OutboxDB.key, OutboxDB.payload)
            .where(OutboxDB.published_at.is_(None))
            .order_by(OutboxDB.id)
            .limit(RELAY_BATCH_SIZE)
            .with_for_update(skip_locked=True)
        ).all()
        if not rows:
            return 0

        futures = [producer.send(r.topic, key=r.key, value=r.payload) for r in rows]
        producer.flush()
        for f in futures:
            f.get(timeout=SEND_TIMEOUT_SECONDS)  # raises → rollback → retry

        conn.execute(
            update(OutboxDB)
            .where(OutboxDB.id.in_([r.id for r in rows]))
            .values(published_at=func.now())
        )
    return len(rows)


def update_gauges() -> None:
    with engine.connect() as conn:
        pending, oldest = conn.execute(
            select(
                func.count(OutboxDB.id),
                func.extract("epoch", func.now() - func.min(OutboxDB.created_at)),
            ).where(OutboxDB.published_at.is_(None))
        ).one()
    PENDING.set(pending)
    OLDEST_AGE.set(float(oldest or 0))


def cleanup_published() -> None:
    """Published rows are only kept briefly for debugging."""
    with engine.begin() as conn:
        conn.execute(
            delete(OutboxDB).where(
                OutboxDB.published_at < func.now() - literal_column("interval '1 hour'")
            )
        )


def main() -> None:
    start_http_server(RELAY_METRICS_PORT)
    producer = make_producer()
    logger.info("Outbox relay started")

    last_gauges = last_cleanup = 0.0
    while True:
        now = time.monotonic()
        try:
            if now - last_gauges >= GAUGE_INTERVAL_SECONDS:
                update_gauges()
                last_gauges = now
            if now - last_cleanup >= CLEANUP_INTERVAL_SECONDS:
                cleanup_published()
                last_cleanup = now

            sent = publish_batch(producer)
        except Exception:
            PUBLISH_FAILURES.inc()
            logger.exception("Publish failed — will retry")
            time.sleep(1)
            continue

        if sent:
            PUBLISHED.inc(sent)
        else:
            time.sleep(RELAY_POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
