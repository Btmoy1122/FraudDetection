"""Shared consumer loop: manual offset commits, retries, dead-letter topic.

Delivery guarantee — at-least-once:
    Offsets are committed only AFTER every message in a polled batch has
    been handled (or dead-lettered).  If the process crashes mid-batch, the
    uncommitted messages are redelivered on restart.  Handlers must
    therefore be idempotent.

Failure handling, per message:
    - Not valid JSON / not an object    → DLQ immediately (poison message)
    - Handler raises NonRetryableError  → DLQ immediately (bad data)
    - Handler raises a transient error  → retry forever with capped backoff
      (e.g. Postgres down: the message is fine, the dependency isn't, so
      dead-lettering would just dump every message into the DLQ)
    - Handler raises anything else      → retry MAX_ATTEMPTS, then DLQ

A message is only skipped after it's safely written to the DLQ (acks=all).
If the DLQ write itself fails the exception propagates, the consumer dies
without committing, and the message is redelivered after restart.
"""

import json
import logging
import time
from typing import Any, Callable

from kafka import KafkaConsumer, KafkaProducer
from prometheus_client import Counter

from settings import KAFKA_BOOTSTRAP_SERVERS, KAFKA_DLQ_TOPIC, KAFKA_TRANSACTION_TOPIC

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 0.5
BACKOFF_MAX_SECONDS = 10
DLQ_SEND_TIMEOUT_SECONDS = 10

EVENTS = Counter(
    "fraud_consumer_events_total", "Messages handled by consumers", ["consumer", "outcome"]
)

Handler = Callable[[dict, Any], None]
DlqPublisher = Callable[[Any, str, str], None]


class NonRetryableError(Exception):
    """The message itself is bad; retrying can't help."""


def _backoff(attempt: int) -> float:
    return min(BACKOFF_BASE_SECONDS * 2 ** (attempt - 1), BACKOFF_MAX_SECONDS)


def process_message(
    message: Any,
    handler: Handler,
    dlq_publish: DlqPublisher,
    consumer_name: str,
    transient_errors: tuple[type[Exception], ...] = (),
    max_attempts: int = MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Handle one message. Returns "ok" or "dlq"."""
    try:
        event = json.loads(message.value)
        if not isinstance(event, dict):
            raise ValueError("event is not a JSON object")
    except ValueError as exc:  # includes JSONDecodeError and UnicodeDecodeError
        dlq_publish(message, f"deserialize: {exc}", consumer_name)
        return "dlq"

    attempt = 0
    while True:
        attempt += 1
        try:
            handler(event, message)
            return "ok"
        except NonRetryableError as exc:
            dlq_publish(message, f"non-retryable: {exc}", consumer_name)
            return "dlq"
        except transient_errors as exc:
            logger.warning("%s: transient error (attempt %d): %s", consumer_name, attempt, exc)
            sleep(_backoff(attempt))
        except Exception as exc:
            if attempt >= max_attempts:
                logger.error("%s: giving up after %d attempts: %r", consumer_name, attempt, exc)
                dlq_publish(message, f"failed after {attempt} attempts: {exc!r}", consumer_name)
                return "dlq"
            logger.warning("%s: error (attempt %d): %r", consumer_name, attempt, exc)
            sleep(_backoff(attempt))


def make_dlq_publisher(producer: KafkaProducer) -> DlqPublisher:
    def publish(message: Any, error: str, consumer_name: str) -> None:
        headers = [
            ("dlq.error", error[:1000].encode("utf-8")),
            ("dlq.consumer", consumer_name.encode("utf-8")),
            ("dlq.source.topic", message.topic.encode("utf-8")),
            ("dlq.source.partition", str(message.partition).encode("utf-8")),
            ("dlq.source.offset", str(message.offset).encode("utf-8")),
        ]
        # Forward the original bytes untouched so the message can be replayed.
        producer.send(
            KAFKA_DLQ_TOPIC, key=message.key, value=message.value, headers=headers
        ).get(timeout=DLQ_SEND_TIMEOUT_SECONDS)
        logger.error("%s: sent offset %s to DLQ: %s", consumer_name, message.offset, error)

    return publish


def run_consumer(
    name: str,
    group_id: str,
    handler: Handler,
    transient_errors: tuple[type[Exception], ...] = (),
) -> None:
    consumer = KafkaConsumer(
        KAFKA_TRANSACTION_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=group_id,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
    )
    dlq_publish = make_dlq_publisher(
        KafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS, acks="all")
    )
    logger.info("%s consumer started (group=%s)", name, group_id)

    while True:
        batch = consumer.poll(timeout_ms=1000, max_records=500)
        if not batch:
            continue
        for messages in batch.values():
            for message in messages:
                outcome = process_message(
                    message, handler, dlq_publish, name, transient_errors
                )
                EVENTS.labels(consumer=name, outcome=outcome).inc()
        consumer.commit()
