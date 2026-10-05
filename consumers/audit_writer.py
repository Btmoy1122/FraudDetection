"""Kafka consumer that writes an append-only audit trail to Postgres.

Why a separate table from `transactions`: the transactions table is
operational state that can change later (manual review, chargebacks).
The audit log records what was decided at the time, with the Kafka
partition/offset it came from, and is never updated.  In production this
would usually live in a separate store (S3, a warehouse, a SIEM).

Idempotent: ON CONFLICT (transaction_id) DO NOTHING means a redelivered
event (at-least-once) is a no-op.
"""

import json
import logging

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from runner import NonRetryableError, run_consumer
from settings import DATABASE_URL

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ("event_id", "transaction_id", "user_id", "amount", "decision", "reason",
                   "timestamp")

CREATE_TABLE_SQL = text("""
    CREATE TABLE IF NOT EXISTS audit_log (
        transaction_id  TEXT PRIMARY KEY,
        event_id        TEXT NOT NULL,
        user_id         TEXT NOT NULL,
        amount          DOUBLE PRECISION NOT NULL,
        decision        TEXT NOT NULL,
        reason          TEXT NOT NULL,
        features        JSONB,
        decided_at      TIMESTAMPTZ NOT NULL,
        kafka_partition INT NOT NULL,
        kafka_offset    BIGINT NOT NULL,
        recorded_at     TIMESTAMPTZ NOT NULL DEFAULT now()
    )
""")

INSERT_SQL = text("""
    INSERT INTO audit_log (transaction_id, event_id, user_id, amount, decision, reason,
                           features, decided_at, kafka_partition, kafka_offset)
    VALUES (:tid, :eid, :uid, :amount, :decision, :reason,
            CAST(:features AS jsonb), :decided_at, :partition, :offset)
    ON CONFLICT (transaction_id) DO NOTHING
""")


def make_handler(engine):
    def handle(event: dict, message) -> None:
        missing = [f for f in REQUIRED_FIELDS if f not in event]
        if missing:
            raise NonRetryableError(f"missing fields {missing}")

        with engine.begin() as conn:
            conn.execute(INSERT_SQL, {
                "tid": event["transaction_id"],
                "eid": event["event_id"],
                "uid": event["user_id"],
                "amount": event["amount"],
                "decision": event["decision"],
                "reason": event["reason"],
                "features": json.dumps(event.get("features", {})),
                "decided_at": event["timestamp"],
                "partition": message.partition,
                "offset": message.offset,
            })

    return handle


def run() -> None:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    with engine.begin() as conn:
        conn.execute(CREATE_TABLE_SQL)
    run_consumer(
        "audit-writer",
        group_id="audit",
        handler=make_handler(engine),
        transient_errors=(OperationalError,),
    )
