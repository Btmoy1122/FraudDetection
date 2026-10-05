"""Fraud Detection API — FastAPI service.

Transaction flow (POST /transactions):
    1. Idempotency check: a known transaction_id returns its stored decision
    2. Features: atomic Redis sliding window + cached 30-day average
       (falls back to Postgres if Redis is down)
    3. Rule engine → APPROVE / DENY
    4. ONE Postgres transaction writes the decision AND an outbox event
    5. Return the decision

The API never talks to Kafka.  relay.py publishes outbox rows to Kafka
asynchronously, so a Kafka outage can't slow down or fail a decision, and
an event can't be lost between "saved" and "published".
"""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest, multiprocess
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import ml_router
from config import KAFKA_TRANSACTION_TOPIC, PROMETHEUS_MULTIPROC_DIR
from database import get_db
from events import TransactionEvent
from feature_store import forget_transaction, get_features
from features import compute_avg_30d_from_db, compute_features_from_db
from metrics import DECISION_LATENCY, DECISIONS, IDEMPOTENT_REPLAYS
from models import OutboxDB, TransactionDB
from redis_client import close_pool
from rules import apply_rules

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_router.load_ml_artifacts()
    logger.info("Application started")
    yield
    close_pool()
    logger.info("Application shut down")


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)
app.include_router(ml_router.router)


class TransactionRequest(BaseModel):
    transaction_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    amount: float = Field(gt=0)


def _stored_response(txn: TransactionDB) -> dict:
    IDEMPOTENT_REPLAYS.inc()
    return {
        "transaction_id": txn.transaction_id,
        "decision": txn.decision,
        "reason": txn.reason,
        "features": txn.features,
        "idempotent": True,
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "ml_model_loaded": ml_router.model_loaded()}


@app.get("/metrics")
def metrics():
    # With several uvicorn workers each process has its own counters, so a
    # scrape would only see whichever worker answered.  In multiprocess mode
    # every worker writes to files in PROMETHEUS_MULTIPROC_DIR and this
    # aggregates them.
    if PROMETHEUS_MULTIPROC_DIR:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/transactions")
def create_transaction(txn: TransactionRequest, db: Session = Depends(get_db)):
    with DECISION_LATENCY.time():
        # Fast path for client retries.  Concurrent duplicates that both get
        # past this check are caught by the primary key below.
        existing = db.get(TransactionDB, txn.transaction_id)
        if existing:
            return _stored_response(existing)

        features = get_features(
            txn.user_id,
            txn.transaction_id,
            txn.amount,
            db_features=lambda uid: compute_features_from_db(uid, db),
            db_avg=lambda uid: compute_avg_30d_from_db(uid, db),
        )
        decision, reason = apply_rules(features, txn.amount)

        event = TransactionEvent(
            transaction_id=txn.transaction_id,
            user_id=txn.user_id,
            amount=txn.amount,
            decision=decision,
            reason=reason,
            features=features,
        )
        db.add(TransactionDB(
            transaction_id=txn.transaction_id,
            user_id=txn.user_id,
            amount=txn.amount,
            decision=decision,
            reason=reason,
            features=features,
        ))
        db.add(OutboxDB(
            topic=KAFKA_TRANSACTION_TOPIC,
            key=txn.user_id,
            payload=event.model_dump(),
        ))

        try:
            db.commit()
        except IntegrityError:
            # Lost a race with a concurrent request for the same
            # transaction_id — answer with the winner's decision.
            db.rollback()
            existing = db.get(TransactionDB, txn.transaction_id)
            if existing is None:
                raise
            return _stored_response(existing)
        except Exception:
            db.rollback()
            forget_transaction(txn.user_id, txn.transaction_id, txn.amount)
            raise

        DECISIONS.labels(decision=decision, reason=reason).inc()
        return {
            "transaction_id": txn.transaction_id,
            "decision": decision,
            "reason": reason,
            "features": features,
            "idempotent": False,
        }


@app.get("/transactions/{transaction_id}")
def get_transaction(transaction_id: str, db: Session = Depends(get_db)):
    txn = db.get(TransactionDB, transaction_id)
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")

    return {
        "transaction_id": txn.transaction_id,
        "user_id": txn.user_id,
        "amount": txn.amount,
        "decision": txn.decision,
        "reason": txn.reason,
    }
