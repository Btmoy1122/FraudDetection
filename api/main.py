"""Fraud Detection API — FastAPI service.

Transaction flow:
    1. Read features from Redis (falls back to PostgreSQL)
    2. Apply rule engine
    3. Persist to PostgreSQL (durable write)
    4. Update Redis counters (so next request sees fresh data)
    5. Publish event to Kafka (fire-and-forget for async consumers)
    6. Return decision

The DB write stays synchronous for reliability.  Redis accelerates
the read path; Kafka decouples downstream processing (audit, analytics)
from the user-facing response.
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError

from config import ARTIFACTS_DIR
from database import SessionLocal, engine
from events import TransactionEvent
from feature_store import read_features, update_features
from features import compute_features_from_db
from kafka_producer import close as close_kafka
from kafka_producer import flush as flush_kafka
from kafka_producer import publish_transaction_event
from models import Base, TransactionDB
from redis_client import close_pool
from rules import apply_rules

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base.metadata.create_all(bind=engine)

MODEL_PATH = ARTIFACTS_DIR / "model_v1.pkl"
MODEL_META_PATH = ARTIFACTS_DIR / "model_v1_meta.json"

ml_model = None
ml_metadata: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_ml_artifacts()
    logger.info("Application started")
    yield
    flush_kafka()
    close_kafka()
    close_pool()
    logger.info("Application shut down")


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)


def load_ml_artifacts() -> None:
    global ml_model, ml_metadata
    if not MODEL_PATH.exists() or not MODEL_META_PATH.exists():
        ml_model = None
        ml_metadata = {}
        logger.warning("ML model artifacts not found at %s", ARTIFACTS_DIR)
        return
    with MODEL_META_PATH.open("r", encoding="utf-8") as f:
        ml_metadata = json.load(f)
    ml_model = joblib.load(MODEL_PATH)
    logger.info("Loaded ML model %s", ml_metadata.get("model_version"))


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class TransactionRequest(BaseModel):
    transaction_id: str
    user_id: str
    amount: float


class KaggleScoreRequest(BaseModel):
    features: dict[str, float]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok", "ml_model_loaded": ml_model is not None}


@app.post("/ml/score-kaggle")
def score_kaggle_transaction(payload: KaggleScoreRequest):
    return _score_kaggle_features(payload.features)


@app.post("/transactions")
def create_transaction(txn: TransactionRequest):
    db = SessionLocal()
    try:
        def db_fallback(uid):
            return compute_features_from_db(uid, db)

        features = read_features(txn.user_id, db_fallback=db_fallback)

        decision, reason = apply_rules(features, txn.amount)

        db_txn = TransactionDB(
            transaction_id=txn.transaction_id,
            user_id=txn.user_id,
            amount=txn.amount,
            decision=decision,
            reason=reason,
            features=features,
        )
        db.add(db_txn)
        db.commit()

        update_features(txn.user_id, txn.transaction_id, txn.amount)

        event = TransactionEvent.from_transaction(
            transaction_id=txn.transaction_id,
            user_id=txn.user_id,
            amount=txn.amount,
            decision=decision,
            reason=reason,
            features=features,
        )
        publish_transaction_event(event.model_dump())

    except IntegrityError:
        db.rollback()
        existing = (
            db.query(TransactionDB)
            .filter(TransactionDB.transaction_id == txn.transaction_id)
            .first()
        )
        if existing:
            return {
                "transaction_id": existing.transaction_id,
                "decision": existing.decision,
                "reason": existing.reason,
                "features": existing.features,
                "idempotent": True,
            }
        raise
    finally:
        db.close()

    return {
        "transaction_id": txn.transaction_id,
        "decision": decision,
        "reason": reason,
        "features": features,
        "idempotent": False,
    }


@app.get("/transactions/{transaction_id}")
def get_transaction(transaction_id: str):
    db = SessionLocal()
    txn = (
        db.query(TransactionDB)
        .filter(TransactionDB.transaction_id == transaction_id)
        .first()
    )
    db.close()

    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")

    return {
        "transaction_id": txn.transaction_id,
        "user_id": txn.user_id,
        "amount": txn.amount,
        "decision": txn.decision,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _score_kaggle_features(features: dict[str, float]) -> dict[str, Any]:
    if ml_model is None:
        raise HTTPException(
            status_code=503,
            detail="ML model artifacts are not loaded.",
        )

    feature_cols = ml_metadata.get("feature_cols", [])
    threshold = float(ml_metadata.get("threshold", 0.5))
    model_version = ml_metadata.get("model_version", "unknown")

    missing = [col for col in feature_cols if col not in features]
    if missing:
        raise HTTPException(
            status_code=422, detail=f"Missing features: {missing}"
        )

    feature_vector = [[float(features[col]) for col in feature_cols]]
    ml_score = float(ml_model.predict_proba(feature_vector)[0][1])
    ml_decision = "DENY" if ml_score >= threshold else "APPROVE"

    return {
        "ml_score": ml_score,
        "ml_threshold": threshold,
        "ml_decision": ml_decision,
        "model_version": model_version,
    }
