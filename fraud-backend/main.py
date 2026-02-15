import json
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError

from database import Sessionlocal, engine
from features import compute_features
from models import Base, TransactionDB
from rules import apply_rules

Base.metadata.create_all(bind=engine)

app = FastAPI()

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model_v1.pkl"
MODEL_META_PATH = ARTIFACTS_DIR / "model_v1_meta.json"

ml_model = None
ml_metadata: dict[str, Any] = {}


@app.on_event("startup")
def load_ml_artifacts():
    """Load model artifacts once at startup for low-latency inference."""
    global ml_model, ml_metadata

    if not MODEL_PATH.exists() or not MODEL_META_PATH.exists():
        ml_model = None
        ml_metadata = {}
        return

    with MODEL_META_PATH.open("r", encoding="utf-8") as f:
        ml_metadata = json.load(f)

    ml_model = joblib.load(MODEL_PATH)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "ml_model_loaded": ml_model is not None
    }


class Transaction(BaseModel):
    transaction_id: str
    user_id: str
    amount: float


class KaggleScoreRequest(BaseModel):
    features: dict[str, float]


def _score_kaggle_features(features: dict[str, float]) -> dict[str, Any]:
    if ml_model is None:
        raise HTTPException(
            status_code=503,
            detail="ML model artifacts are not loaded. Train and save artifacts first."
        )

    feature_cols = ml_metadata.get("feature_cols", [])
    threshold = float(ml_metadata.get("threshold", 0.5))
    model_version = ml_metadata.get("model_version", "unknown")

    missing = [col for col in feature_cols if col not in features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required features: {missing}"
        )

    feature_vector = [[float(features[col]) for col in feature_cols]]
    ml_score = float(ml_model.predict_proba(feature_vector)[0][1])
    ml_decision = "DENY" if ml_score >= threshold else "APPROVE"

    return {
        "ml_score": ml_score,
        "ml_threshold": threshold,
        "ml_decision": ml_decision,
        "model_version": model_version
    }


@app.post("/ml/score-kaggle")
def score_kaggle_transaction(payload: KaggleScoreRequest):
    """Score Kaggle-style features (V1..V28 + Amount) with the trained model."""
    return _score_kaggle_features(payload.features)


@app.post("/transactions")
def create_transaction(txn: Transaction):
    db = Sessionlocal()

    try:
        # 1) compute features
        features = compute_features(txn.user_id, db)

        # 2) apply rules
        decision, reason = apply_rules(features, txn.amount)

        # 3) insert row
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

    except IntegrityError:
        db.rollback()
        existing = db.query(TransactionDB).filter(
            TransactionDB.transaction_id == txn.transaction_id
        ).first()
        if existing:
            return {
                "transaction_id": existing.transaction_id,
                "decision": existing.decision,
                "reason": existing.reason,
                "features": existing.features,
                "idempotent": True
            }
        raise
    finally:
        db.close()

    return {
        "transaction_id": txn.transaction_id,
        "decision": decision,
        "reason": reason,
        "features": features,
        "idempotent": False
    }


@app.get("/transactions/{transaction_id}")
def get_transaction(transaction_id: str):
    db = Sessionlocal()

    txn = db.query(TransactionDB).filter(
        TransactionDB.transaction_id == transaction_id
    ).first()

    db.close()

    if not txn:
        raise HTTPException(
            status_code=404,
            detail="Transaction not found"
        )

    return {
        "transaction_id": txn.transaction_id,
        "user_id": txn.user_id,
        "amount": txn.amount,
        "decision": txn.decision
    }
