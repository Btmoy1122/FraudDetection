"""Experimental ML scoring endpoint — NOT part of the decision path.

The model is a logistic regression trained offline on the Kaggle credit
card dataset (see ml/train_model.py).  Its inputs are the dataset's
anonymised PCA columns V1–V28 plus Amount, which can't be derived from a
live {user_id, amount} transaction, so it's served on its own endpoint
until it's retrained on features the live system actually has.
"""

import json
import logging
from typing import Any

import joblib
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import ARTIFACTS_DIR

logger = logging.getLogger(__name__)

MODEL_PATH = ARTIFACTS_DIR / "model_v1.pkl"
MODEL_META_PATH = ARTIFACTS_DIR / "model_v1_meta.json"

router = APIRouter(prefix="/ml", tags=["ml (experimental)"])

ml_model = None
ml_metadata: dict[str, Any] = {}


class KaggleScoreRequest(BaseModel):
    features: dict[str, float]


def load_ml_artifacts() -> None:
    global ml_model, ml_metadata
    if not MODEL_PATH.exists() or not MODEL_META_PATH.exists():
        ml_model = None
        ml_metadata = {}
        logger.warning("ML model artifacts not found at %s", ARTIFACTS_DIR)
        return
    # The model is experimental: a bad or incompatible artifact must never
    # stop the API from starting.
    try:
        with MODEL_META_PATH.open("r", encoding="utf-8") as f:
            ml_metadata = json.load(f)
        ml_model = joblib.load(MODEL_PATH)
    except Exception:
        ml_model = None
        ml_metadata = {}
        logger.exception("Failed to load ML model from %s", ARTIFACTS_DIR)
        return
    logger.info("Loaded ML model %s", ml_metadata.get("model_version"))


def model_loaded() -> bool:
    return ml_model is not None


@router.post("/score-kaggle")
def score_kaggle_transaction(payload: KaggleScoreRequest) -> dict[str, Any]:
    if ml_model is None:
        raise HTTPException(status_code=503, detail="ML model artifacts are not loaded.")

    feature_cols = ml_metadata.get("feature_cols", [])
    threshold = float(ml_metadata.get("threshold", 0.5))

    missing = [col for col in feature_cols if col not in payload.features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing}")

    feature_vector = [[float(payload.features[col]) for col in feature_cols]]
    ml_score = float(ml_model.predict_proba(feature_vector)[0][1])

    return {
        "ml_score": ml_score,
        "ml_threshold": threshold,
        "ml_decision": "DENY" if ml_score >= threshold else "APPROVE",
        "model_version": ml_metadata.get("model_version", "unknown"),
    }
