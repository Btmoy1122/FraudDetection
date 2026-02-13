from fastapi import FastAPI
from fastapi import HTTPException
from database import engine
from models import Base
from database import Sessionlocal
from models import TransactionDB
from sqlalchemy.exc import IntegrityError
from features import compute_features
from rules import apply_rules

Base.metadata.create_all(bind=engine)



app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

from pydantic import BaseModel

class Transaction(BaseModel):
    transaction_id: str
    user_id: str
    amount: float



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
        "reason": reason,       # optional for learning/debugging
        "features": features,   # optional for learning/debugging
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
