from fastapi import FastAPI
from fastapi import HTTPException
from database import engine
from models import Base
from database import Sessionlocal
from models import TransactionDB
from sqlalchemy.exc import IntegrityError

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
    decision = "DENY" if txn.amount > 1000 else "APPROVE"

    db = Sessionlocal()

    try:
        db_txn = TransactionDB(
            transaction_id=txn.transaction_id,
            user_id=txn.user_id,
            amount=txn.amount,
            decision=decision)
        
        db.add(db_txn)
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = db.query(TransactionDB).filter(
            TransactionDB.transaction_id == txn.transaction_id
        ).first()
        if existing:
            return {
                "transaction_id": txn.transaction_id,
                "decision": existing.decision,
                "idempotent": True
            }
    finally:
        db.close()
    
    return {
        "transaction_id": txn.transaction_id,
        "decision": decision,
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
