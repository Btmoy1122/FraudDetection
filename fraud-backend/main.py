from fastapi import FastAPI
from database import engine
from models import Base
from database import Sessionlocal
from models import TransactionDB

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

    db_txn = TransactionDB(
        transaction_id=txn.transaction_id,
        user_id=txn.user_id,
        amount=txn.amount,
        decision=decision
    )

    db.add(db_txn)
    db.commit()
    db.close()

    return {
        "transaction_id": txn.transaction_id,
        "decision": decision
    }