from sqlalchemy import Column, String, Float, DateTime
from database import Base
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB


class TransactionDB(Base):
    __tablename__ = "transactions"

    transaction_id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    amount = Column(Float)
    decision = Column(String)
    reason = Column(String)
    features = Column(JSONB)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

