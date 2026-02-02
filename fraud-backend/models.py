from sqlalchemy import Column, String, Float
from database import Base

class TransactionDB(Base):
    __tablename__ = "transactions"

    transaction_id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    amount = Column(Float)
    decision = Column(String)
