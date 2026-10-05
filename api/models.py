from sqlalchemy import BigInteger, Column, DateTime, Float, Index, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from database import Base


class TransactionDB(Base):
    __tablename__ = "transactions"

    transaction_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    decision = Column(String, nullable=False)
    reason = Column(String, nullable=False)
    features = Column(JSONB)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # The fallback feature queries filter on (user_id, created_at range).
    __table_args__ = (Index("ix_transactions_user_created", "user_id", "created_at"),)


class OutboxDB(Base):
    """Transactional outbox: events written in the same DB transaction as the
    decision, then published to Kafka by relay.py.

    This guarantees "decision saved" and "event will be published" are
    atomic — there is no window where one happens without the other.
    """

    __tablename__ = "outbox"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    topic = Column(String, nullable=False)
    key = Column(String, nullable=False)
    payload = Column(JSONB, nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    published_at = Column(DateTime(timezone=True), nullable=True)

    # Partial index: the relay only ever scans unpublished rows.
    __table_args__ = (
        Index("ix_outbox_unpublished", "id", postgresql_where=text("published_at IS NULL")),
    )
