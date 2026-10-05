"""Pydantic schemas for Kafka events.

These schemas are the *contract* between the producer (API) and every
consumer.  If you change a field here, every consumer must be updated —
which is exactly why schema registries exist in production Kafka setups.
"""

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class TransactionEvent(BaseModel):
    event_type: str = "transaction_scored"
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    transaction_id: str
    user_id: str
    amount: float
    decision: str
    reason: str
    features: dict
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
