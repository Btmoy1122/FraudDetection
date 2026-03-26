"""Pydantic schemas for Kafka events.

These schemas are the *contract* between the producer (API) and every
consumer.  If you change a field here, every consumer must be updated —
which is exactly why schema registries exist in production Kafka setups.
"""

from datetime import datetime, timezone

from pydantic import BaseModel


class TransactionEvent(BaseModel):
    event_type: str = "transaction_scored"
    transaction_id: str
    user_id: str
    amount: float
    decision: str
    reason: str
    features: dict
    timestamp: str

    @classmethod
    def from_transaction(
        cls,
        *,
        transaction_id: str,
        user_id: str,
        amount: float,
        decision: str,
        reason: str,
        features: dict,
    ) -> "TransactionEvent":
        return cls(
            transaction_id=transaction_id,
            user_id=user_id,
            amount=amount,
            decision=decision,
            reason=reason,
            features=features,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
