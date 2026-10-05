MAX_TXNS_PER_HOUR = 5
SPIKE_MULTIPLIER = 3


def apply_rules(features: dict, amount: float) -> tuple[str, str]:
    """Return (decision, reason).  Features describe the user's history
    BEFORE this transaction."""
    if features["txn_count_last_1h"] >= MAX_TXNS_PER_HOUR:
        return "DENY", "too_many_txns_last_1h"

    avg = features["avg_amount_last_30d"]
    if avg is not None and amount > SPIKE_MULTIPLIER * avg:
        return "DENY", "amount_spike"

    return "APPROVE", "rules_passed"
