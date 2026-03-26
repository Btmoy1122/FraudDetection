def apply_rules(features: dict, amount: float) -> tuple[str, str]:
    if features["txn_count_last_1h"] > 5:
        return "DENY", "too_many_txns_last_1h"

    avg = features["avg_amount_last_30d"]
    if avg is not None and amount > 3 * avg:
        return "DENY", "amount_spike"

    return "APPROVE", "rules_passed"
