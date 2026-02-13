def apply_rules(features: dict, amount: float):
    if features["txn_count_last_1h"] > 5:
        return "DENY", "too_many_txns_last_1h"
    if features["avg_amount_last_30d"] is not None and amount > 3 * features["avg_amount_last_30d"]:
        return "DENY", "amount_spike"
    return "APPROVE", "rules_passed"