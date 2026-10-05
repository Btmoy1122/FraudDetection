import pytest

from rules import MAX_TXNS_PER_HOUR, apply_rules


def features(count=0, avg=None):
    return {"txn_count_last_1h": count, "total_amount_last_1h": 0.0, "avg_amount_last_30d": avg}


def test_new_user_is_approved():
    assert apply_rules(features(), 500.0) == ("APPROVE", "rules_passed")


@pytest.mark.parametrize("prior", range(MAX_TXNS_PER_HOUR))
def test_up_to_limit_is_approved(prior):
    assert apply_rules(features(count=prior), 10.0)[0] == "APPROVE"


def test_over_limit_is_denied():
    assert apply_rules(features(count=MAX_TXNS_PER_HOUR), 10.0) == (
        "DENY",
        "too_many_txns_last_1h",
    )


def test_amount_spike_is_denied():
    assert apply_rules(features(avg=100.0), 300.01) == ("DENY", "amount_spike")


def test_amount_at_spike_boundary_is_approved():
    assert apply_rules(features(avg=100.0), 300.0)[0] == "APPROVE"


def test_velocity_checked_before_spike():
    assert apply_rules(features(count=10, avg=1.0), 1000.0)[1] == "too_many_txns_last_1h"
