from datetime import datetime

from events import TransactionEvent


def make_event():
    return TransactionEvent(
        transaction_id="t1", user_id="u1", amount=10.0,
        decision="APPROVE", reason="rules_passed", features={},
    )


def test_event_gets_unique_id_and_utc_timestamp():
    a, b = make_event(), make_event()
    assert a.event_id != b.event_id
    assert datetime.fromisoformat(a.timestamp).utcoffset().total_seconds() == 0


def test_event_round_trips_through_json():
    e = make_event()
    assert TransactionEvent.model_validate_json(e.model_dump_json()) == e
