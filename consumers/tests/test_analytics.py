from datetime import datetime, timezone

import pytest

from analytics import STREAM_DECISIONS, SeenSet, make_handler
from runner import NonRetryableError


def event(event_id="e1", decision="DENY"):
    return {
        "event_id": event_id,
        "decision": decision,
        "reason": "amount_spike",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def counter_value(decision):
    return STREAM_DECISIONS.labels(decision=decision, reason="amount_spike")._value.get()


def test_seen_set_detects_duplicates_and_evicts_oldest():
    s = SeenSet(maxlen=2)
    assert s.add("a") and s.add("b")
    assert not s.add("a")
    s.add("c")  # evicts "a"
    assert s.add("a")


def test_duplicate_events_are_counted_once():
    handle = make_handler()
    before = counter_value("DENY")
    handle(event("dup-1"), None)
    handle(event("dup-1"), None)
    assert counter_value("DENY") == before + 1


def test_malformed_event_is_non_retryable():
    with pytest.raises(NonRetryableError):
        make_handler()({"event_id": "x"}, None)
