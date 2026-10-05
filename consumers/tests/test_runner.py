import json
from dataclasses import dataclass

import pytest

from runner import MAX_ATTEMPTS, NonRetryableError, process_message


@dataclass
class FakeMessage:
    value: bytes
    key: bytes = b"u1"
    topic: str = "transactions"
    partition: int = 0
    offset: int = 42


class Transient(Exception):
    pass


@pytest.fixture
def dlq():
    sent = []
    return sent, lambda msg, error, consumer: sent.append((msg, error, consumer))


def msg(obj) -> FakeMessage:
    return FakeMessage(value=json.dumps(obj).encode())


def no_sleep(_):
    pass


def test_success_calls_handler_once(dlq):
    sent, publish = dlq
    calls = []
    outcome = process_message(msg({"a": 1}), lambda e, m: calls.append(e), publish, "c",
                              sleep=no_sleep)
    assert outcome == "ok"
    assert calls == [{"a": 1}]
    assert sent == []


@pytest.mark.parametrize("raw", [b"not json", b"\xff\xfe", b"[1, 2]", b'"str"'])
def test_poison_message_goes_to_dlq_without_calling_handler(dlq, raw):
    sent, publish = dlq
    calls = []
    outcome = process_message(FakeMessage(value=raw), lambda e, m: calls.append(e), publish,
                              "c", sleep=no_sleep)
    assert outcome == "dlq"
    assert calls == []
    assert sent[0][1].startswith("deserialize")


def test_non_retryable_error_goes_to_dlq_after_one_attempt(dlq):
    sent, publish = dlq
    calls = []

    def handler(e, m):
        calls.append(1)
        raise NonRetryableError("missing fields")

    assert process_message(msg({}), handler, publish, "c", sleep=no_sleep) == "dlq"
    assert len(calls) == 1
    assert "non-retryable" in sent[0][1]


def test_generic_error_retries_then_dead_letters(dlq):
    sent, publish = dlq
    calls = []
    sleeps = []

    def handler(e, m):
        calls.append(1)
        raise RuntimeError("boom")

    outcome = process_message(msg({}), handler, publish, "c", sleep=sleeps.append)
    assert outcome == "dlq"
    assert len(calls) == MAX_ATTEMPTS
    assert len(sleeps) == MAX_ATTEMPTS - 1
    assert sleeps == sorted(sleeps)  # exponential backoff
    assert sent[0][2] == "c"


def test_generic_error_that_recovers_is_not_dead_lettered(dlq):
    sent, publish = dlq
    attempts = iter([RuntimeError("blip"), None])

    def handler(e, m):
        exc = next(attempts)
        if exc:
            raise exc

    assert process_message(msg({}), handler, publish, "c", sleep=no_sleep) == "ok"
    assert sent == []


def test_transient_error_retries_past_max_attempts(dlq):
    """A dependency outage (e.g. Postgres down) must not flood the DLQ."""
    sent, publish = dlq
    failures = MAX_ATTEMPTS * 3
    calls = []

    def handler(e, m):
        calls.append(1)
        if len(calls) <= failures:
            raise Transient("db down")

    outcome = process_message(msg({}), handler, publish, "c",
                              transient_errors=(Transient,), sleep=no_sleep)
    assert outcome == "ok"
    assert len(calls) == failures + 1
    assert sent == []
