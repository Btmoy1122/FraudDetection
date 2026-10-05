import uuid
from concurrent.futures import ThreadPoolExecutor

from kafka import KafkaConsumer, KafkaProducer

from conftest import KAFKA_BOOTSTRAP_SERVERS, audit_row, post_txn, wait_for


def test_transaction_is_scored_and_stored(api, unique):
    tid, uid = unique("txn"), unique("user")
    resp = post_txn(api, tid, uid, 49.99)
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "APPROVE"
    assert body["idempotent"] is False

    stored = api.get(f"{api.base_url}/transactions/{tid}", timeout=5).json()
    assert stored["decision"] == "APPROVE"
    assert stored["user_id"] == uid


def test_retry_returns_original_decision(api, unique):
    tid, uid = unique("txn"), unique("user")
    first = post_txn(api, tid, uid, 20.0).json()
    retry = post_txn(api, tid, uid, 20.0).json()
    assert retry["idempotent"] is True
    assert retry["decision"] == first["decision"]


def test_concurrent_duplicates_get_one_decision(api, unique):
    tid, uid = unique("txn"), unique("user")
    with ThreadPoolExecutor(10) as pool:
        bodies = [r.json() for r in pool.map(lambda _: post_txn(api, tid, uid, 5.0), range(10))]
    assert sum(not b["idempotent"] for b in bodies) == 1
    assert len({b["decision"] for b in bodies}) == 1


def test_velocity_limit_holds_under_concurrent_burst(api, unique):
    """20 simultaneous transactions for one user: exactly 5 may be approved.
    Before the atomic Lua script, concurrent requests read the same stale
    count and more than 5 got through."""
    uid = unique("burst")
    with ThreadPoolExecutor(20) as pool:
        bodies = [
            r.json()
            for r in pool.map(lambda i: post_txn(api, f"{uid}-{i}", uid, 10.0), range(20))
        ]
    decisions = [b["decision"] for b in bodies]
    assert decisions.count("APPROVE") == 5
    assert all(
        b["reason"] == "too_many_txns_last_1h" for b in bodies if b["decision"] == "DENY"
    )


def test_amount_spike_is_denied(api, unique):
    uid = unique("user")
    for i in range(3):
        post_txn(api, f"{uid}-{i}", uid, 10.0)
    resp = post_txn(api, f"{uid}-spike", uid, 1000.0).json()
    assert (resp["decision"], resp["reason"]) == ("DENY", "amount_spike")


def test_event_flows_through_kafka_to_audit_log(api, db, unique):
    """API → outbox → relay → Kafka → audit consumer → Postgres."""
    tid, uid = unique("txn"), unique("user")
    post_txn(api, tid, uid, 33.0)

    row = wait_for(lambda: audit_row(db, tid), timeout=60)
    assert row is not None, "event never reached the audit_log table"
    assert row["decision"] == "APPROVE"
    assert row["user_id"] == uid


def test_poison_message_is_dead_lettered():
    marker = f"poison-{uuid.uuid4().hex}"
    producer = KafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS, acks="all")
    producer.send("transactions", key=b"poison", value=f"not json {marker}".encode()).get(10)
    producer.close()

    dlq = KafkaConsumer(
        "transactions.dlq",
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=marker,
        auto_offset_reset="earliest",
        consumer_timeout_ms=60_000,
    )
    try:
        for message in dlq:
            if marker.encode() in message.value:
                headers = dict(message.headers)
                assert headers["dlq.error"].startswith(b"deserialize")
                assert headers["dlq.source.topic"] == b"transactions"
                return
    finally:
        dlq.close()
    raise AssertionError("poison message never reached the DLQ")


def test_transactions_topic_has_multiple_partitions():
    consumer = KafkaConsumer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
    try:
        assert len(consumer.partitions_for_topic("transactions")) == 6
    finally:
        consumer.close()


def test_metrics_endpoint_exposes_decision_metrics(api):
    body = api.get(f"{api.base_url}/metrics", timeout=5).text
    assert "fraud_decisions_total" in body
    assert "fraud_decision_latency_seconds_bucket" in body


def test_invalid_payload_is_rejected(api):
    resp = api.post(
        f"{api.base_url}/transactions",
        json={"transaction_id": "x", "user_id": "u", "amount": -5},
        timeout=5,
    )
    assert resp.status_code == 422
