import threading

import fakeredis
import pytest
import redis

import feature_store
from feature_store import AVG_30D_TTL, WINDOW_SECONDS, get_features, record_and_read_window

NOW = 1_700_000_000.0


@pytest.fixture
def r(monkeypatch):
    client = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(feature_store, "get_redis", lambda: client)
    monkeypatch.setattr(feature_store, "_script", None)
    return client


def test_first_transaction_sees_empty_window(r):
    assert record_and_read_window(r, "u1", "t1", 50.0, now=NOW) == (0, 0.0)


def test_window_counts_and_sums_prior_transactions(r):
    record_and_read_window(r, "u1", "t1", 50.0, now=NOW)
    record_and_read_window(r, "u1", "t2", 25.5, now=NOW + 1)
    assert record_and_read_window(r, "u1", "t3", 1.0, now=NOW + 2) == (2, 75.5)


def test_entries_older_than_window_are_pruned(r):
    record_and_read_window(r, "u1", "old", 100.0, now=NOW)
    record_and_read_window(r, "u1", "recent", 10.0, now=NOW + WINDOW_SECONDS - 1)
    count, total = record_and_read_window(r, "u1", "t3", 1.0, now=NOW + WINDOW_SECONDS + 1)
    assert (count, total) == (1, 10.0)
    assert "100.0:old" not in r.zrange("user:u1:txns_1h", 0, -1)


def test_users_are_isolated(r):
    record_and_read_window(r, "u1", "t1", 50.0, now=NOW)
    assert record_and_read_window(r, "u2", "t2", 50.0, now=NOW) == (0, 0.0)


def test_retry_of_same_transaction_does_not_count_itself(r):
    record_and_read_window(r, "u1", "t1", 50.0, now=NOW)
    assert record_and_read_window(r, "u1", "t1", 50.0, now=NOW + 1) == (0, 0.0)
    assert r.zcard("user:u1:txns_1h") == 1


def test_colon_in_transaction_id_is_parsed(r):
    record_and_read_window(r, "u1", "order:123:abc", 42.5, now=NOW)
    assert record_and_read_window(r, "u1", "t2", 1.0, now=NOW) == (1, 42.5)


def test_window_key_expires(r):
    record_and_read_window(r, "u1", "t1", 50.0, now=NOW)
    assert 0 < r.ttl("user:u1:txns_1h") <= WINDOW_SECONDS


def test_concurrent_requests_each_see_distinct_counts(r):
    """The race the Lua script fixes: N concurrent requests must observe
    counts 0..N-1 exactly once each, never the same stale count."""
    n = 20
    results = []
    lock = threading.Lock()

    def worker(i):
        count, _ = record_and_read_window(r, "burst", f"t{i}", 10.0, now=NOW)
        with lock:
            results.append(count)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == list(range(n))


def test_avg_30d_is_cached_after_db_lookup(r):
    calls = []

    def db_avg(uid):
        calls.append(uid)
        return 80.0

    for i in range(3):
        f = get_features("u1", f"t{i}", 10.0, db_features=None, db_avg=db_avg)
        assert f["avg_amount_last_30d"] == 80.0

    assert calls == ["u1"]
    assert 0 < r.ttl("user:u1:avg_30d") <= AVG_30D_TTL


def test_missing_avg_is_not_cached(r):
    get_features("u1", "t1", 10.0, db_features=None, db_avg=lambda uid: None)
    assert r.get("user:u1:avg_30d") is None


def test_falls_back_to_db_when_redis_is_down(monkeypatch):
    class DeadRedis:
        def register_script(self, _):
            raise redis.ConnectionError("down")

    monkeypatch.setattr(feature_store, "get_redis", lambda: DeadRedis())
    monkeypatch.setattr(feature_store, "_script", None)
    db_result = {"txn_count_last_1h": 3, "total_amount_last_1h": 30.0, "avg_amount_last_30d": 10.0}

    before = feature_store.FEATURE_FALLBACKS._value.get()
    f = get_features("u1", "t1", 10.0, db_features=lambda uid: db_result, db_avg=None)

    assert f == db_result
    assert feature_store.FEATURE_FALLBACKS._value.get() == before + 1


def test_forget_transaction_removes_entry(r):
    record_and_read_window(r, "u1", "t1", 50.0, now=NOW)
    feature_store.forget_transaction("u1", "t1", 50.0)
    assert r.zcard("user:u1:txns_1h") == 0
