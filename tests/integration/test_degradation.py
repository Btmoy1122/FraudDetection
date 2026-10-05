"""Failure-mode tests: stop a dependency, prove the system keeps working.

These stop and start containers, so they run as a separate step:
    INTEGRATION=1 pytest tests/integration/test_degradation.py
"""

import subprocess
from pathlib import Path

import pytest
import requests

from conftest import audit_row, post_txn, wait_for

COMPOSE_FILE = Path(__file__).resolve().parents[2] / "docker-compose.yml"


def compose(*args):
    subprocess.run(["docker", "compose", "-f", str(COMPOSE_FILE), *args], check=True)


def fallback_count(api):
    for line in requests.get(f"{api.base_url}/metrics", timeout=5).text.splitlines():
        if line.startswith("fraud_feature_store_fallback_total "):
            return float(line.split()[1])
    return 0.0


@pytest.fixture
def redis_stopped():
    compose("stop", "redis")
    yield
    compose("start", "redis")


@pytest.fixture
def kafka_stopped():
    compose("stop", "kafka")
    yield
    compose("start", "kafka")


def test_api_keeps_scoring_when_redis_is_down(api, unique, redis_stopped):
    before = fallback_count(api)
    resp = post_txn(api, unique("txn"), unique("user"), 25.0)
    assert resp.status_code == 200
    assert resp.json()["decision"] == "APPROVE"
    assert fallback_count(api) > before


def test_events_survive_kafka_outage(api, db, unique, kafka_stopped):
    """With Kafka down the API still answers (it only writes to Postgres);
    the event waits in the outbox and is delivered once Kafka is back."""
    tid = unique("txn")
    resp = post_txn(api, tid, unique("user"), 40.0)
    assert resp.status_code == 200

    compose("start", "kafka")
    row = wait_for(lambda: audit_row(db, tid), timeout=180, interval=2)
    assert row is not None, "event was lost during the Kafka outage"
