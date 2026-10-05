"""Integration tests run against the full docker-compose stack.

    docker compose up -d --build
    INTEGRATION=1 pytest tests/integration
"""

import os
import time
import uuid

import pytest
import requests
from sqlalchemy import create_engine, text

# Only collect these tests when explicitly asked to; they need the stack up.
if os.getenv("INTEGRATION") != "1":
    collect_ignore_glob = ["test_*.py"]

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://fraud_user:password@localhost:5432/fraud_db"
)
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


@pytest.fixture(scope="session")
def api():
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            if requests.get(f"{BASE_URL}/health", timeout=2).ok:
                break
        except requests.ConnectionError:
            pass
        time.sleep(2)
    else:
        pytest.fail(f"API at {BASE_URL} never became healthy")

    session = requests.Session()
    session.base_url = BASE_URL
    return session


@pytest.fixture(scope="session")
def db():
    engine = create_engine(DATABASE_URL)
    yield engine
    engine.dispose()


@pytest.fixture
def unique():
    return lambda prefix: f"{prefix}-{uuid.uuid4().hex[:12]}"


def post_txn(api, transaction_id, user_id, amount):
    return api.post(
        f"{api.base_url}/transactions",
        json={"transaction_id": transaction_id, "user_id": user_id, "amount": amount},
        timeout=10,
    )


def wait_for(predicate, timeout=60, interval=0.5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(interval)
    return None


def audit_row(db, transaction_id):
    with db.connect() as conn:
        return conn.execute(
            text("SELECT * FROM audit_log WHERE transaction_id = :t"), {"t": transaction_id}
        ).mappings().first()
