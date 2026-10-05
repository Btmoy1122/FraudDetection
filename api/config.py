import os
from pathlib import Path

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://fraud_user:password@localhost:5432/fraud_db",
)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
# Fail fast when Redis is down so the Postgres fallback kicks in quickly
# instead of every request hanging on a dead socket.
REDIS_TIMEOUT_SECONDS = float(os.getenv("REDIS_TIMEOUT_SECONDS", "0.5"))

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TRANSACTION_TOPIC = os.getenv("KAFKA_TRANSACTION_TOPIC", "transactions")

# Outbox relay
RELAY_BATCH_SIZE = int(os.getenv("RELAY_BATCH_SIZE", "500"))
RELAY_POLL_INTERVAL_SECONDS = float(os.getenv("RELAY_POLL_INTERVAL_SECONDS", "0.1"))
RELAY_METRICS_PORT = int(os.getenv("RELAY_METRICS_PORT", "8002"))

_default_artifacts = Path(__file__).resolve().parent / "artifacts"
if not _default_artifacts.exists():
    _default_artifacts = Path(__file__).resolve().parent.parent / "ml" / "artifacts"
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(_default_artifacts)))
