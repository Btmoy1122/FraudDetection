import os
from pathlib import Path

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://fraud_user:password@localhost:5432/fraud_db",
)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TRANSACTION_TOPIC = os.getenv("KAFKA_TRANSACTION_TOPIC", "transactions")

_default_artifacts = Path(__file__).resolve().parent / "artifacts"
if not _default_artifacts.exists():
    _default_artifacts = Path(__file__).resolve().parent.parent / "ml" / "artifacts"
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(_default_artifacts)))
