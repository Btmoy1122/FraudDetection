import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://fraud_user:password@localhost:5432/fraud_db",
)

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TRANSACTION_TOPIC = os.getenv("KAFKA_TRANSACTION_TOPIC", "transactions")
