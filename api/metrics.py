"""Prometheus metrics for the API (scraped from GET /metrics)."""

from prometheus_client import Counter, Histogram

DECISION_LATENCY = Histogram(
    "fraud_decision_latency_seconds",
    "Server-side time to handle POST /transactions",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

DECISIONS = Counter(
    "fraud_decisions_total",
    "Newly scored transactions",
    ["decision", "reason"],
)

IDEMPOTENT_REPLAYS = Counter(
    "fraud_idempotent_replays_total",
    "Duplicate transaction_ids answered with the stored decision",
)

FEATURE_FALLBACKS = Counter(
    "fraud_feature_store_fallback_total",
    "Requests whose features came from Postgres because Redis failed",
)
