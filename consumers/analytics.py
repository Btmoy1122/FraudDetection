"""Kafka consumer that turns the decision stream into Prometheus metrics.

Key metric: end-to-end pipeline latency — the time from the API making a
decision to this consumer seeing the event.  That covers the outbox, the
relay, Kafka, and consumer lag in one number.

Duplicates (at-least-once redelivery) are filtered with a bounded set of
recently seen event_ids.  It lives in memory, so it resets on restart;
that's acceptable for best-effort metrics, not for anything financial.
"""

import logging
from collections import OrderedDict
from datetime import datetime, timezone

from prometheus_client import Counter, Histogram

from runner import NonRetryableError, run_consumer

logger = logging.getLogger(__name__)

SEEN_MAX = 100_000

STREAM_DECISIONS = Counter(
    "fraud_stream_decisions_total", "Decisions seen on the Kafka stream", ["decision", "reason"]
)
PIPELINE_LATENCY = Histogram(
    "fraud_event_pipeline_latency_seconds",
    "Decision time → analytics consumer receipt (outbox + relay + Kafka + lag)",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 300),
)


class SeenSet:
    """Bounded FIFO set of recently processed ids."""

    def __init__(self, maxlen: int = SEEN_MAX):
        self._items: OrderedDict[str, None] = OrderedDict()
        self._maxlen = maxlen

    def add(self, key: str) -> bool:
        """Add *key*; return False if it was already present."""
        if key in self._items:
            return False
        self._items[key] = None
        if len(self._items) > self._maxlen:
            self._items.popitem(last=False)
        return True


def make_handler():
    seen = SeenSet()

    def handle(event: dict, message) -> None:
        try:
            event_id = event["event_id"]
            decision, reason = event["decision"], event["reason"]
            decided_at = datetime.fromisoformat(event["timestamp"])
        except (KeyError, ValueError) as exc:
            raise NonRetryableError(str(exc)) from exc

        if not seen.add(event_id):
            return

        STREAM_DECISIONS.labels(decision=decision, reason=reason).inc()
        PIPELINE_LATENCY.observe(
            max((datetime.now(timezone.utc) - decided_at).total_seconds(), 0)
        )

    return handle


def run() -> None:
    run_consumer("analytics", group_id="analytics", handler=make_handler())
