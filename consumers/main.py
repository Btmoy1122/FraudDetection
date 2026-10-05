"""Consumer runner — starts all Kafka consumers in separate threads.

Each consumer joins a different consumer group, so they all independently
receive every message from the transactions topic.  Threading is fine here
because each consumer is I/O-bound (waiting for Kafka messages), not
CPU-bound.

If any consumer thread dies, the process exits non-zero and Docker
restarts it.  Uncommitted offsets are redelivered, so nothing is lost.
"""

import logging
import sys
import threading
import time

from prometheus_client import start_http_server

from analytics import run as run_analytics
from audit_writer import run as run_audit_writer
from settings import METRICS_PORT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

CONSUMERS = [
    ("audit-writer", run_audit_writer),
    ("analytics", run_analytics),
]


def main() -> int:
    start_http_server(METRICS_PORT)
    threads: list[threading.Thread] = []

    for name, target in CONSUMERS:
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()
        threads.append(t)
        logger.info("Started consumer: %s", name)

    try:
        while True:
            time.sleep(5)
            for t in threads:
                if not t.is_alive():
                    logger.error("Consumer %s died — exiting", t.name)
                    return 1
    except KeyboardInterrupt:
        logger.info("Shutting down consumers")
        return 0


if __name__ == "__main__":
    sys.exit(main())
