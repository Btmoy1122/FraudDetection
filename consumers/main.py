"""Consumer runner — starts all Kafka consumers in separate threads.

Each consumer joins a different consumer group, so they all independently
receive every message from the transactions topic.  Threading is fine here
because each consumer is I/O-bound (waiting for Kafka messages), not
CPU-bound.
"""

import logging
import threading
import time

from analytics import run as run_analytics
from audit_logger import run as run_audit_logger
from db_writer import run as run_db_writer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

CONSUMERS = [
    ("db-writer", run_db_writer),
    ("audit-logger", run_audit_logger),
    ("analytics", run_analytics),
]


def main() -> None:
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
                    return
    except KeyboardInterrupt:
        logger.info("Shutting down consumers")


if __name__ == "__main__":
    main()
