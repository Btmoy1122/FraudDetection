"""Redis-backed feature store for real-time fraud detection features.

Read path:
    Sorted-set sliding window for 1-hour transaction count and sum.
    Cache-aside with TTL for 30-day average amount.

Write path:
    ZADD new transactions into the sorted set after scoring.
    Invalidate the 30-day average cache so the next miss re-fetches.

Fallback:
    If Redis is unreachable the caller-supplied db_fallback function
    is used instead.  Redis should accelerate the hot path, never be
    a single point of failure.
"""

import logging
import time
from typing import Callable

from redis_client import get_redis

logger = logging.getLogger(__name__)

ONE_HOUR = 3600
AVG_30D_TTL = 300  # 5 min — a 30-day average barely moves per txn


def _empty_features() -> dict:
    return {
        "txn_count_last_1h": 0,
        "total_amount_last_1h": 0.0,
        "avg_amount_last_30d": None,
    }


# ---------------------------------------------------------------------------
# Read interface
# ---------------------------------------------------------------------------

def read_features(
    user_id: str,
    db_fallback: Callable | None = None,
) -> dict:
    """Fetch features from Redis.  Falls back to *db_fallback* on error."""
    try:
        r = get_redis()
        r.ping()
        return _read_from_redis(r, user_id, db_fallback)
    except Exception:
        logger.warning("Redis unavailable — falling back to database")
        if db_fallback:
            return db_fallback(user_id)
        return _empty_features()


def _read_from_redis(r, user_id: str, db_fallback) -> dict:
    now = time.time()
    one_hour_ago = now - ONE_HOUR

    key = f"user:{user_id}:txns_1h"

    # Prune stale entries then read the window in a pipeline (one round-trip).
    pipe = r.pipeline()
    pipe.zremrangebyscore(key, "-inf", one_hour_ago)
    pipe.zrangebyscore(key, one_hour_ago, "+inf")
    _, entries = pipe.execute()

    txn_count = len(entries)
    total_amount = sum(
        float(member.split(":")[1]) for member in entries
    ) if entries else 0.0

    # Cache-aside for the 30-day average
    avg_key = f"user:{user_id}:avg_30d"
    avg_30d_raw = r.get(avg_key)

    if avg_30d_raw is not None:
        avg_30d = float(avg_30d_raw)
    elif db_fallback:
        db_features = db_fallback(user_id)
        avg_30d = db_features.get("avg_amount_last_30d")
        if avg_30d is not None:
            r.setex(avg_key, AVG_30D_TTL, str(avg_30d))
    else:
        avg_30d = None

    return {
        "txn_count_last_1h": txn_count,
        "total_amount_last_1h": total_amount,
        "avg_amount_last_30d": avg_30d,
    }


# ---------------------------------------------------------------------------
# Write interface
# ---------------------------------------------------------------------------

def update_features(user_id: str, transaction_id: str, amount: float) -> None:
    """Record a new transaction in Redis so the next read reflects it.

    Called inline (synchronously) right after scoring, because the very
    next request for this user must see updated counters.
    """
    try:
        r = get_redis()
        now = time.time()
        key = f"user:{user_id}:txns_1h"

        pipe = r.pipeline()
        pipe.zadd(key, {f"{transaction_id}:{amount}": now})
        pipe.expire(key, ONE_HOUR)
        pipe.delete(f"user:{user_id}:avg_30d")
        pipe.execute()
    except Exception:
        logger.warning("Redis unavailable — skipping feature update")
