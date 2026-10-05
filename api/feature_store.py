"""Redis-backed feature store for real-time fraud detection features.

1-hour window (count + sum):
    One sorted set per user, ``user:{id}:txns_1h``.  Each member is
    ``"{amount}:{transaction_id}"`` scored by its Unix timestamp.  A Lua
    script prunes expired entries, reads the window, and records the new
    transaction in ONE atomic step.

    Why atomic: the old flow was read → decide → write.  Two concurrent
    requests for the same user could both read count=5, both pass the
    "> 5 per hour" rule, and both get approved.  Redis runs a Lua script
    without interleaving other commands, so every request sees a distinct
    count and a burst can't slip past the velocity limit.

30-day average:
    Cache-aside with a 5-minute TTL.  A 30-day average barely moves per
    transaction, so bounded staleness is fine and keeps Postgres load flat.

Fallback:
    Any Redis error falls back to Postgres queries.  Redis accelerates the
    hot path; it is never a single point of failure.
"""

import logging
import time
from typing import Callable

import redis

from metrics import FEATURE_FALLBACKS
from redis_client import get_redis

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 3600
AVG_30D_TTL = 300

# KEYS[1] = window key
# ARGV[1] = now (unix seconds), ARGV[2] = window seconds, ARGV[3] = member
# Returns {count, total} of the OTHER transactions in the window, i.e. the
# state before this transaction.  Excluding the member itself means a retry
# of the same transaction_id doesn't count against itself.
_RECORD_AND_READ_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local member = ARGV[3]

redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)

local count = 0
local total = 0
for _, m in ipairs(redis.call('ZRANGE', key, 0, -1)) do
    if m ~= member then
        count = count + 1
        total = total + tonumber(string.match(m, '^([^:]+):'))
    end
end

redis.call('ZADD', key, 'NX', now, member)
redis.call('EXPIRE', key, window)

-- Lua numbers are truncated to integers in Redis replies, so send the
-- sum back as a string.
return {count, tostring(total)}
"""

_script = None


def _window_key(user_id: str) -> str:
    return f"user:{user_id}:txns_1h"


def _avg_key(user_id: str) -> str:
    return f"user:{user_id}:avg_30d"


def _member(transaction_id: str, amount: float) -> str:
    # Amount first, so a ':' inside the transaction_id can't break parsing.
    return f"{amount}:{transaction_id}"


def record_and_read_window(
    r: redis.Redis, user_id: str, transaction_id: str, amount: float, now: float | None = None
) -> tuple[int, float]:
    """Atomically record the transaction and return the prior (count, total)."""
    global _script
    if _script is None:
        # Script objects call EVALSHA and transparently re-send the source
        # if Redis doesn't have it cached (NOSCRIPT).
        _script = r.register_script(_RECORD_AND_READ_LUA)
    now = time.time() if now is None else now
    count, total = _script(
        keys=[_window_key(user_id)],
        args=[now, WINDOW_SECONDS, _member(transaction_id, amount)],
        client=r,
    )
    return int(count), float(total)


def _cached_avg_30d(
    r: redis.Redis, user_id: str, db_avg: Callable[[str], float | None]
) -> float | None:
    cached = r.get(_avg_key(user_id))
    if cached is not None:
        return float(cached)
    avg = db_avg(user_id)
    if avg is not None:
        r.set(_avg_key(user_id), str(avg), ex=AVG_30D_TTL)
    return avg


def get_features(
    user_id: str,
    transaction_id: str,
    amount: float,
    db_features: Callable[[str], dict],
    db_avg: Callable[[str], float | None],
) -> dict:
    """Return features for scoring, recording this transaction in the window.

    Falls back to *db_features* (no atomicity guarantee) if Redis fails.
    """
    try:
        r = get_redis()
        count, total = record_and_read_window(r, user_id, transaction_id, amount)
        avg_30d = _cached_avg_30d(r, user_id, db_avg)
    except redis.RedisError:
        FEATURE_FALLBACKS.inc()
        logger.warning("Redis unavailable — computing features from Postgres")
        return db_features(user_id)

    return {
        "txn_count_last_1h": count,
        "total_amount_last_1h": total,
        "avg_amount_last_30d": avg_30d,
    }


def forget_transaction(user_id: str, transaction_id: str, amount: float) -> None:
    """Best-effort removal from the window when the DB write fails, so a
    transaction that was never persisted doesn't count toward velocity."""
    try:
        get_redis().zrem(_window_key(user_id), _member(transaction_id, amount))
    except redis.RedisError:
        logger.warning("Redis unavailable — could not roll back window entry")
