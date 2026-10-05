import logging

import redis

from config import REDIS_TIMEOUT_SECONDS, REDIS_URL

logger = logging.getLogger(__name__)

_pool: redis.ConnectionPool | None = None


def _get_pool() -> redis.ConnectionPool:
    global _pool
    if _pool is None:
        _pool = redis.ConnectionPool.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=REDIS_TIMEOUT_SECONDS,
            socket_timeout=REDIS_TIMEOUT_SECONDS,
        )
    return _pool


def get_redis() -> redis.Redis:
    """Return a Redis client backed by a shared connection pool.

    Connection pooling avoids opening a new TCP socket on every request.
    The pool is created lazily on first call so the import itself never
    blocks or raises if Redis is unreachable.
    """
    return redis.Redis(connection_pool=_get_pool())


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.disconnect()
        _pool = None
