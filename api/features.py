"""Database-backed feature computation — the fallback path.

When Redis is available, feature_store.py handles feature reads.
This module is called only when Redis is unreachable or for
cache-miss backfill of the 30-day average.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import func as sqlfunc

from models import TransactionDB


def compute_features_from_db(user_id: str, db) -> dict:
    one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)

    txn_count_last_1h, total_amount_last_1h = db.query(
        sqlfunc.count(TransactionDB.transaction_id),
        sqlfunc.coalesce(sqlfunc.sum(TransactionDB.amount), 0.0),
    ).filter(
        TransactionDB.user_id == user_id,
        TransactionDB.created_at >= one_hour_ago,
    ).one()

    return {
        "txn_count_last_1h": int(txn_count_last_1h or 0),
        "total_amount_last_1h": float(total_amount_last_1h or 0.0),
        "avg_amount_last_30d": compute_avg_30d_from_db(user_id, db),
    }


def compute_avg_30d_from_db(user_id: str, db) -> float | None:
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)

    avg = db.query(
        sqlfunc.avg(TransactionDB.amount)
    ).filter(
        TransactionDB.user_id == user_id,
        TransactionDB.created_at >= thirty_days_ago,
    ).scalar()

    return float(avg) if avg is not None else None
