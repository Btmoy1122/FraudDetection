"""Database-backed feature computation — the fallback path.

When Redis is available, feature_store.py handles feature reads.
This module is called only when Redis is unreachable or for
cache-miss backfill of the 30-day average.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import func as sqlfunc

from models import TransactionDB


def compute_features_from_db(user_id: str, db) -> dict:
    now = datetime.now(timezone.utc)
    one_hour_ago = now - timedelta(hours=1)
    thirty_days_ago = now - timedelta(days=30)

    txn_count_last_1h = db.query(
        sqlfunc.count(TransactionDB.transaction_id)
    ).filter(
        TransactionDB.user_id == user_id,
        TransactionDB.created_at >= one_hour_ago,
    ).scalar()

    total_amount_last_1h = db.query(
        sqlfunc.coalesce(sqlfunc.sum(TransactionDB.amount), 0.0)
    ).filter(
        TransactionDB.user_id == user_id,
        TransactionDB.created_at >= one_hour_ago,
    ).scalar()

    avg_amount_last_30d = db.query(
        sqlfunc.avg(TransactionDB.amount)
    ).filter(
        TransactionDB.user_id == user_id,
        TransactionDB.created_at >= thirty_days_ago,
    ).scalar()

    return {
        "txn_count_last_1h": int(txn_count_last_1h or 0),
        "total_amount_last_1h": float(total_amount_last_1h or 0.0),
        "avg_amount_last_30d": (
            float(avg_amount_last_30d) if avg_amount_last_30d else None
        ),
    }
