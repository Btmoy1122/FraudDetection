from datetime import datetime, timedelta
from sqlalchemy import func
from models import TransactionDB

def compute_features(user_id: str, db):
    now = datetime.utcnow()
    one_hour_ago = now - timedelta(hours=1)
    thirty_days_ago = now - timedelta(days=30)

    txn_count_last_1h = db.query(func.count(TransactionDB.transaction_id)).filter(
        TransactionDB.user_id == user_id,
        TransactionDB.created_at >= one_hour_ago
    ).scalar()

    total_amount_last_1h = db.query(func.coalesce(func.sum(TransactionDB.amount), 0.0)).filter(
        TransactionDB.user_id == user_id,
        TransactionDB.created_at >= one_hour_ago
    ).scalar()

    avg_amount_last_30d = db.query(func.avg(TransactionDB.amount)).filter(
        TransactionDB.user_id == user_id,
        TransactionDB.created_at >= thirty_days_ago
    ).scalar()

    return {
        "txn_count_last_1h": int(txn_count_last_1h or 0),
        "total_amount_last_1h": float(total_amount_last_1h or 0.0),
        "avg_amount_last_30d": float(avg_amount_last_30d or 0.0) if avg_amount_last_30d else None
    }