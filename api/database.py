from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config import DATABASE_URL

# FastAPI runs sync endpoints in a 40-thread pool; size the connection pool
# so request threads aren't queued behind SQLAlchemy's default of 5.
engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=10, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
