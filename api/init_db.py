"""Create tables once, before the API workers start.

Running create_all inside every uvicorn worker would race on a fresh
database (several processes issuing CREATE TABLE at the same moment).
"""

import models  # noqa: F401  (registers the tables on Base.metadata)
from database import Base, engine

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("Database tables ready")
