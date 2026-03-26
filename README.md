# Real-Time Fraud Detection Platform

A production-style fraud detection system that evaluates financial transactions in real-time, combining rule-based heuristics with machine learning across an event-driven architecture.

## Architecture

```
                          ┌─────────────────────────────────────────────┐
                          │              Docker Compose                 │
                          │                                             │
  ┌────────┐   POST      │  ┌─────────┐  features  ┌───────┐          │
  │ Client ├─────────────►│  │   API   ├───────────►│ Redis │          │
  │        │◄─────────────┤  │(FastAPI)│◄───────────┤       │          │
  └────────┘  decision    │  │         │  update    └───┬───┘          │
                          │  │         │                │ cache miss    │
                          │  │         │  persist   ┌───▼────┐         │
                          │  │         ├───────────►│Postgres│         │
                          │  │         │            └───▲────┘         │
                          │  │         │                │              │
                          │  │         │  publish   ┌───┴────┐         │
                          │  │         ├───────────►│ Kafka  │         │
                          │  └─────────┘            └───┬────┘         │
                          │                             │              │
                          │              ┌──────────────┼──────────┐   │
                          │              │   Consumers  │          │   │
                          │              │              │          │   │
                          │              │  ┌───────────▼───────┐  │   │
                          │              │  │    DB Writer      │──┼───┘
                          │              │  │  (persistence)    │  │
                          │              │  ├───────────────────┤  │
                          │              │  │  Audit Logger     │  │
                          │              │  │  (audit trail)    │  │
                          │              │  ├───────────────────┤  │
                          │              │  │   Analytics       │  │
                          │              │  │ (fraud metrics)   │  │
                          │              │  └───────────────────┘  │
                          │              └─────────────────────────┘
                          └─────────────────────────────────────────────┘
```

### Data Flow

1. **Transaction arrives** at the FastAPI service via `POST /transactions`
2. **Feature lookup** from Redis using sorted-set sliding windows (sub-millisecond). Falls back to PostgreSQL if Redis is unavailable
3. **Rule engine** evaluates features (velocity checks, amount spike detection)
4. **Decision returned** to the client immediately
5. **Redis updated** with new transaction data for future lookups
6. **Event published** to Kafka for async downstream processing
7. **Consumers** independently handle persistence, audit logging, and analytics

### Key Design Decisions

| Decision | Why |
|----------|-----|
| Redis sorted sets for sliding windows | True sliding window (not tumbling). Accurate 1-hour counts regardless of when you query |
| Cache-aside with 5-min TTL for 30-day averages | A 30-day average barely changes per transaction — slight staleness is acceptable |
| Kafka partitioned by user_id | Preserves per-user event ordering across consumers |
| Idempotent DB writer (ON CONFLICT DO NOTHING) | At-least-once delivery + idempotent writes = effectively exactly-once |
| Synchronous DB write + async Kafka publish | Reliability over pure performance — DB is the durability guarantee |

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| API | Python, FastAPI | Transaction scoring, feature reads |
| Feature Store | Redis | Sub-millisecond feature lookups, sliding window counters |
| Event Streaming | Apache Kafka (KRaft) | Async event fan-out to consumers |
| Database | PostgreSQL | Durable transaction storage |
| ML | scikit-learn | Logistic regression fraud classifier |
| Containerization | Docker, Docker Compose | Full-stack orchestration |

## Project Structure

```
├── api/                    # FastAPI transaction scoring service
│   ├── main.py             # API endpoints and request handling
│   ├── feature_store.py    # Redis-backed feature reads/writes
│   ├── redis_client.py     # Redis connection pool
│   ├── kafka_producer.py   # Event publishing to Kafka
│   ├── events.py           # Pydantic event schemas
│   ├── features.py         # PostgreSQL fallback feature computation
│   ├── rules.py            # Rule engine
│   ├── models.py           # SQLAlchemy ORM models
│   ├── database.py         # Database session management
│   ├── config.py           # Environment-based configuration
│   ├── requirements.txt
│   └── Dockerfile
│
├── consumers/              # Kafka consumer workers
│   ├── main.py             # Consumer runner (all consumers in threads)
│   ├── db_writer.py        # Persists events to PostgreSQL
│   ├── audit_logger.py     # Structured audit trail
│   ├── analytics.py        # Running fraud metrics
│   ├── config.py           # Consumer configuration
│   ├── requirements.txt
│   └── Dockerfile
│
├── ml/                     # ML training pipeline
│   ├── train_model.py      # Model training and evaluation
│   ├── artifacts/          # Trained model + metadata
│   │   └── model_v1_meta.json
│   └── requirements.txt
│
├── docker-compose.yml      # Full stack: Postgres, Redis, Kafka, API, Consumers
├── .env.example            # Environment variable template
└── info.md                 # Design notes and interview prep
```

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose

### Run the full stack

```bash
docker-compose up --build
```

This starts PostgreSQL, Redis, Kafka, the API server, and all Kafka consumers.

### Test a transaction

```bash
# Score a transaction
curl -X POST http://localhost:8000/transactions \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "txn-001", "user_id": "user-123", "amount": 49.99}'

# Check health
curl http://localhost:8000/health

# Look up a transaction
curl http://localhost:8000/transactions/txn-001
```

### Verify the pipeline

```bash
# Check Redis has the sliding window data
docker-compose exec redis redis-cli ZRANGE user:user-123:txns_1h 0 -1 WITHSCORES

# Watch Kafka consumer logs
docker-compose logs -f consumers
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health + ML model status |
| `POST` | `/transactions` | Score a transaction (rule engine + features) |
| `GET` | `/transactions/{id}` | Look up a stored transaction |
| `POST` | `/ml/score-kaggle` | Score Kaggle-format features with ML model |

## System Design Concepts

This project demonstrates several concepts commonly discussed in system design interviews:

- **Cache-aside pattern** — check Redis first, query DB on miss, populate cache
- **Sliding windows via sorted sets** — accurate time-windowed aggregates without scheduled jobs
- **Hot/warm/cold data separation** — 1h counters (Redis), 30d averages (cached), full history (PostgreSQL)
- **Event-driven architecture** — Kafka decouples scoring from downstream processing
- **At-least-once + idempotent consumers** — effectively exactly-once without transaction overhead
- **Graceful degradation** — Redis failure falls back to DB; Kafka failure doesn't block the API
- **12-factor configuration** — environment variables, no hardcoded connection strings
