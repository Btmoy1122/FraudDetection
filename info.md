# Fraud Detection Platform — Design Notes

## Phase 1–2: API + ML Integration

### What we built
- FastAPI service with PostgreSQL persistence
- Rule engine using rolling DB aggregates (velocity, amount spikes)
- Logistic regression fraud classifier trained on Kaggle credit card data
- Threshold tuning targeting recall >= 0.85
- Versioned model artifacts (pkl + metadata JSON)

### Core concepts
- **Model**: learned function mapping features -> fraud probability
- **Threshold**: converts probability to binary decision (DENY vs APPROVE)
- **Offline vs Online**: train/tune offline, serve saved model in real-time
- **Idempotent transactions**: duplicate transaction_id returns existing result

### Interview one-liner
"I trained a logistic regression baseline offline on labeled fraud data, tuned the threshold on held-out evaluation data, versioned model artifacts, and served low-latency online inference in FastAPI with explicit feature-contract and threshold control."

---

## Phase 3: Redis Feature Store

### What we built
- Redis sorted sets for true sliding-window counters (1-hour transaction count and sum)
- Cache-aside pattern with TTL for 30-day average amount
- Graceful fallback to PostgreSQL when Redis is unavailable
- Redis pipelines to batch commands in a single network round-trip

### Core concepts
- **Cache-aside pattern**: check cache first, DB on miss, populate cache with TTL
- **Sliding window vs tumbling window**: sorted sets give true sliding windows; simple key TTLs give tumbling windows that reset at arbitrary times
- **TTL (Time To Live)**: cached 30-day average expires after 5 minutes because a 30-day average barely moves per transaction — slight staleness is acceptable
- **Hot vs cold data**: 1-hour counters are hot (Redis), 30-day averages are warm (cached from DB), full history is cold (PostgreSQL)
- **Connection pooling**: reuse TCP connections to Redis instead of opening new ones per request
- **Pipelining**: batch multiple Redis commands into one network round-trip to reduce latency

### Why Redis for real-time features
- In-memory storage: sub-millisecond reads vs ~5-50ms for PostgreSQL queries
- Native data structures (sorted sets, hashes) designed for exactly these patterns
- Single-threaded command execution means no lock contention

### Cache invalidation strategy
- 1-hour counters: self-expiring via sorted set pruning (ZREMRANGEBYSCORE)
- 30-day average: invalidated on every new transaction (DELETE key), re-populated on next cache miss
- Why delete instead of update: computing a new 30-day rolling average requires all 30 days of data that Redis doesn't have

### When cache lies and why it's okay
- The 30-day average can be up to 5 minutes stale
- For fraud detection, a 5-minute-old average that's 0.1% off won't change any rule outcome
- The 1-hour sliding window is always accurate because we write to it synchronously before returning

### Interview one-liner
"I built a Redis feature store using sorted sets for true sliding-window aggregates and cache-aside with TTLs for longer-term features, with graceful PostgreSQL fallback. Hot-path reads dropped from ~50ms database queries to sub-millisecond Redis lookups."

### Interview question: "How do you handle millions of real-time feature lookups per second?"
- Redis sorted sets for sliding-window counters — O(log N) writes, O(log N + M) reads
- Connection pooling and pipelining to minimize network round-trips
- Partition users across Redis instances (cluster mode) for horizontal scaling
- Cache-aside for expensive aggregates with TTL-based invalidation
- Graceful degradation: if Redis fails, fall back to DB queries at lower throughput

---

## Phase 5: Kafka Event Streaming

### What we built
- Kafka producer publishing transaction events after scoring
- Three independent consumers in separate consumer groups:
  - DB writer: persists to PostgreSQL with ON CONFLICT DO NOTHING
  - Audit logger: structured audit trail
  - Analytics: running fraud metrics (denial rate, volume)
- Events keyed by user_id for per-user partition ordering

### Core concepts
- **Event-driven architecture**: scoring is synchronous (user needs an answer), everything else is asynchronous
- **Consumer groups**: each group independently receives every message. DB writer, audit, and analytics all get the same events without coordinating
- **At-least-once delivery**: Kafka guarantees every message is delivered at least once, but may duplicate on consumer failure
- **Idempotent consumers**: ON CONFLICT DO NOTHING makes duplicate processing safe
- **Partition ordering**: keying by user_id ensures all events for one user land on the same partition, preserving per-user ordering
- **Fire-and-forget publishing**: the API doesn't wait for Kafka broker acknowledgement on the hot path

### Why async pipelines exist
- The user only cares about the scoring decision. Persistence, auditing, and analytics don't need to finish before the response
- Async consumers can be scaled independently: if audit logging is slow, add more audit consumers without touching the API
- Backpressure is absorbed by Kafka: if consumers fall behind, messages buffer in Kafka (disk-backed), not in the API's memory

### Backpressure
- Kafka topics retain messages on disk (configurable retention period)
- If consumers are slower than producers, Kafka buffers the difference
- Consumer lag is measurable: you can monitor how far behind each consumer group is
- The API never slows down because of slow consumers

### Event replay
- Kafka retains messages for a configurable period (default 7 days)
- A consumer can reset its offset to re-process old events
- Use case: deploy a new analytics consumer and replay the last week of events to backfill metrics

### Interview one-liner
"I designed an event-driven pipeline using Kafka to decouple real-time transaction scoring from downstream processing. Events are partitioned by user_id for ordering guarantees, consumed by independent consumer groups for persistence, auditing, and analytics, with idempotent writes ensuring at-least-once delivery is safe."

### Interview question: "How would you process events without blocking user requests?"
- Publish events to Kafka asynchronously (fire-and-forget) after returning the scoring decision
- Independent consumer groups handle persistence, audit, and analytics
- Each consumer group can scale horizontally by adding more instances
- Kafka absorbs backpressure: consumers can fall behind without affecting the API
- Idempotent consumers handle duplicate delivery safely
