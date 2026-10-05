# Fraud Detection Platform: Study Guide

Read this before anything goes on your resume. Every section is something an interviewer can push on. If you can't explain a part out loud without looking, re-read the code it points to.

---

## 1. The full data flow (be able to draw this)

```
Client → API ──(Lua)──► Redis
           └──(1 txn)─► Postgres [transactions + outbox]
                              ▲
              Outbox relay ───┘ polls, publishes (acks=all)
                   ▼
           Kafka "transactions" (6 partitions, key=user_id)
              ├─ group "audit"     → audit-writer → audit_log table
              └─ group "analytics" → analytics    → Prometheus metrics
           anything unprocessable → "transactions.dlq"
```

**Synchronous (client waits), in `api/main.py`:**
1. `db.get(transaction_id)`. If it exists, return the stored decision (idempotency).
2. `get_features()` in `feature_store.py`:
   - Lua script: prune old entries, count and sum the window, add this transaction. One atomic step.
   - 30-day average from the Redis cache, or Postgres on a miss (cached for 5 minutes).
   - If Redis errors at all, compute everything from Postgres.
3. `apply_rules()`: 5 or more prior transactions in the last hour → DENY. Amount more than 3× the 30-day average → DENY.
4. Insert the `transactions` row and the `outbox` row, then a single `commit()`.
5. Return the decision.

**Asynchronous:**
6. `relay.py` selects unpublished outbox rows, sends them to Kafka, waits for the acks, and marks them published.
7. `consumers/` read from Kafka, process, then commit offsets.

**One-liner:** "The API makes the decision synchronously and writes it, plus an outbox event, in one Postgres transaction. A relay publishes those events to Kafka, where independent consumer groups handle audit and analytics with at-least-once delivery."

---

## 2. Why Kafka instead of calling things directly?

What it buys you:
- **Decoupling.** The API doesn't know or care who consumes events. Adding a new consumer (e.g. an ML feature pipeline) needs zero API changes.
- **Fan-out.** Each consumer group gets every event independently. Audit and analytics never coordinate.
- **Buffering / backpressure.** If consumers are slow, events pile up on Kafka's disk, not in API memory. The API never slows down.
- **Replay.** Events are retained (7 days by default). A new consumer can start from the beginning and backfill.
- **Ordering per key.** Keyed by `user_id`, so one user's events always land on the same partition, in order.

The honest trade-off: more moving parts, eventual consistency (the audit log lags slightly), and you have to think about duplicates.

---

## 3. The Redis sliding window

**Key:** `user:{user_id}:txns_1h`, a **sorted set**.
**Member:** `"{amount}:{transaction_id}"`. Amount comes first so a `:` inside a transaction ID can't break parsing.
**Score:** Unix timestamp of the transaction.

**What the Lua script does** (`feature_store.py`, `_RECORD_AND_READ_LUA`):
1. `ZREMRANGEBYSCORE key -inf (now - 3600)`: delete entries older than 1 hour.
2. `ZRANGE key 0 -1`: read what's left, then count and sum the amounts, **skipping this transaction's own member** so a retry doesn't count itself.
3. `ZADD key NX now member`: record this transaction. NX means a retry doesn't move its timestamp.
4. `EXPIRE key 3600`: if the user goes quiet, the whole key disappears.
5. Return `{count, total}` as it was **before** this transaction.

**How entries expire:** two ways. Each entry is pruned the next time that user transacts (step 1), and the whole key expires after an hour of inactivity (step 4).

**Sliding vs tumbling:** a tumbling window (e.g. `INCR` with a 1h TTL) resets at arbitrary times, so 5 transactions at 12:59 and 5 at 1:01 both pass. A sorted set gives the exact last 60 minutes at any moment.

**Complexity:** ZADD is O(log N). The range read is O(log N + M). N is tiny (bounded by the velocity limit).

### The race condition this fixes (whiteboard this)

The old code did: **read** the count → **decide** → **write** the new entry, as separate steps.

```
Request A (user U)          Request B (user U)
read count = 4
                            read count = 4
decide: 4 < 5 → APPROVE
                            decide: 4 < 5 → APPROVE
write
                            write          ← user now has 6, both approved
```

Fire 20 at once and many get through. **The fix:** Redis is single-threaded and runs a Lua script start to finish with no other command in between. Read and write become one indivisible operation, so 20 concurrent requests see counts 0, 1, 2 … 19, each exactly once. Exactly 5 are approved.

**Proven by:**
- `api/tests/test_feature_store.py::test_concurrent_requests_each_see_distinct_counts` (unit test)
- `tests/integration/test_pipeline.py::test_velocity_limit_holds_under_concurrent_burst` (20 real HTTP requests, asserts exactly 5 APPROVE)

**Why Lua instead of MULTI/EXEC?** MULTI/EXEC batches commands atomically, but you can't read a value and branch on it inside the transaction. WATCH plus a retry loop works but is clunkier. Lua lets you read and write in one atomic unit.

**Caveat to say out loud:** when Redis is down, the Postgres fallback is NOT atomic, so a burst during an outage could slip through. Fixing that would need a DB-level lock (e.g. `SELECT … FOR UPDATE` on a per-user row) or failing closed.

**Why denied transactions still count:** velocity counts *attempts*. A fraudster hammering a card should keep getting denied.

### 30-day average cache

Cache-aside: check Redis, on a miss query Postgres, and store the result with a 5-minute TTL. It is **not** invalidated on every transaction (the old code did that, which made the cache nearly useless for active users). A 30-day average barely moves per transaction, so up to 5 minutes of staleness is fine.

---

## 4. Failure modes ("what happens if…")

**Redis goes down?**
The Redis client has a 0.5s socket timeout, so the call fails fast with a `RedisError`. `get_features` catches it and computes features from Postgres (one aggregate query on the `(user_id, created_at)` index). The `fraud_feature_store_fallback_total` metric goes up. Requests are slower but still succeed. When Redis comes back, the client reconnects automatically. The Lua script was lost with Redis's memory, so the script object gets `NOSCRIPT` and re-sends it. The window starts empty, which is a known gap: velocity under-counts for up to an hour after a Redis restart.
*Tested:* `tests/integration/test_degradation.py::test_api_keeps_scoring_when_redis_is_down`.

**Kafka goes down?**
The API doesn't notice at all, because it only writes to Postgres. The relay's sends fail, its DB transaction rolls back, the rows stay unpublished, and it retries every second. `fraud_outbox_pending` climbs. When Kafka is back, the relay drains the backlog in order.
*Tested:* `test_degradation.py::test_events_survive_kafka_outage` stops Kafka, posts a transaction, restarts Kafka, and checks the event reaches `audit_log`.

**Consumers fall behind?**
Messages wait on Kafka's disk. Lag is visible as `kafka_consumergroup_lag` in Grafana. To catch up, run more consumer instances. With 6 partitions, up to 6 consumers per group can work in parallel. A 7th would sit idle, because a partition is owned by exactly one consumer in a group. The API is unaffected either way.

**The same message is processed twice?**
That's expected with at-least-once delivery. It happens when a consumer crashes after processing but before committing, or when the relay crashes after sending but before marking rows published. It's handled by idempotency:
- `audit_log`: `INSERT … ON CONFLICT (transaction_id) DO NOTHING`
- analytics: a bounded set of recently seen `event_id`s (in memory, so it resets on restart; fine for metrics)

**The same *request* arrives twice** (client retry)?
`transaction_id` is the primary key. The second request finds the first one's row and returns the original decision. If two duplicates arrive at the same moment, both pass the initial check, but only one INSERT can win. The loser gets `IntegrityError`, rolls back (including its outbox row), and returns the winner's decision.
*Tested:* `test_concurrent_duplicates_get_one_decision`.

**A message can't be processed (poison)?**
`consumers/runner.py`:
- Not valid JSON → straight to the DLQ.
- Missing fields (`NonRetryableError`) → straight to the DLQ.
- Transient error (Postgres down, `OperationalError`) → retry forever with capped exponential backoff. Dead-lettering here would dump *every* message into the DLQ during an outage.
- Any other error → 3 attempts, then the DLQ.

DLQ messages keep the original bytes, plus headers for the error, consumer, source partition, and source offset, so they can be inspected and replayed. The offset is committed only after the DLQ write is acknowledged.
*Tested:* `consumers/tests/test_runner.py` (all branches) and `test_poison_message_is_dead_lettered` (real Kafka).

**Postgres goes down?**
The API returns 500. Postgres is the source of truth, so it fails closed rather than approving blind. The Redis window entry for that request is removed (`forget_transaction`) so the failed attempt doesn't count toward velocity.

---

## 5. The transactional outbox (strong fintech talking point)

**The problem (dual write):** the API has to (a) save the decision and (b) publish an event. Those are two different systems, so you can't commit both atomically.
- Save, then publish: crash in between and the event is lost forever.
- Publish, then save: crash in between and consumers see a decision that doesn't exist.

The old code saved, then did a fire-and-forget publish. Kafka down meant events silently lost.

**The fix:** write the event into an `outbox` table **in the same Postgres transaction** as the decision. Either both commit or neither does. A separate relay process reads the outbox and publishes to Kafka.

**Relay details (`api/relay.py`):**
- `SELECT … WHERE published_at IS NULL ORDER BY id LIMIT 500 FOR UPDATE SKIP LOCKED`
- `acks="all"`: the broker confirms only after all in-sync replicas have the message.
- `max_in_flight_requests_per_connection=1`: with retries, more than one in-flight request could reorder messages.
- Waits on every send future, then marks the rows published in the same DB transaction.
- A crash after sending but before the commit means the batch is re-sent. That's at-least-once, so consumers must be idempotent (and they are).
- A partial index (`WHERE published_at IS NULL`) keeps the poll query fast. Published rows are deleted after an hour.

**Trade-offs to mention:** it adds up to ~100ms of event latency (the poll interval). Postgres `LISTEN/NOTIFY` or CDC with Debezium reading the WAL would cut that. Run exactly one relay: two would grab different batches and could publish one user's events out of order.

**Bonus:** the API no longer depends on Kafka at all. Kafka could be down for an hour and users would never know.

---

## 6. Kafka partitions and consumer groups

- `transactions` has **6 partitions**, created explicitly by the `kafka-init` service. Auto-creation is disabled so nothing silently gets 1 partition.
- **The key is `user_id`.** Kafka hashes the key to choose a partition, so one user's events always go to the same partition, in order.
- **Ordering is per partition, not global.** That's fine here: we only need each user's events in order.
- **Consumer group:** the partitions are divided among the group's members, and each partition is read by exactly one member. Different groups each get everything. That's how audit and analytics both see every event.
- **Scaling limit:** max parallelism per group equals the partition count (6).
- **Manual commits:** `enable_auto_commit=False`, and `consumer.commit()` runs only after the whole polled batch is handled. Auto-commit can commit an offset *before* processing finishes, so a crash would lose that message.
- **Caveat:** a dead-lettered message is skipped, so per-user order has a gap there. And if a transient retry blocks longer than `max.poll.interval.ms` (5 minutes), the group rebalances, the commit fails, the consumer crashes, Docker restarts it, and it reprocesses from the last commit. That's safe because of idempotency.

---

## 7. Observability

Metrics (Prometheus, scraped every 5s). The Grafana dashboard is at `localhost:3000`.

| Metric | Source | Why it matters |
|---|---|---|
| `fraud_decision_latency_seconds` (histogram) | API | p50/p99 of the request path |
| `fraud_decisions_total{decision,reason}` | API | throughput, deny rate |
| `fraud_feature_store_fallback_total` | API | is Redis healthy? |
| `fraud_idempotent_replays_total` | API | how often clients retry |
| `fraud_outbox_pending`, `fraud_outbox_oldest_pending_age_seconds` | relay | is publishing keeping up / is Kafka down? |
| `kafka_consumergroup_lag` | kafka-exporter | are consumers keeping up? |
| `fraud_event_pipeline_latency_seconds` | analytics | decision → consumer, end to end |
| `fraud_consumer_events_total{consumer,outcome}` | consumers | throughput, DLQ rate |

Why a histogram for latency: averages hide tail latency, and p99 is what users feel. Histograms can also be aggregated across instances. Precomputed quantiles (summaries) can't.

---

## 8. Testing and CI

- **Unit tests (36):** rules, the Lua window (pruning, isolation, retries, concurrency), the cache, the fallback, every retry/DLQ branch of the runner, analytics de-duplication. Redis is simulated with `fakeredis`, which runs real Lua.
- **Integration tests (10):** against the real compose stack. End to end through Kafka into `audit_log`, the burst/velocity race, concurrent duplicates, the DLQ, the partition count, metrics, validation.
- **Failure-mode tests (2):** stop Redis, and stop Kafka, using `docker compose stop`.
- **CI** (`.github/workflows/ci.yml`): ruff lint → unit tests → build images → start the stack → integration → failure modes → k6 load test → results posted to the job summary.

---

## 9. Load testing

`loadtest/k6.js` uses an **open model** (`constant-arrival-rate`): k6 sends N requests per second no matter how slowly the server answers. A closed model (fixed users who wait for each response) hides overload, because a slow server automatically gets fewer requests. That's called "coordinated omission."

A run is "sustained" only if `dropped_iterations == 0` and errors are under 1%.

**Measured results** (GitHub Actions runner, 4 vCPU, whole stack plus k6 on one machine):

| Workers | Target | Achieved | Dropped | p50 | p99 | Errors |
|---|---|---|---|---|---|---|
| 1 | 100 | 100 | 0 | 4.5 ms | 13.7 ms | 0% |
| 1 | 200 | 200 | 0 | 3.7 ms | 396 ms | 0% |
| 1 | 500 | 324 | 8,974 | 6.0 s | 6.7 s | 0% |
| 1 | 500 | 200 | 16,432 | 9.9 s | 14.2 s | 0% |
| 4 | 500 | 500 | 0 | 184 ms | 1.01 s | 0% |

**The story (measure → hypothesis → change → re-measure):**
1. **Measure.** One worker sustained 200 tx/s, but at a 500 target it saturated around 200–325 tx/s, with multi-second latency.
2. **Hypothesis.** Python's GIL means one process uses one core, so the single uvicorn worker is the bottleneck.
3. **Change.** Run 4 workers. That required:
   - Prometheus multiprocess mode, so metrics from every worker are aggregated rather than showing only whichever worker answered the scrape.
   - Moving `CREATE TABLE` into a one-time init step, so 4 processes don't race on a fresh database.
   - Shrinking the per-worker DB pool, since 4 × 30 connections would exceed Postgres's default of 100.
4. **Re-measure.** At the same 500 tx/s target, achieved throughput went from ~200 to 500 tx/s (~2.5×). Dropped requests went from 16k to 0, and errors stayed at 0.

**How to explain the numbers:**
- *Why does p99 jump at 200 (1 worker) while p50 stays flat?* Queueing. Bursts briefly exceed what one worker can serve, and only the requests that arrive during a burst wait.
- *How do you know 1 worker's ceiling?* At 500, k6 hit its 2,000-VU cap. Little's Law (L = λW): λ = 2,000 / 6.2 s ≈ 323 tx/s, which matches the measured 324.
- *Why do the two 1-worker runs differ (324 vs 200)?* Shared CI machines are noisy neighbors. That's why I compare 1 vs 4 workers at the same rate, and quote a range, not the best number.
- *Why not 4× with 4 workers?* All 10 containers and k6 share 4 vCPUs. At 500 tx/s, p50 is already 184 ms, so the machine's CPU is the next limit. In production the API is stateless, so you'd add replicas on separate hosts behind a load balancer.
- *Zero errors even when saturated?* Overload showed up as queueing, not failures. The next step would be load shedding (return 503 early when the queue is deep) so latency stays bounded.

**Never put a number on your resume that you didn't measure.** Run it, record the hardware, and use the real numbers. Beyond 4 workers on one box, you'd scale with more API replicas on separate hosts behind a load balancer (the API is stateless), PgBouncer in front of Postgres, and async DB drivers.

---

## 10. ML (honest framing)

- A logistic regression with `class_weight="balanced"` (fraud is ~0.17% of the data), trained offline on the Kaggle credit card dataset.
- The threshold is chosen for the best precision while keeping recall at or above 85% (missing fraud costs more than a false alarm).
- **It's not in the decision path.** Its inputs (V1–V28) are anonymized PCA components from that dataset, which a live transaction doesn't have. It's served separately at `/ml/score-kaggle`.
- **If asked:** "The production decision path is rules-based on features I compute in real time. I trained a baseline model offline, but its features don't exist at serving time, so I kept it out of the decision path rather than fake it. Next step is training on the features the system actually computes (velocity, amount vs. average) and blending its score with the rules."

---

## 11. Known limitations / what I'd do next

- The velocity check isn't atomic while Redis is down (Postgres fallback).
- The Redis window is in memory; a Redis restart loses up to 1 hour of velocity history (could rebuild it from Postgres on a miss).
- A single relay is a throughput ceiling; it could be partitioned by key hash or replaced with Debezium CDC.
- One API container (4 workers) on one host; production would run several replicas behind a load balancer, with no load shedding yet.
- No authentication or rate limiting on the API.
- No schema registry; the event contract lives in Pydantic models.
- There's no DLQ replay tool yet (messages are kept with their headers, so it's straightforward to add).

---

## Resume bullets (fill numbers in only after measuring)

- Built a real-time fraud scoring service (FastAPI, Redis, Kafka, Postgres) with atomic Redis Lua sliding-window velocity checks, fixing a race that let concurrent requests bypass limits.
- Guaranteed event delivery with a transactional outbox and at-least-once Kafka consumers (manual commits, idempotent writes, dead-letter topic), verified by failure-injection tests that stop Redis and Kafka in CI.
- Load-tested with k6 and found a single-process (GIL) bottleneck capping throughput at ~200–325 tx/s. Moved to multi-worker serving, with aggregated Prometheus metrics and race-free schema init, and reached 500 tx/s with zero errors or dropped requests (~2.5× on the same hardware).
- Containerized 9 services with Docker Compose; GitHub Actions CI runs lint, 36 unit tests, integration tests, and a load test on every push.
