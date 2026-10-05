# Real-Time Fraud Detection Platform

A fraud-scoring service that approves or denies transactions in real time. It uses atomic sliding-window features in Redis, a transactional outbox to Kafka, at-least-once consumers with a dead-letter topic, and Prometheus/Grafana observability. It runs as 9 services plus a topic-setup job, all started with one command.

## Architecture

```
            POST /transactions
  Client ──────────────────────►┌──────────────┐  atomic Lua    ┌───────┐
         ◄──────────────────────┤  API         ├───────────────►│ Redis │  1h sliding window
            APPROVE / DENY      │  (FastAPI)   │◄───────────────┤       │  30d avg cache
                                │              │                └───────┘
                                │              │  ONE transaction:
                                │              ├──────────────────────────►┌──────────┐
                                └──────────────┘  transactions + outbox    │ Postgres │
                                                                           └────┬─────┘
                                                      poll unpublished rows     │
                                ┌──────────────┐◄───────────────────────────────┘
                                │ Outbox relay │  acks=all, keyed by user_id
                                └──────┬───────┘
                                       ▼
                         ┌──────────────────────────┐        ┌──────────────────┐
                         │ Kafka: transactions (×6) │───────►│ transactions.dlq │
                         └──────┬────────────┬──────┘        └──────────────────┘
                   group=audit  │            │ group=analytics        ▲
                                ▼            ▼                        │ poison /
                       ┌──────────────┐ ┌──────────────┐              │ exhausted
                       │ audit-writer │ │  analytics   │──────────────┘ retries
                       │ → audit_log  │ │ → Prometheus │
                       └──────────────┘ └──────────────┘

  Prometheus scrapes api, relay, consumers, kafka-exporter → Grafana dashboard
```

### Request path (synchronous, what the client waits for)

1. **Idempotency check.** If `transaction_id` already exists, the stored decision is returned (`"idempotent": true`).
2. **Features.** A Redis Lua script prunes, reads, and records the user's 1-hour window in one atomic step. The 30-day average comes from a cache with a 5-minute TTL (Postgres on a miss). If Redis is down, everything is computed from Postgres instead.
3. **Rules.** More than 5 transactions in the last hour → `DENY too_many_txns_last_1h`. Amount more than 3× the 30-day average → `DENY amount_spike`. Otherwise `APPROVE`.
4. **One Postgres transaction** writes the decision row and an outbox event row.
5. **Return the decision.**

### Event path (asynchronous)

6. The **outbox relay** publishes unpublished rows to Kafka (`acks=all`, keyed by `user_id`), then marks them published.
7. **Consumers** (separate groups, so each one gets every event) commit offsets only after processing. They retry with backoff and send poison messages to `transactions.dlq`.
   - `audit-writer`: append-only `audit_log` table, idempotent insert.
   - `analytics`: decision counts and end-to-end pipeline latency for Prometheus.

## Design decisions

| Decision | Why |
|---|---|
| Lua script for the velocity window | Read-then-write let concurrent requests see the same stale count and all pass the limit. Redis runs a script without interleaving other commands, so each request sees a distinct count. |
| Sorted set per user, member `"{amount}:{txn_id}"`, scored by timestamp | A true sliding window, not a tumbling one. Entries older than an hour are pruned on every access, and the key has a 1-hour TTL so idle users disappear. |
| Transactional outbox instead of publishing from the API | A DB commit and a Kafka send can't be atomic. With an outbox, "decision saved" and "event will be published" commit together. The API also doesn't depend on Kafka at all. |
| Relay uses `acks=all`, `max_in_flight=1` | An event counts as sent only once it's fully replicated, and retries can't reorder it. |
| Manual offset commits after processing | At-least-once delivery: a crash means redelivery, not loss. Consumers are idempotent, so redelivery is safe. |
| DLQ with error headers | One bad message can't block a partition forever. The original bytes are kept for replay. |
| Transient errors retry forever, not DLQ | If Postgres is down, the messages are fine and the dependency isn't. Dead-lettering would dump everything into the DLQ. |
| 6 partitions keyed by `user_id` | Per-user ordering, with up to 6 consumers per group working in parallel. |
| Idempotency on `transaction_id` (primary key) | A client retry, or two concurrent duplicates, gets the one original decision and never a second charge. |
| Redis socket timeout of 0.5s plus Postgres fallback | A dead Redis fails fast and degrades to slower DB queries instead of hanging requests. |

## Failure modes

| Failure | What happens |
|---|---|
| Redis down | Features are computed from Postgres (`fraud_feature_store_fallback_total` goes up). Requests still succeed, but the velocity check is no longer atomic while degraded. |
| Kafka down | The API is unaffected. Events wait in the outbox (`fraud_outbox_pending` grows) and are delivered when Kafka returns. |
| Consumers behind | Kafka buffers on disk. Lag shows as `kafka_consumergroup_lag`. Add consumer instances, up to 6 per group. |
| Same message delivered twice | `audit_log` insert is `ON CONFLICT DO NOTHING`; analytics de-dupes on `event_id`. |
| Same request sent twice | The primary key on `transaction_id` returns the original decision. |
| Poison message | Sent to `transactions.dlq` with error, source partition, and offset headers, then the offset is committed. |
| Relay crashes mid-batch | Rows weren't marked published, so they're re-sent. That's a duplicate, which consumers handle. |

Two of these are tested automatically by stopping containers in CI: Redis down and Kafka down.

## Running it

Requires Docker.

```bash
docker compose up -d --build
```

| Service | URL |
|---|---|
| API (+ Swagger at `/docs`) | http://localhost:8000 |
| Grafana dashboard | http://localhost:3000 |
| Prometheus | http://localhost:9090 |

```bash
curl -X POST http://localhost:8000/transactions \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "txn-001", "user_id": "user-123", "amount": 49.99}'

# Inspect the sliding window
docker compose exec redis redis-cli ZRANGE user:user-123:txns_1h 0 -1 WITHSCORES

# Inspect the audit trail written by the Kafka consumer
docker compose exec postgres psql -U fraud_user -d fraud_db -c "SELECT * FROM audit_log LIMIT 5"
```

## Testing

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt   # .venv\Scripts on Windows

ruff check .
pytest api/tests           # rules, Lua window (incl. concurrency), cache, fallback
pytest consumers/tests     # retry / DLQ / transient-error logic, de-duplication

# Against the running stack:
INTEGRATION=1 pytest tests/integration/test_pipeline.py     # end-to-end, burst, DLQ, partitions
INTEGRATION=1 pytest tests/integration/test_degradation.py  # stops Redis and Kafka
```

CI (`.github/workflows/ci.yml`) runs all of the above, then the load test, on every push.

## Load testing

```bash
k6 run loadtest/k6.js                               # 200 tx/s for 60s
k6 run -e RATE=500 -e DURATION=2m loadtest/k6.js
```

The test uses an open model (`constant-arrival-rate`): requests arrive at a fixed rate whether or not the server keeps up. A run counts as sustained only if `dropped_iterations` is 0 and errors are under 1%. Results are written to `loadtest/results/summary.md`.

### Results

_Not yet measured. Fill in from a real run, and say what hardware it ran on._

| Environment | Sustained rate | p50 | p99 | Errors |
|---|---|---|---|---|
| — | — | — | — | — |

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/transactions` | Score a transaction |
| `GET` | `/transactions/{id}` | Look up a stored decision |
| `GET` | `/health` | Liveness, plus whether the ML model is loaded |
| `GET` | `/metrics` | Prometheus metrics |
| `POST` | `/ml/score-kaggle` | Experimental ML scoring (see below) |

## ML (experimental, not in the decision path)

`ml/train_model.py` trains a logistic regression on the [Kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It uses `class_weight="balanced"`, and the threshold is tuned for at least 85% recall. It's served at `/ml/score-kaggle`. Its inputs are the dataset's anonymized PCA columns (V1–V28) plus Amount. A live `{user_id, amount}` transaction doesn't have those, so the model is kept separate from the rule-based decision path. The next step is retraining on features the live system does compute.

## Project structure

```
api/                  FastAPI service + outbox relay (same image)
  main.py             request handling
  feature_store.py    Redis Lua sliding window, 30d cache, fallback
  features.py         Postgres fallback queries
  rules.py            rule engine
  models.py           transactions + outbox tables
  relay.py            outbox → Kafka publisher
  metrics.py          Prometheus metrics
  ml_router.py        experimental ML endpoint
consumers/            Kafka consumers (one process, one thread each)
  runner.py           manual commits, retries, DLQ
  audit_writer.py     audit_log table
  analytics.py        stream metrics
ml/                   offline training
monitoring/           Prometheus config, Grafana provisioning + dashboard
loadtest/             k6 script
tests/integration/    end-to-end and failure-mode tests
```
