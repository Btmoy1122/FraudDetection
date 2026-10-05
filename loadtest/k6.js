// Load test for POST /transactions.
//
// Uses an open model (constant-arrival-rate): k6 starts RATE requests per
// second regardless of how fast the server answers, which is how real
// traffic behaves.  If the server can't keep up, k6 reports
// dropped_iterations > 0 — a run only counts as "sustained" when that's 0.
//
//   k6 run loadtest/k6.js                         # 200 tx/s for 60s
//   k6 run -e RATE=500 -e DURATION=2m loadtest/k6.js
//
// Env: BASE_URL, RATE, DURATION, USERS (size of the user pool), P99_MS

import http from 'k6/http';
import { check } from 'k6';
import exec from 'k6/execution';
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.2/index.js';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const RATE = parseInt(__ENV.RATE || '200', 10);
const DURATION = __ENV.DURATION || '60s';
const USERS = parseInt(__ENV.USERS || '10000', 10);
const P99_MS = parseInt(__ENV.P99_MS || '500', 10);

export const options = {
  scenarios: {
    steady: {
      executor: 'constant-arrival-rate',
      rate: RATE,
      timeUnit: '1s',
      duration: DURATION,
      preAllocatedVUs: Math.max(50, RATE),
      maxVUs: RATE * 4,
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.01'],
    http_req_duration: [`p(99)<${P99_MS}`],
    dropped_iterations: ['count<1'],
  },
  summaryTrendStats: ['avg', 'p(50)', 'p(95)', 'p(99)', 'max'],
};

export function setup() {
  return { runId: Date.now().toString(36) };
}

export default function (data) {
  // A large user pool keeps the velocity rule from denying most traffic,
  // so the test measures the normal path, not the deny path.
  const userId = `load-user-${Math.floor(Math.random() * USERS)}`;
  const payload = JSON.stringify({
    transaction_id: `load-${data.runId}-${exec.scenario.iterationInTest}`,
    user_id: userId,
    amount: Math.round((5 + Math.random() * 195) * 100) / 100,
  });

  const res = http.post(`${BASE_URL}/transactions`, payload, {
    headers: { 'Content-Type': 'application/json' },
  });
  check(res, {
    'status is 200': (r) => r.status === 200,
    'has decision': (r) => r.status === 200 && r.json('decision') !== undefined,
  });
}

function metric(data, name, stat) {
  const m = data.metrics[name];
  return m && m.values[stat] !== undefined ? m.values[stat] : 0;
}

export function handleSummary(data) {
  const r = {
    target_rate_per_s: RATE,
    duration: DURATION,
    achieved_rate_per_s: metric(data, 'http_reqs', 'rate'),
    requests: metric(data, 'http_reqs', 'count'),
    dropped_iterations: metric(data, 'dropped_iterations', 'count'),
    error_rate: metric(data, 'http_req_failed', 'rate'),
    p50_ms: metric(data, 'http_req_duration', 'p(50)'),
    p95_ms: metric(data, 'http_req_duration', 'p(95)'),
    p99_ms: metric(data, 'http_req_duration', 'p(99)'),
    max_ms: metric(data, 'http_req_duration', 'max'),
  };
  const sustained = r.dropped_iterations === 0 && r.error_rate < 0.01;

  const md = [
    '## Load test results',
    '',
    '| Metric | Value |',
    '|---|---|',
    `| Target rate | ${RATE} tx/s for ${DURATION} |`,
    `| Achieved rate | ${r.achieved_rate_per_s.toFixed(1)} tx/s (${r.requests} requests) |`,
    `| Dropped iterations | ${r.dropped_iterations} |`,
    `| Error rate | ${(r.error_rate * 100).toFixed(2)}% |`,
    `| Latency p50 / p95 / p99 | ${r.p50_ms.toFixed(1)} / ${r.p95_ms.toFixed(1)} / ${r.p99_ms.toFixed(1)} ms |`,
    `| Max latency | ${r.max_ms.toFixed(1)} ms |`,
    `| Sustained? | ${sustained ? 'yes' : 'NO — server could not keep up'} |`,
    '',
  ].join('\n');

  return {
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
    'loadtest/results/summary.json': JSON.stringify(r, null, 2),
    'loadtest/results/summary.md': md,
  };
}
