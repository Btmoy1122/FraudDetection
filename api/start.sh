#!/bin/sh
set -e

python init_db.py

# Prometheus multiprocess mode needs an empty directory on every start;
# stale files from a previous run would be counted again.
if [ -n "$PROMETHEUS_MULTIPROC_DIR" ]; then
    rm -rf "$PROMETHEUS_MULTIPROC_DIR"
    mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
fi

exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers "${API_WORKERS:-1}"
