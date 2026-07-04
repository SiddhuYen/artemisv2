#!/bin/sh
set -e

# The mounted volume gives us /data; create the sub-dirs the app writes into
# (SQLite won't create parent directories for its files).
mkdir -p /data/graphs /data/cached_graphs

# ONE worker only — the app keeps in-process state (build lock + per-session
# engine cache) that must not be duplicated across workers. Host provides $PORT.
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}" --workers 1
