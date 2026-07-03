"""Persistent SQLite-backed cache for search/page/wiki responses.

Keyed by (provider, kind, key). Values are JSON. TTL-checked on read. Survives
across CLI runs (separate sqlite file from the graph DB). Thread-safe — accessed
from the concurrent network phase.

Contract: before any network request, callers check the cache; on hit they
return immediately and never repeat an identical request within the TTL.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Optional

from .. import config
from .stats import STATS

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _connect() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(config.CACHE_DB, check_same_thread=False)
        _conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "  key TEXT PRIMARY KEY,"
            "  kind TEXT,"
            "  value TEXT,"
            "  created_at REAL,"
            "  expires_at REAL"
            ")"
        )
        _conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at)")
        _conn.commit()
    return _conn


def make_key(provider: str, kind: str, ident: str) -> str:
    return f"{provider}::{kind}::{ident}"


def get(key: str, track: bool = True):
    """Return cached JSON value (and count a hit) or None (and count a miss).
    Set track=False for internal lookups that shouldn't skew provider stats."""
    now = time.time()
    with _lock:
        conn = _connect()
        row = conn.execute(
            "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
        ).fetchone()
    if row and row[1] > now:
        if track:
            STATS.hit()
        try:
            return json.loads(row[0])
        except Exception:
            return None
    if track:
        STATS.miss()
    return None


def set(key: str, kind: str, value, ttl: int) -> None:
    now = time.time()
    payload = json.dumps(value)
    with _lock:
        conn = _connect()
        conn.execute(
            "INSERT OR REPLACE INTO cache(key, kind, value, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (key, kind, payload, now, now + ttl),
        )
        conn.commit()


def purge_expired() -> int:
    with _lock:
        conn = _connect()
        cur = conn.execute("DELETE FROM cache WHERE expires_at <= ?", (time.time(),))
        conn.commit()
        return cur.rowcount


# --- persistent counters (e.g. Brave monthly quota, survives across runs) ----
def get_counter(name: str) -> int:
    with _lock:
        conn = _connect()
        row = conn.execute("SELECT value FROM cache WHERE key = ?", (name,)).fetchone()
    if not row:
        return 0
    try:
        return int(json.loads(row[0]).get("n", 0))
    except Exception:
        return 0


def incr_counter(name: str, by: int = 1, ttl: int = 366 * 86400) -> int:
    """Atomically increment a persisted integer counter and return the new value."""
    with _lock:
        conn = _connect()
        row = conn.execute("SELECT value FROM cache WHERE key = ?", (name,)).fetchone()
        n = 0
        if row:
            try:
                n = int(json.loads(row[0]).get("n", 0))
            except Exception:
                n = 0
        n += by
        now = time.time()
        conn.execute(
            "INSERT OR REPLACE INTO cache(key, kind, value, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, "counter", json.dumps({"n": n}), now, now + ttl),
        )
        conn.commit()
        return n
