"""What the Claude calls actually cost. Measured, not estimated from call counts.

Search spend was already visible -- serper_status() reports `used: 3127 of
50000` because every search increments one counter. The Claude side had no
equivalent: `response.usage` was returned on every call and dropped on the
floor, so the expensive half of a build (per-source extraction reads a whole
page per source, on the strong model) was the unmeasured half. A route could be
called cheap or costly on vibes alone.

Two audiences, two shapes:

  - `status()` -- month-to-date, for the operator. Persisted through the same
    cache counters the search quotas use, so it survives a restart and answers
    "what has this deployment spent".
  - `checkpoint()` / `since()` -- a delta around one build, for "what did THIS
    route cost". Process-wide, so a concurrent build lands in both jobs' deltas;
    `since()` reports how many other builds were running so a reader can tell a
    clean measurement from a contaminated one rather than trusting both alike.

Cost is derived here rather than stored, because the tokens are the fact and
the price is a guess that goes stale. PRICES is a snapshot of published
first-party rates; a model missing from it still has its tokens counted and is
named in `unpriced`, so the failure mode is "cost is understated and says so"
rather than a silently wrong number.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, Optional

# providers.cache is imported lazily inside the two functions that touch it,
# NOT at module scope: providers/__init__ pulls in the orchestrator, which
# pulls in wikidata, which imports claude_client -- which imports this module.
# A top-level import here closes that loop and breaks `import app.extraction`
# outright.

# USD per million tokens: (input, output). First-party Anthropic API rates as
# published 2026-06-24. Cache reads bill at 0.1x input, cache writes at 1.25x.
#
# Sonnet 5 is listed at its standard $3/$15 rather than the $2/$10 introductory
# rate: the intro rate expires 2026-08-31, and a table that quietly overstates
# spend for a few weeks is safer than one that understates it forever after.
PRICES: Dict[str, tuple] = {
    "claude-fable-5": (10.0, 50.0),
    "claude-mythos-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

_CACHE_READ_MULT = 0.1
_CACHE_WRITE_MULT = 1.25

_lock = threading.Lock()

# Per-model process totals. Every read and write holds _lock: Claude calls run
# inside two nested ThreadPoolExecutors (per-hop nodes, and batched stages
# within a node), so unsynchronised += here would drop records under load.
_by_model: Dict[str, Dict[str, int]] = {}

# Month-to-date, persisted. Buffered rather than written per call: a build makes
# hundreds of Claude calls, and cache.incr_counter is a SELECT+INSERT+commit
# under a global sqlite lock. Flushed on a timer and on every read, so /status
# is always current and the steady-state cost is ~1 write per model per
# FLUSH_EVERY_S instead of one per call.
FLUSH_EVERY_S = 5.0
_pending: Dict[str, Dict[str, int]] = {}
_last_flush = 0.0

_FIELDS = ("input_tokens", "output_tokens",
           "cache_write_tokens", "cache_read_tokens", "calls")


def _blank() -> Dict[str, int]:
    return {f: 0 for f in _FIELDS}


def _current_month() -> str:
    return time.strftime("%Y-%m", time.gmtime())


def _counter_key(field: str) -> str:
    return f"claudeusage::{_current_month()}::{field}"


def record(model: str, usage) -> None:
    """Bank one call's token counts. Never raises -- accounting must not be
    able to fail a build that already succeeded.

    Called for every response the API returned, including refusals and
    `max_tokens` truncations. Those produce no usable verdict but they were
    billed, and a ledger that only counts useful calls understates the thing it
    exists to measure.
    """
    if usage is None:
        return
    try:
        row = {
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "cache_write_tokens": int(
                getattr(usage, "cache_creation_input_tokens", 0) or 0),
            "cache_read_tokens": int(
                getattr(usage, "cache_read_input_tokens", 0) or 0),
            "calls": 1,
        }
    except (TypeError, ValueError):
        return

    with _lock:
        target = _by_model.setdefault(model or "unknown", _blank())
        buffered = _pending.setdefault(model or "unknown", _blank())
        for field, n in row.items():
            target[field] += n
            buffered[field] += n
        _maybe_flush_locked()


def _maybe_flush_locked(force: bool = False) -> None:
    """Drain `_pending` into the persisted monthly counters. Caller holds _lock.

    Per-model keys would need an index of which models a month has seen in
    order to be read back; the persisted view is deliberately aggregate-only
    (tokens and a cost computed at record time, when the model IS known) and
    the per-model breakdown lives in the process totals above.
    """
    global _last_flush
    now = time.time()
    if not force and now - _last_flush < FLUSH_EVERY_S:
        return
    _last_flush = now
    if not _pending:
        return
    totals = _blank()
    micros = 0
    for model, row in _pending.items():
        for field in _FIELDS:
            totals[field] += row[field]
        micros += int(round(_cost(model, row) * 1_000_000))
    _pending.clear()
    try:
        from ..providers import cache
        for field in _FIELDS:
            if totals[field]:
                cache.incr_counter(_counter_key(field), totals[field])
        if micros:
            cache.incr_counter(_counter_key("cost_micro_usd"), micros)
    except Exception:  # noqa: BLE001 -- a cache write must never break a build
        pass


def _cost(model: str, row: Dict[str, int]) -> float:
    price = PRICES.get(model)
    if not price:
        return 0.0
    in_rate, out_rate = price
    return (
        row["input_tokens"] * in_rate
        + row["output_tokens"] * out_rate
        + row["cache_write_tokens"] * in_rate * _CACHE_WRITE_MULT
        + row["cache_read_tokens"] * in_rate * _CACHE_READ_MULT
    ) / 1_000_000.0


def _summarise(by_model: Dict[str, Dict[str, int]]) -> dict:
    totals = _blank()
    models = {}
    unpriced = []
    cost = 0.0
    for model, row in by_model.items():
        for field in _FIELDS:
            totals[field] += row[field]
        c = _cost(model, row)
        cost += c
        if model not in PRICES and any(row[f] for f in _FIELDS):
            unpriced.append(model)
        models[model] = dict(row, cost_usd=round(c, 6))
    out = dict(totals)
    out["cost_usd"] = round(cost, 6)
    out["by_model"] = models
    # Named, not silently zero-costed: the total is a floor when this is set.
    out["unpriced"] = sorted(unpriced)
    return out


def totals() -> dict:
    """Everything this process has spent since it started."""
    with _lock:
        snapshot = {m: dict(r) for m, r in _by_model.items()}
    return _summarise(snapshot)


def month_to_date() -> dict:
    """The persisted view: survives restarts, aggregate only (no per-model)."""
    with _lock:
        _maybe_flush_locked(force=True)
    try:
        from ..providers import cache
        row = {f: cache.get_counter(_counter_key(f)) for f in _FIELDS}
        micros = cache.get_counter(_counter_key("cost_micro_usd"))
    except Exception:  # noqa: BLE001
        return {}
    row["cost_usd"] = round(micros / 1_000_000.0, 6)
    row["month"] = _current_month()
    return row


def status() -> dict:
    """For /status, beside the search quotas."""
    return {"month_to_date": month_to_date(), "process": totals()}


def checkpoint() -> dict:
    """An opaque marker to diff against later. Cheap; take one per build."""
    with _lock:
        return {m: dict(r) for m, r in _by_model.items()}


def since(mark: Optional[dict], concurrent_builds: int = 0) -> dict:
    """Tokens banked since `mark`.

    Process-wide, so any OTHER build running in the same window is included
    here too. `concurrent_builds` carries that fact through to the reader
    instead of leaving a contaminated number looking like a clean one -- 0
    means this job had the server to itself at every point it checked.
    """
    if mark is None:
        mark = {}
    with _lock:
        now = {m: dict(r) for m, r in _by_model.items()}
    delta: Dict[str, Dict[str, int]] = {}
    for model, row in now.items():
        before = mark.get(model) or _blank()
        diff = {f: row[f] - before.get(f, 0) for f in _FIELDS}
        if any(diff.values()):
            delta[model] = diff
    out = _summarise(delta)
    out["concurrent_builds"] = concurrent_builds
    # False means every token below was spent by this build and nothing else.
    out["approximate"] = concurrent_builds > 0
    return out


def reset() -> None:
    """Process totals only -- for tests. Does not touch persisted counters."""
    global _last_flush
    with _lock:
        _by_model.clear()
        _pending.clear()
        _last_flush = 0.0
