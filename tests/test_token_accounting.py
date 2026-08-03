"""The Claude ledger must agree with the invoice, and must never break a build.

Two independent contracts, and they pull in opposite directions:

  1. COUNT EVERYTHING BILLED. Not just calls that produced a usable verdict --
     a refusal and a max_tokens truncation are billed responses that return
     None to the caller, and those are exactly the calls an operator wants to
     find when spend and output disagree.

  2. NEVER FAIL A BUILD OVER BOOKKEEPING. Accounting runs after the API call
     already succeeded. A bad usage object, a wedged cache, a model nobody has
     priced -- none of it may raise into a caller that has a real answer in
     hand.
"""
import threading

import pytest

from app.extraction import claude_client, usage


class _NullCache:
    """Stands in for providers.cache so nothing here touches the real
    month-to-date counters an operator reads off /status."""
    @staticmethod
    def incr_counter(*_a, **_k):
        return 0

    @staticmethod
    def get_counter(*_a, **_k):
        return 0


@pytest.fixture(autouse=True)
def _clean_ledger(monkeypatch):
    import app.providers
    usage.reset()
    # usage resolves providers.cache at call time (lazily, to dodge an import
    # cycle), so patching the attribute on the package is what intercepts it.
    monkeypatch.setattr(app.providers, "cache", _NullCache())
    yield
    usage.reset()


class _Usage:
    def __init__(self, inp=1000, out=200, cw=0, cr=0):
        self.input_tokens = inp
        self.output_tokens = out
        self.cache_creation_input_tokens = cw
        self.cache_read_input_tokens = cr


class _Resp:
    """Minimal stand-in for an SDK Message."""
    def __init__(self, text='{"ok": true}', stop_reason="end_turn", usage_obj=None):
        self.stop_reason = stop_reason
        self.usage = usage_obj if usage_obj is not None else _Usage()
        block = type("B", (), {"type": "text", "text": text})()
        self.content = [block]


def _client_returning(resp):
    class _C:
        class messages:
            @staticmethod
            def create(**_kwargs):
                return resp
    return _C()


# --- contract 1: everything the API billed for is counted -------------------
def test_successful_call_is_banked(monkeypatch):
    monkeypatch.setattr(claude_client, "_get_client",
                        lambda: _client_returning(_Resp()))
    assert claude_client.call_json("p", {}, "claude-haiku-4-5") == {"ok": True}
    t = usage.totals()
    assert t["calls"] == 1
    assert t["input_tokens"] == 1000 and t["output_tokens"] == 200


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_billed_non_verdicts_are_counted(monkeypatch, stop_reason):
    """The whole point of measuring. These return None to the caller and cost
    money anyway; a ledger that skipped them would understate spend precisely
    when a prompt is overrunning or tripping a classifier."""
    monkeypatch.setattr(
        claude_client, "_get_client",
        lambda: _client_returning(_Resp(stop_reason=stop_reason)))
    assert claude_client.call_json("p", {}, "claude-haiku-4-5") is None
    assert usage.totals()["calls"] == 1


def test_failed_call_is_not_counted(monkeypatch):
    """A raised exception means no response and no bill -- counting it would
    invent spend, the opposite failure to the one above."""
    class _Boom:
        class messages:
            @staticmethod
            def create(**_kwargs):
                raise RuntimeError("network")
    monkeypatch.setattr(claude_client, "_get_client", lambda: _Boom())
    assert claude_client.call_json("p", {}, "claude-haiku-4-5") is None
    assert usage.totals()["calls"] == 0


# --- contract 2: bookkeeping cannot raise -----------------------------------
def test_missing_usage_object_does_not_break_the_call(monkeypatch):
    resp = _Resp()
    resp.usage = None
    monkeypatch.setattr(claude_client, "_get_client", lambda: _client_returning(resp))
    assert claude_client.call_json("p", {}, "claude-haiku-4-5") == {"ok": True}
    assert usage.totals()["calls"] == 0


def test_garbage_usage_fields_do_not_break_the_call(monkeypatch):
    """An SDK that grows a new shape must not take the pipeline with it."""
    bad = _Usage()
    bad.input_tokens = object()
    monkeypatch.setattr(claude_client, "_get_client",
                        lambda: _client_returning(_Resp(usage_obj=bad)))
    assert claude_client.call_json("p", {}, "claude-haiku-4-5") == {"ok": True}


def test_cache_write_failure_is_swallowed(monkeypatch):
    """The persisted counters are a convenience; a wedged sqlite file must not
    surface as a failed build."""
    class _Wedged:
        @staticmethod
        def incr_counter(*_a, **_k):
            raise OSError("database is locked")

        @staticmethod
        def get_counter(*_a, **_k):
            raise OSError("database is locked")

    import app.providers
    monkeypatch.setattr(app.providers, "cache", _Wedged())
    monkeypatch.setattr(usage, "FLUSH_EVERY_S", -1)   # flush on every record
    usage.record("claude-haiku-4-5", _Usage())
    assert usage.totals()["calls"] == 1
    assert usage.month_to_date() == {}   # unknown, reported as such


# --- cost --------------------------------------------------------------------
def test_cost_uses_published_rates():
    usage.record("claude-sonnet-5", _Usage(inp=1_000_000, out=1_000_000))
    assert usage.totals()["cost_usd"] == pytest.approx(18.0)   # $3 in + $15 out


def test_cache_tokens_are_priced_off_the_input_rate():
    usage.record("claude-haiku-4-5", _Usage(inp=0, out=0, cw=1_000_000, cr=1_000_000))
    # $1/MTok input -> 1.25x on a write, 0.1x on a read
    assert usage.totals()["cost_usd"] == pytest.approx(1.35)


def test_unpriced_model_is_named_not_silently_free():
    """Tokens still count; the cost is a FLOOR and says so. Reporting $0.00 with
    no signal would read as 'this stage is free' rather than 'not priced yet'."""
    usage.record("claude-next-9", _Usage())
    t = usage.totals()
    assert t["calls"] == 1 and t["input_tokens"] == 1000
    assert t["cost_usd"] == 0.0
    assert t["unpriced"] == ["claude-next-9"]


def test_configured_models_are_all_priced():
    """The models this deployment actually calls must be in PRICES. A model
    swapped in via env with no price entry costs real money and reports $0.00
    -- this fails at test time instead of at invoice time."""
    from app import config
    configured = {config.CLAUDE_MODEL, config.CLAUDE_BATCH_MODEL,
                  config.CLAUDE_EXTRACT_MODEL, config.CLAUDE_FILTER_MODEL,
                  config.CLAUDE_CLASSIFY_MODEL}
    assert configured <= set(usage.PRICES), configured - set(usage.PRICES)


# --- windows -----------------------------------------------------------------
def test_since_reports_only_the_window():
    usage.record("claude-haiku-4-5", _Usage())
    mark = usage.checkpoint()
    usage.record("claude-haiku-4-5", _Usage())
    usage.record("claude-sonnet-5", _Usage())
    delta = usage.since(mark)
    assert delta["calls"] == 2
    assert set(delta["by_model"]) == {"claude-haiku-4-5", "claude-sonnet-5"}
    assert usage.totals()["calls"] == 3


def test_window_flags_itself_approximate_when_builds_overlapped():
    """A delta taken while another build was running includes that build's
    tokens. The number is still useful; it just must not claim to be exact."""
    mark = usage.checkpoint()
    assert usage.since(mark, concurrent_builds=0)["approximate"] is False
    assert usage.since(mark, concurrent_builds=2)["approximate"] is True


def test_checkpoint_is_a_copy_not_a_live_view():
    """A shallow reference to the running totals would make every delta zero --
    the mark would move with the counters it is supposed to be measured against."""
    usage.record("claude-haiku-4-5", _Usage())
    mark = usage.checkpoint()
    usage.record("claude-haiku-4-5", _Usage())
    assert usage.since(mark)["calls"] == 1


# --- concurrency -------------------------------------------------------------
def test_records_from_many_threads_are_not_lost():
    """Claude calls run inside nested ThreadPoolExecutors (per-hop nodes, and
    batched stages within a node), so an unsynchronised += would drop records
    under exactly the load worth measuring."""
    def worker():
        for _ in range(200):
            usage.record("claude-haiku-4-5", _Usage(inp=1, out=1))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert usage.totals()["calls"] == 1600


# --- the job surface ---------------------------------------------------------
def test_job_gets_its_spend_however_it_ended():
    """Especially when CANCELLED. _update_job drops status-less writes once a
    job is cancelling, so routing this through it would hide the cost of
    exactly the builds that spent money and returned nothing."""
    import app.main as M
    M._JOBS.clear()
    job_id = M._new_job("connect")
    window = M._UsageWindow()
    usage.record("claude-haiku-4-5", _Usage())
    M.cancel_job(job_id)
    # what the worker's `except JobCancelled` does on its way out
    M._update_job(job_id, status="cancelled", message="cancelled",
                  error=None, result=None)

    M._set_job_usage(job_id, window)

    job = M._get_job(job_id)
    assert job["status"] == "cancelled"
    assert job["claude_usage"]["calls"] == 1
    assert job["claude_usage"]["cost_usd"] > 0


def test_window_measures_from_its_own_start_not_process_start():
    """Tokens spent before the build was admitted -- by a build that ran while
    this one sat in the queue -- are not this job's cost."""
    import app.main as M
    usage.record("claude-sonnet-5", _Usage())   # somebody else's build
    window = M._UsageWindow()
    usage.record("claude-haiku-4-5", _Usage())  # ours
    delta = window.close()
    assert set(delta["by_model"]) == {"claude-haiku-4-5"}


def test_window_close_is_stable_across_repeated_reads():
    """The done path reads it twice -- into the result and onto the job. A
    second live diff would report the tokens a LATER build spent as ours."""
    import app.main as M
    window = M._UsageWindow()
    usage.record("claude-haiku-4-5", _Usage())
    first = window.close()
    usage.record("claude-haiku-4-5", _Usage())
    assert window.close() == first
