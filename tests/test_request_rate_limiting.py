"""Regression tests for the "real fix" behind the /connect speedup: a global
cap on concurrent outbound requests (independent of how many logical
node/side workers want to fire at once) plus jitter on retry backoff, so
concurrent failures don't retry in lockstep.

Root cause this addresses: EXPAND_NODE_CONCURRENCY x SEARCH_WORKERS x (2
connect_people sides) can want ~64 requests in flight at once -- far past
what a real provider (Serper: 5 QPS; EDGAR/OpenCorporates/ProPublica: far
less) absorbs. Measured live, that produced a self-inflicted 429 storm where
every failing thread computed the SAME deterministic backoff and retried in
the same instant again, never converging.
"""
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

from app import config
from app.providers import base


class _FakeResponse:
    def __init__(self, status_code=200, headers=None, text="ok"):
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text


class _ConcurrencyTrackingClient:
    """Stands in for the httpx.Client make_client() returns -- records the
    PEAK number of simultaneously-in-flight requests across all threads."""
    def __init__(self, counter, lock, peak, hold_s=0.05, status_code=200):
        self._counter, self._lock, self._peak = counter, lock, peak
        self._hold_s, self._status_code = hold_s, status_code

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def request(self, method, url, **kwargs):
        with self._lock:
            self._counter[0] += 1
            self._peak[0] = max(self._peak[0], self._counter[0])
        time.sleep(self._hold_s)
        with self._lock:
            self._counter[0] -= 1
        return _FakeResponse(self._status_code)


def test_request_with_retry_never_exceeds_the_global_concurrency_cap(monkeypatch):
    # Use a small cap so 10 concurrent callers can actually exceed it if the
    # semaphore isn't wired correctly -- proves the ceiling is enforced,
    # not just "happens not to be hit" at the real default of 10.
    monkeypatch.setattr(base, "_REQUEST_SLOTS", threading.Semaphore(3))

    counter, peak = [0], [0]
    lock = threading.Lock()
    monkeypatch.setattr(
        base, "make_client",
        lambda: _ConcurrencyTrackingClient(counter, lock, peak, hold_s=0.05))

    def call(_i):
        return base.request_with_retry("GET", "http://example.test/x", provider="test")

    with ThreadPoolExecutor(max_workers=10) as ex:
        results = list(ex.map(call, range(10)))

    assert peak[0] <= 3, f"peak concurrent requests was {peak[0]}, cap was 3"
    assert all(r is not None and r.status_code == 200 for r in results)


def test_request_with_retry_releases_its_slot_on_success(monkeypatch):
    """A slot must free up promptly after a call completes -- otherwise the
    cap silently ratchets down over the life of the process."""
    monkeypatch.setattr(base, "_REQUEST_SLOTS", threading.Semaphore(1))
    counter, peak = [0], [0]
    lock = threading.Lock()
    monkeypatch.setattr(
        base, "make_client",
        lambda: _ConcurrencyTrackingClient(counter, lock, peak, hold_s=0.01))

    for _ in range(5):
        result = base.request_with_retry("GET", "http://example.test/x", provider="test")
        assert result is not None and result.status_code == 200
    # If a slot ever leaked, this fifth call would deadlock (never runs) --
    # reaching this line at all is the proof, but assert something concrete too.
    assert peak[0] == 1


def test_sleep_backoff_is_jittered_not_deterministic(monkeypatch):
    slept = []
    fake_time = types.SimpleNamespace(sleep=lambda s: slept.append(s),
                                      monotonic=time.monotonic)
    monkeypatch.setattr(base, "time", fake_time)
    monkeypatch.setattr(config, "HTTP_BACKOFF_JITTER", 1.0)

    for _ in range(25):
        base._sleep_backoff(0)  # attempt 0 -> base delay = HTTP_BACKOFF_BASE

    assert len(slept) == 25
    assert len(set(slept)) > 1, (
        "25 calls at the same attempt number produced identical delays -- "
        "jitter isn't actually varying the sleep, so concurrent failures "
        "would still retry in lockstep"
    )
    lo, hi = config.HTTP_BACKOFF_BASE, config.HTTP_BACKOFF_BASE + 1.0
    assert all(lo <= s < hi for s in slept)


def test_jitter_is_bounded_by_config():
    for _ in range(50):
        j = base._jitter()
        assert 0 <= j < config.HTTP_BACKOFF_JITTER


# --- hard timeout: the DDG-hang fix ----------------------------------------
def test_call_with_hard_timeout_returns_the_result_on_success():
    result = base._call_with_hard_timeout(lambda: 42, timeout=1.0)
    assert result == 42


def test_call_with_hard_timeout_reraises_the_wrapped_exception():
    def boom():
        raise ValueError("real failure")
    with pytest.raises(ValueError, match="real failure"):
        base._call_with_hard_timeout(boom, timeout=1.0)


def test_call_with_hard_timeout_bounds_a_call_that_never_returns():
    """The scenario this exists for: a call stuck in a blocking operation
    (measured live: raw socket.connect() to an unreachable host) that
    httpx's own configured timeout failed to bound. The hard timeout must
    return control to the caller regardless -- the background thread is
    abandoned, not killed (Python can't kill threads), which is why this
    only asserts on TIMING, not on the leaked thread ever stopping."""
    def hangs_forever():
        time.sleep(60)
        return "should never get here"

    start = time.monotonic()
    with pytest.raises(base.HardTimeout):
        base._call_with_hard_timeout(hangs_forever, timeout=0.2)
    elapsed = time.monotonic() - start

    assert elapsed < 2.0, (
        f"hard timeout took {elapsed:.2f}s to fire for a 0.2s deadline -- "
        "it isn't actually bounding the call"
    )


def test_request_with_retry_does_not_hang_when_the_transport_never_returns(monkeypatch):
    """End-to-end version of the above, through the real retry loop: a
    make_client() that hangs forever must not make request_with_retry hang
    -- it must eventually give up and return None (all attempts exhausted),
    same as any other failure."""
    monkeypatch.setattr(config, "HTTP_TIMEOUT", 0.1)
    monkeypatch.setattr(config, "HTTP_HARD_TIMEOUT_BUFFER", 0.1)
    monkeypatch.setattr(config, "HTTP_RETRIES", 1)
    monkeypatch.setattr(config, "HTTP_BACKOFF_BASE", 0.05)
    monkeypatch.setattr(config, "HTTP_BACKOFF_JITTER", 0.05)

    def hanging_request(method, url, **kwargs):
        time.sleep(60)
        raise AssertionError("unreachable")

    monkeypatch.setattr(base, "_do_single_request", hanging_request)

    start = time.monotonic()
    result = base.request_with_retry("GET", "http://example.test/never", provider="test")
    elapsed = time.monotonic() - start

    assert result is None
    assert elapsed < 5.0, (
        f"request_with_retry took {elapsed:.2f}s against a transport that "
        "never returns -- it's still hanging on the underlying call"
    )
