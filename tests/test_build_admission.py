"""Build admission: bounded concurrency, a FIFO queue, and honest refusal.

Builds used to run one at a time behind a process-wide mutex — not because one
at a time was right, but because connect_people mutated a config global
(EXPAND_PREFER_REACHABLE) that a second concurrent build would have corrupted.
That global is now a per-call argument, so these tests pin both halves: the
argument really is per-call, and the queue that replaced the mutex bounds
concurrency without silently dropping or starving anyone.
"""
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app import config
from app.buildqueue import BuildQueue, QueueFull
import app.main as M


# --- the config global is gone ---------------------------------------------
def test_connect_no_longer_mutates_the_expansion_global(monkeypatch):
    """The whole reason builds were serialized. If connect_people ever assigns
    to config.EXPAND_PREFER_REACHABLE again, concurrent builds corrupt each
    other's frontier strategy and this catches it."""
    from app.graph import connect as C

    seen = {}

    def fake_expand_both(db, a, b, depth_a, depth_b, protected, progress, ca, cb, **kwargs):
        seen["during"] = config.EXPAND_PREFER_REACHABLE

    monkeypatch.setattr(config, "EXPAND_PREFER_REACHABLE", True)
    monkeypatch.setattr(C, "_expand_both_concurrently", fake_expand_both)
    monkeypatch.setattr(C, "_adjacency", lambda db, *a: ({}, {}, {}, {}))
    monkeypatch.setattr(C, "_direct_pair_search", lambda *a, **k: (False, False))
    monkeypatch.setattr(C, "_route_exists", lambda *a, **k: False)
    monkeypatch.setattr(C.ORCH, "notable_set", lambda names: set())

    class _Db:
        def execute(self, *_a, **_k):
            class _R:
                @staticmethod
                def scalar_one_or_none():
                    return None
            return _R()

        # connect_people clears its identity map before the final scoring read,
        # so that read builds fresh objects rather than reusing whatever the
        # expansion left mapped (see the comment at that call site). A no-op
        # here: this stub has no session state, and the subject of this test is
        # the config global, not session lifecycle.
        def expunge_all(self):
            pass

    C.connect_people(_Db(), "A", "B", depth=1)

    assert seen["during"] is True, "connect_people must not flip the global"
    assert config.EXPAND_PREFER_REACHABLE is True


def test_connect_requests_strongest_expansion_per_call(monkeypatch):
    """Behavior must be preserved: connect still wants strongest-first
    expansion, it just asks for it per call instead of globally."""
    from app.graph import connect as C

    captured = {}

    def fake_expand_graph(worker_db, name, depth, **kwargs):
        captured[name] = kwargs.get("prefer_reachable")
        return {}

    monkeypatch.setattr(C, "expand_graph", fake_expand_graph)

    class _Sess:
        def close(self): pass

    monkeypatch.setattr(C, "sessionmaker", lambda **kw: (lambda: _Sess()))

    class _Db:
        def get_bind(self): return object()

    C._expand_both_concurrently(_Db(), "Alpha", "Beta", 2, 2, set(), None, "", "")
    assert captured == {"Alpha": False, "Beta": False}


def test_expand_graph_defaults_to_the_configured_mode(monkeypatch):
    """None means 'use the config default' — a caller that passes nothing must
    behave exactly as before."""
    from app.graph import expansion

    monkeypatch.setattr(config, "EXPAND_PREFER_REACHABLE", False)
    monkeypatch.setattr(expansion, "is_filtering_active", lambda: False)

    cand = expansion._Candidate(
        name="Someone", sources={"u"}, confidences=[0.9], max_conf=0.9,
        strong_edges=1, explicit_edges=1, professional_edges=1,
        family_edges=0, trusted=True)
    # prefer_reachable=None -> falls back to config (False) -> no notable_set
    # lookup, which would be a live Wikipedia call.
    out = expansion._ranked_expandable({"someone": cand}, set(), prefer_reachable=None)
    assert out == ["Someone"]


# --- the queue --------------------------------------------------------------
def test_queue_admits_up_to_capacity_then_queues():
    q = BuildQueue(capacity=2, max_queued=5)
    t1, t2, t3 = q.reserve(), q.reserve(), q.reserve()
    q.acquire(t1)
    q.acquire(t2)
    assert q.stats()["running"] == 2
    assert q.position(t3) == 1, "third build waits at the head of the line"


def test_queue_refuses_when_full():
    """A queue longer than anyone will wait through is worse than a refusal."""
    q = BuildQueue(capacity=1, max_queued=2)
    held = [q.reserve() for _ in range(3)]
    q.acquire(held[0])
    with pytest.raises(QueueFull):
        q.reserve()


def test_release_frees_a_slot_for_the_next_in_line():
    q = BuildQueue(capacity=1, max_queued=4)
    t1, t2 = q.reserve(), q.reserve()
    q.acquire(t1)
    assert q.position(t2) == 1

    done = threading.Event()

    def waiter():
        q.acquire(t2)
        done.set()

    threading.Thread(target=waiter, daemon=True).start()
    time.sleep(0.05)
    assert not done.is_set(), "t2 must not start while t1 holds the only slot"
    q.release(t1)
    assert done.wait(2.0), "t2 should start as soon as t1 releases"
    assert q.position(t2) == 0


def test_queue_is_fifo():
    """A bare Semaphore makes no ordering promise, so under sustained load an
    unlucky caller can be passed over indefinitely."""
    q = BuildQueue(capacity=1, max_queued=10)
    first = q.reserve()
    q.acquire(first)
    tickets = [q.reserve() for _ in range(4)]
    order = []
    lock = threading.Lock()

    def waiter(i, t):
        q.acquire(t)
        with lock:
            order.append(i)
        time.sleep(0.02)
        q.release(t)

    threads = [threading.Thread(target=waiter, args=(i, t), daemon=True)
               for i, t in enumerate(tickets)]
    for th in threads:
        th.start()
        time.sleep(0.02)  # stagger so arrival order is unambiguous
    q.release(first)
    for th in threads:
        th.join(3.0)
    assert order == [0, 1, 2, 3]


def test_release_of_a_queued_ticket_removes_it_from_the_line():
    """Cancelling while QUEUED must not leave a ticket holding a place forever."""
    q = BuildQueue(capacity=1, max_queued=4)
    t1, t2, t3 = q.reserve(), q.reserve(), q.reserve()
    q.acquire(t1)
    assert q.position(t3) == 2
    q.release(t2)  # t2's user gave up while waiting
    assert q.position(t3) == 1
    assert q.stats()["queued"] == 1


def test_release_is_idempotent():
    """Workers call it from a plain finally-block without tracking their state."""
    q = BuildQueue(capacity=1, max_queued=1)
    t = q.reserve()
    q.acquire(t)
    q.release(t)
    q.release(t)
    assert q.stats()["running"] == 0


def test_acquire_aborts_when_cancelled_while_queued():
    q = BuildQueue(capacity=1, max_queued=2)
    t1, t2 = q.reserve(), q.reserve()
    q.acquire(t1)

    class Cancelled(Exception):
        pass

    def check():
        raise Cancelled()

    with pytest.raises(Cancelled):
        q.acquire(t2, check_cancel=check)
    q.release(t2)
    assert q.stats()["queued"] == 0


def test_concurrent_reserves_never_exceed_the_cap():
    """reserve() is called from request threads; the bound must hold under race."""
    q = BuildQueue(capacity=2, max_queued=3)
    granted, refused = [], []
    lock = threading.Lock()

    def attempt():
        try:
            t = q.reserve()
            with lock:
                granted.append(t)
        except QueueFull:
            with lock:
                refused.append(1)

    threads = [threading.Thread(target=attempt) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(3.0)

    assert len(granted) == 5, f"capacity(2) + max_queued(3); got {len(granted)}"
    assert len(refused) == 15


# --- the HTTP surface -------------------------------------------------------
@pytest.fixture
def open_client(monkeypatch):
    monkeypatch.setattr(config, "ACCESS_TOKEN", "")
    from app import auth
    auth.build_limiter.reset()
    M.BUILDS.reset()
    monkeypatch.setattr(M.threading, "Thread", lambda **kw: type(
        "T", (), {"start": lambda self: None})())
    with TestClient(M.app) as c:
        yield c
    M.BUILDS.reset()
    auth.build_limiter.reset()


def test_saturated_server_refuses_with_429(open_client, monkeypatch):
    monkeypatch.setattr(M.BUILDS, "_capacity", 1)
    monkeypatch.setattr(M.BUILDS, "_max_queued", 1)
    assert open_client.post("/discover", json={"person_name": "A"}).status_code == 200
    assert open_client.post("/discover", json={"person_name": "B"}).status_code == 200
    res = open_client.post("/discover", json={"person_name": "C"})
    assert res.status_code == 429
    assert "busy" in res.json()["detail"]
    assert res.headers["Retry-After"] == "60"


def test_job_reports_its_place_in_line(open_client, monkeypatch):
    monkeypatch.setattr(M.BUILDS, "_capacity", 1)
    monkeypatch.setattr(M.BUILDS, "_max_queued", 4)
    # Threads are stubbed out, so nothing acquires -- every job sits queued and
    # position is purely a function of arrival order.
    ids = [open_client.post("/discover", json={"person_name": n}).json()["job_id"]
           for n in ("A", "B", "C")]
    positions = [open_client.get(f"/jobs/{j}").json()["queue_position"] for j in ids]
    assert positions == [1, 2, 3]


def test_queued_job_starts_out_queued_not_running(open_client):
    job_id = open_client.post("/discover", json={"person_name": "A"}).json()["job_id"]
    assert open_client.get(f"/jobs/{job_id}").json()["status"] == "queued"


def test_targets_search_is_now_a_background_job(open_client):
    """It used to run the build inside the request — minutes, past every
    managed host's proxy idle timeout."""
    res = open_client.post("/targets/search",
                           json={"target_name": "Ada Lovelace", "max_depth": 1})
    assert res.status_code == 200
    assert "job_id" in res.json()


def test_status_exposes_queue_depth(open_client):
    body = open_client.get("/status").json()
    assert set(body["builds"]) == {"running", "queued", "capacity", "max_queued",
                                   "reserved_for_interactive"}
    assert body["builds"]["capacity"] == config.MAX_CONCURRENT_BUILDS
    # slots an enrichment run may never occupy — see buildqueue's background lane
    assert body["builds"]["reserved_for_interactive"] == \
        config.ENRICH_RESERVED_BUILD_SLOTS
