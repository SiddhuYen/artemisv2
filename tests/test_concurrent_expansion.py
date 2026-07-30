"""Regression tests for parallelizing (a) frontier nodes within a hop and
(b) the two connect_people sides -- the two speedups requested on top of the
trace-route fix.

These need REAL separate DB connections (not the shared in-memory `db`
fixture from conftest.py, which pools every session onto one physical
sqlite3 connection -- fine for single-threaded tests, unsafe to actually
exercise from multiple threads at once). Each test here builds its own
file-backed engine via app.db._make_engine, mirroring exactly how expand_graph
and connect_people derive their own worker sessions in production
(`sessionmaker(bind=db.get_bind())`).
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy.orm import sessionmaker

from app import config
from app.db import Base, _make_engine
from app.extraction.schemas import EdgeSignals, ExtractedEdge
from app.graph import builder, connect as C, expansion
from app.models import Organization, Person


def _engine(tmp_path, name):
    eng = _make_engine(f"sqlite:///{tmp_path / name}")
    Base.metadata.create_all(eng)
    return eng


def _sessionmaker(engine):
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


# --- _merge_disc: pure accumulation logic, no threading needed -------------
def test_merge_disc_combines_overlapping_and_disjoint_candidates():
    into = {}
    a = expansion._Candidate(name="Shared", sources={"u1"}, confidences=[0.5],
                             max_conf=0.5, strong_edges=1, explicit_edges=1,
                             professional_edges=1, family_edges=0, trusted=False)
    expansion._merge_disc(into, {"shared": a})

    b = expansion._Candidate(name="Shared", sources={"u2"}, confidences=[0.9],
                             max_conf=0.9, strong_edges=0, explicit_edges=0,
                             professional_edges=0, family_edges=1, trusted=True)
    only_in_other = expansion._Candidate(name="Solo", sources={"u3"}, confidences=[0.4],
                                         max_conf=0.4, strong_edges=0, explicit_edges=0,
                                         professional_edges=1, family_edges=0, trusted=False)
    expansion._merge_disc(into, {"shared": b, "solo": only_in_other})

    merged = into["shared"]
    assert merged.sources == {"u1", "u2"}
    assert merged.confidences == [0.5, 0.9]
    assert merged.max_conf == 0.9
    assert merged.strong_edges == 1
    assert merged.explicit_edges == 1
    assert merged.professional_edges == 1
    assert merged.family_edges == 1
    assert merged.trusted is True  # either side trusted -> merged trusted
    assert "solo" in into and into["solo"] is only_in_other


# --- builder races: norm_name is unique; concurrent creators must collapse -
def test_concurrent_get_or_create_person_never_duplicates(tmp_path):
    engine = _engine(tmp_path, "person_race.db")
    Session = _sessionmaker(engine)

    def worker(_i):
        s = Session()
        try:
            p = builder.get_or_create_person(s, "Race Condition Person")
            s.commit()
            return p.id
        finally:
            s.close()

    with ThreadPoolExecutor(max_workers=12) as ex:
        ids = list(ex.map(worker, range(12)))

    assert len(set(ids)) == 1, "every concurrent caller must resolve to the SAME person row"

    verify = Session()
    try:
        rows = verify.query(Person).filter_by(norm_name="race condition person").all()
        assert len(rows) == 1
    finally:
        verify.close()


def test_concurrent_get_or_create_org_never_duplicates(tmp_path):
    engine = _engine(tmp_path, "org_race.db")
    Session = _sessionmaker(engine)

    def worker(_i):
        s = Session()
        try:
            o = builder.get_or_create_org(s, "Race Condition Org")
            s.commit()
            return o.id
        finally:
            s.close()

    with ThreadPoolExecutor(max_workers=12) as ex:
        ids = list(ex.map(worker, range(12)))

    assert len(set(ids)) == 1

    verify = Session()
    try:
        rows = verify.query(Organization).filter_by(norm_name="race condition org").all()
        assert len(rows) == 1
    finally:
        verify.close()


# --- expand_graph: hop-1 frontier nodes actually run concurrently ----------
def test_expand_graph_runs_hop_nodes_concurrently_and_skips_a_failure(tmp_path, monkeypatch):
    # Keep _ranked_expandable's frontier selection hermetic: EXPAND_PREFER_REACHABLE
    # (on by default) calls ORCH.notable_set() -- a real Wikipedia lookup -- and
    # an untrusted candidate goes through the real Claude entity filter. Neither
    # belongs in a test of the CONCURRENCY mechanism; force the plain
    # highest-score ranking path and mark candidates trusted (structured-source
    # equivalent) so both are skipped, same as e.g. wikidata-colleagues edges.
    monkeypatch.setattr(config, "EXPAND_PREFER_REACHABLE", False)

    engine = _engine(tmp_path, "hop_concurrency.db")
    Session = _sessionmaker(engine)
    root = Session()
    root.close()  # expand_graph only needs `db.get_bind()`; it opens its own workers

    intervals = []
    lock = threading.Lock()
    CANDIDATES = [f"Candidate {i}" for i in range(5)]

    def fake_process_person(worker_db, name, hop, local_disc, progress=None,
                            is_person=True, context=""):
        if hop == 0:
            subject = builder.get_or_create_person(worker_db, name)
            subject.processed = 1
            # hop 0 discovers 5 distinct, clearly-expandable candidates
            for cand in CANDIDATES:
                edge = ExtractedEdge(
                    person_a=name, person_b=cand, other_kind="person",
                    relationship_type="coworker", confidence_adjusted=0.8,
                    confidence_base=0.5, source_url=f"http://x/{cand}",
                    signals=EdgeSignals(explicit_keyword_match=True, trusted=True))
                expansion._record(local_disc, edge)
            worker_db.commit()
            return
        if name == "Candidate 2":
            raise RuntimeError("simulated network failure")
        # Simulate the SLOW part (network searches + page fetches in the real
        # _process_person) happening BEFORE any DB write -- exactly how the
        # real function is structured (phases 0-4 are pure network/extraction;
        # DB writes only start in phase 5, right before the single commit at
        # the end). Sleeping AFTER a flush instead would hold SQLite's
        # single-writer lock through the sleep and artificially serialize
        # every worker on that lock -- a bug in a fake, not in the code under
        # test, but it would make this assertion fail for the wrong reason.
        start = time.monotonic()
        time.sleep(0.08)  # hold the slot long enough for overlap to be observable
        with lock:
            intervals.append((name, start, time.monotonic()))
        subject = builder.get_or_create_person(worker_db, name)
        subject.processed = 1
        worker_db.commit()

    monkeypatch.setattr(expansion, "_process_person", fake_process_person)

    root2 = Session()
    try:
        stats = expansion.expand_graph(root2, "Seed Person", max_depth=2)
    finally:
        root2.close()

    # hop 0 processed 1 node (the seed); hop 1 processed all 5 candidates,
    # including the one that raised -- a failing node must not abort the hop.
    assert stats["nodes_processed_per_depth"] == [1, 5]

    processed_names = {n for n, _s, _e in intervals}
    assert processed_names == {"Candidate 0", "Candidate 1", "Candidate 3", "Candidate 4"}

    # genuine concurrency, not accidental serialization: at least two workers'
    # [start, end) windows must overlap.
    def overlaps(a, b):
        return a[1] < b[2] and b[1] < a[2]

    assert any(overlaps(a, b) for i, a in enumerate(intervals) for b in intervals[i + 1:]), (
        "no overlap observed between hop-1 workers -- they ran sequentially, "
        "not concurrently"
    )


def test_expand_graph_single_frontier_node_does_not_use_a_thread_pool(tmp_path, monkeypatch):
    """A hop with exactly one node to process takes the plain sequential path
    -- ThreadPoolExecutor should never even be constructed for it."""
    engine = _engine(tmp_path, "single_node.db")
    Session = _sessionmaker(engine)

    def fake_process_person(worker_db, name, hop, local_disc, progress=None,
                            is_person=True, context=""):
        subject = builder.get_or_create_person(worker_db, name)
        subject.processed = 1
        worker_db.commit()

    monkeypatch.setattr(expansion, "_process_person", fake_process_person)

    def _forbidden(*a, **k):
        raise AssertionError("ThreadPoolExecutor must not be used for a single-node hop")
    monkeypatch.setattr(expansion, "ThreadPoolExecutor", _forbidden)

    s = Session()
    try:
        stats = expansion.expand_graph(s, "Solo Seed", max_depth=1)
    finally:
        s.close()
    assert stats["nodes_processed_per_depth"] == [1]


def test_expand_graph_checks_stop_condition_after_each_enriched_node(db, monkeypatch):
    monkeypatch.setattr(config, "EXPAND_PREFER_REACHABLE", False)

    processed = []

    def fake_process_person(worker_db, name, hop, local_disc, progress=None,
                            is_person=True, context=""):
        processed.append((name, hop))
        subject = builder.get_or_create_person(worker_db, name)
        subject.processed = 1
        if hop == 0:
            edge = ExtractedEdge(
                person_a=name, person_b="Bridge Person", other_kind="person",
                relationship_type="coworker", confidence_adjusted=0.8,
                confidence_base=0.5, source_url="http://x/bridge",
                signals=EdgeSignals(explicit_keyword_match=True, trusted=True))
            expansion._record(local_disc, edge)
        worker_db.commit()

    monkeypatch.setattr(expansion, "_process_person", fake_process_person)

    def should_stop(_session):
        return ("Bridge Person", 1) in processed

    stats = expansion.expand_graph(db, "Seed Person", max_depth=3,
                                   should_stop=should_stop)

    assert stats["nodes_processed_per_depth"] == [1, 1]
    assert processed == [("Seed Person", 0), ("Bridge Person", 1)]


# --- connect_people: the two sides run concurrently, on separate threads ---
def test_expand_both_concurrently_runs_both_sides_on_separate_threads(tmp_path, monkeypatch):
    engine = _engine(tmp_path, "ab_concurrency.db")
    Session = _sessionmaker(engine)

    seen = []
    lock = threading.Lock()

    # **kwargs so this double keeps working as expand_graph grows keyword-only
    # options (prefer_reachable, cancel_checker, should_stop) that this test
    # doesn't exercise -- the subject here is thread fan-out, not the signature.
    def fake_expand_graph(worker_db, name, depth, progress=None, seed_is_person=True,
                          seed_context="", protected_norms=None, on_step=None, **kwargs):
        with lock:
            seen.append((name, threading.get_ident()))
        time.sleep(0.05)
        return {}

    monkeypatch.setattr(C, "expand_graph", fake_expand_graph)

    s = Session()
    try:
        C._expand_both_concurrently(s, "Alpha", "Beta", 2, 2, {"alpha", "beta"},
                                    None, "", "")
    finally:
        s.close()

    names = {n for n, _t in seen}
    assert names == {"Alpha", "Beta"}
    thread_ids = {t for _n, t in seen}
    assert len(thread_ids) == 2, "both sides must run on distinct worker threads"


def test_expand_both_concurrently_propagates_a_side_failure(tmp_path, monkeypatch):
    engine = _engine(tmp_path, "ab_failure.db")
    Session = _sessionmaker(engine)

    def fake_expand_graph(worker_db, name, depth, progress=None, seed_is_person=True,
                          seed_context="", protected_norms=None, on_step=None, **kwargs):
        if name == "Beta":
            raise RuntimeError("simulated crawl failure")
        return {}

    monkeypatch.setattr(C, "expand_graph", fake_expand_graph)

    s = Session()
    try:
        with pytest.raises(RuntimeError, match="simulated crawl failure"):
            C._expand_both_concurrently(s, "Alpha", "Beta", 2, 2, {"alpha", "beta"},
                                        None, "", "")
    finally:
        s.close()
