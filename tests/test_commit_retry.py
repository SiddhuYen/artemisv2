"""Generalized SQLite-lock retry: builder.commit_with_retry, and every write
path that now goes through it or through the widened _is_transient check.

Backstory: PR #20 added delete_relationship_edges_with_retry for ONE call
site (the pruning bulk delete) after it kept failing live with "database is
locked". Stress-testing the rest of the app surfaced the same error on a
completely different, far hotter path -- builder.save_source's plain
db.add()+flush(), with zero retry protection -- and further digging found
the two PRE-EXISTING retry-wrapped functions (_new_person_or_existing,
get_or_create_org) only ever retried a Postgres deadlock, never a SQLite
lock, because their except clause checked _is_deadlock alone. On SQLite,
_is_deadlock is always False, so they re-raised a lock immediately -- the
exact same failure mode as the totally-unprotected call sites, just hidden
behind retry-shaped code that never actually retried for this app's own
database engine.

These tests use a live in-memory SQLite session (see conftest.db) and
monkeypatch Session.commit/flush to simulate a transient lock a bounded
number of times before succeeding for real -- the same technique
tests/test_prune_lock_retry.py already established for this file's sibling.
"""
import pytest
from sqlalchemy.exc import OperationalError

from app.extraction.schemas import EdgeSignals, ExtractedEdge
from app.graph import builder
from app.models import Person, RelationshipEdge, Source
from app.providers.base import SearchResult
from app.utils.names import person_norm_key


def _locked_error() -> OperationalError:
    return OperationalError("...", {}, Exception("database is locked"))


def _other_error() -> OperationalError:
    return OperationalError("...", {}, Exception("disk I/O error"))


class _FakePgOrig(Exception):
    pgcode = "40P01"


def _deadlock_error() -> OperationalError:
    return OperationalError("...", {}, _FakePgOrig())


def _person(db, name):
    p = Person(canonical_name=name, norm_name=person_norm_key(name))
    db.add(p)
    db.flush()
    return p


def _flaky(real_fn, fail_times, exc_factory):
    """Wrap `real_fn` so the first `fail_times` calls raise `exc_factory()`
    and every call after that runs for real. Returns (wrapper, call_counter)."""
    calls = {"n": 0}

    def wrapper(*a, **k):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise exc_factory()
        return real_fn(*a, **k)

    return wrapper, calls


# ---------------------------------------------------------------------------
# _is_transient
# ---------------------------------------------------------------------------
def test_is_transient_recognizes_a_sqlite_lock():
    assert builder._is_transient(_locked_error()) is True


def test_is_transient_recognizes_a_postgres_deadlock():
    assert builder._is_transient(_deadlock_error()) is True


def test_is_transient_rejects_an_unrelated_operational_error():
    assert builder._is_transient(_other_error()) is False


# ---------------------------------------------------------------------------
# commit_with_retry
# ---------------------------------------------------------------------------
def test_commit_with_retry_succeeds_immediately_with_no_apply(db):
    _person(db, "Alpha")
    builder.commit_with_retry(db)  # must not raise
    assert db.query(Person).count() == 1


def test_commit_with_retry_reapplies_a_mutation_lost_to_rollback(db, monkeypatch):
    """The whole point of `apply`: a bare db.commit() retry would silently
    commit nothing here, because rollback() (forced by the first failure)
    reverts the pending attribute change back to its last-committed value."""
    p = _person(db, "Alpha")
    db.commit()

    real_commit = db.commit
    wrapper, calls = _flaky(real_commit, fail_times=2, exc_factory=_locked_error)
    monkeypatch.setattr(db, "commit", wrapper)
    monkeypatch.setattr(builder, "_deadlock_backoff", lambda attempt: None)

    builder.commit_with_retry(db, lambda: setattr(p, "canonical_name", "Alpha Prime"))

    assert calls["n"] == 3
    fresh = db.query(Person).filter(Person.id == p.id).one()
    assert fresh.canonical_name == "Alpha Prime", (
        "apply() must have been re-run on each attempt -- if only commit() "
        "were retried, rollback() would have reverted the name back to "
        "'Alpha' and the retry would have committed that silently")


def test_commit_with_retry_returns_apply_result_on_success(db):
    result = builder.commit_with_retry(db, lambda: 42)
    assert result == 42


def test_commit_with_retry_gives_up_after_exhausting_retries(db, monkeypatch):
    _person(db, "Alpha")
    always_locked = lambda *a, **k: (_ for _ in ()).throw(_locked_error())
    monkeypatch.setattr(db, "commit", always_locked)
    monkeypatch.setattr(builder, "_deadlock_backoff", lambda attempt: None)

    with pytest.raises(OperationalError):
        builder.commit_with_retry(db, _retries=3)


def test_commit_with_retry_never_retries_a_non_lock_error(db, monkeypatch):
    calls = {"n": 0}

    def always_broken(*a, **k):
        calls["n"] += 1
        raise _other_error()

    monkeypatch.setattr(db, "commit", always_broken)

    with pytest.raises(OperationalError):
        builder.commit_with_retry(db)
    assert calls["n"] == 1, "a non-lock/non-deadlock error must not be retried at all"


# ---------------------------------------------------------------------------
# save_source -- was a bare add()+flush() with zero protection; this is the
# exact call site that failed live during stress testing.
# ---------------------------------------------------------------------------
def test_save_source_recovers_from_a_transient_lock(db, monkeypatch):
    real_flush = db.flush
    wrapper, calls = _flaky(real_flush, fail_times=2, exc_factory=_locked_error)
    monkeypatch.setattr(db, "flush", wrapper)
    monkeypatch.setattr(builder, "_deadlock_backoff", lambda attempt: None)

    res = SearchResult("A Title", "https://example.com/a", "a snippet", "serper")
    source = builder.save_source(db, res, "query used")

    # >= 3, not == 3: db.begin_nested()'s own internals may issue an extra
    # flush beyond the explicit one this function calls -- what matters is
    # that it took more than one attempt (proving the retry actually fired)
    # and still landed the row.
    assert calls["n"] >= 3
    assert source.url == "https://example.com/a"
    assert db.query(Source).count() == 1


def test_save_source_gives_up_after_exhausting_retries(db, monkeypatch):
    always_locked = lambda *a, **k: (_ for _ in ()).throw(_locked_error())
    monkeypatch.setattr(db, "flush", always_locked)
    monkeypatch.setattr(builder, "_deadlock_backoff", lambda attempt: None)

    res = SearchResult("A Title", "https://example.com/b", "snip", "serper")
    with pytest.raises(OperationalError):
        builder.save_source(db, res, "query used", _retries=2)


# ---------------------------------------------------------------------------
# add_edge_from_extraction -- same shape/risk as save_source, same fix.
# ---------------------------------------------------------------------------
def test_add_edge_from_extraction_recovers_from_a_transient_lock(db, monkeypatch):
    subject = _person(db, "Alpha")
    counterpart = _person(db, "Beta")
    db.commit()

    real_flush = db.flush
    wrapper, calls = _flaky(real_flush, fail_times=2, exc_factory=_locked_error)
    monkeypatch.setattr(db, "flush", wrapper)
    monkeypatch.setattr(builder, "_deadlock_backoff", lambda attempt: None)

    edge = ExtractedEdge(
        person_a="Alpha", person_b="Beta", other_kind="person",
        relationship_type="coworker", confidence_base=0.7, confidence_adjusted=0.7,
        signals=EdgeSignals(explicit_keyword_match=True),
    )
    row = builder.add_edge_from_extraction(db, subject, edge, 0, None, counterpart)

    assert calls["n"] >= 3  # see save_source's identical note on this bound
    assert row is not None and row.relationship_type == "coworker"
    assert db.query(RelationshipEdge).count() == 1


# ---------------------------------------------------------------------------
# _new_person_or_existing / get_or_create_org -- the actual gap: these
# already LOOKED retry-protected, but their except clause checked
# _is_deadlock alone, which is Postgres-only and always False on SQLite. A
# SQLite lock here used to re-raise on the very first attempt.
# ---------------------------------------------------------------------------
def test_new_person_or_existing_now_retries_a_sqlite_lock(db, monkeypatch):
    calls = {"n": 0}
    real_begin_nested = db.begin_nested

    def flaky_begin_nested():
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _locked_error()
        return real_begin_nested()

    monkeypatch.setattr(db, "begin_nested", flaky_begin_nested)
    monkeypatch.setattr(builder, "_deadlock_backoff", lambda attempt: None)

    person = builder._new_person_or_existing(db, "Alpha", person_norm_key("Alpha"), None)

    assert calls["n"] == 3, "must have retried the SQLite lock, not re-raised immediately"
    assert person.canonical_name == "Alpha"


def test_get_or_create_org_now_retries_a_sqlite_lock(db, monkeypatch):
    calls = {"n": 0}
    real_begin_nested = db.begin_nested

    def flaky_begin_nested():
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _locked_error()
        return real_begin_nested()

    monkeypatch.setattr(db, "begin_nested", flaky_begin_nested)
    monkeypatch.setattr(builder, "_deadlock_backoff", lambda attempt: None)

    org = builder.get_or_create_org(db, "Acme Corp")

    assert calls["n"] == 3
    assert org is not None and org.name == "Acme Corp"
