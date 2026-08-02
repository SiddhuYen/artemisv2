"""Postgres as the deployment database, plus the pre-savepoint flush bug that
prompted the move.

Two independent things are covered here because they share one root story:

1. `builder.flush_in_savepoint` — `Session.begin_nested()` flushes PENDING
   state before it establishes the SAVEPOINT (SQLAlchemy's
   `SessionTransaction._take_snapshot`), and does so regardless of
   autoflush=False. `_homonym_conflict` used to leave its advisory
   `people.metadata` write pending, so the NEXT unrelated savepoint entry
   flushed it. A lock there is unrecoverable — the savepoint never exists, the
   session is deactivated, and every retry dies instantly on
   PendingRollbackError while `_is_locked` still calls it transient. Live, that
   silently dropped whole nodes from a /connect walk and turned a real route
   into "NO PATH".

2. The database wiring itself — URL resolution and dialect-agnostic schema
   introspection. The migration helpers used to read `PRAGMA table_info` and
   `sqlite_master`, which exist only on SQLite, so on Postgres every migration
   raised straight into an `except: pass` and silently did nothing.
"""
import sqlite3

import pytest
from sqlalchemy import event, select
from sqlalchemy.orm import Session

from app import config, db as db_module
from app.extraction.schemas import EdgeSignals, ExtractedEdge
from app.graph import builder
from app.models import Person
from app.providers.base import SearchResult


# A signal/candidate pair that disambiguate.domain_conflict reports as
# conflicting -- borrowed from tests/test_homonym_identity_gate.py.
_BUSINESS_EVIDENCE = "works at Trinamix Inc as Vice President Sales & Strategy"
_ACADEMIC_IDENTITY = (
    "Prantik Chakraborty, affiliated with Indian Space Research Organisation, "
    "researcher."
)


def _person_with_evidence(db: Session, name: str) -> Person:
    """A persisted person carrying enough of their own edge evidence for
    _existing_evidence_signal to return a domain-anchored signal."""
    person = builder.get_or_create_person(db, name)
    counterpart = builder.get_or_create_person(db, "Someone Else")
    edge = ExtractedEdge(
        person_a=name, person_b="Someone Else", other_kind="person",
        relationship_type="coworker", evidence_snippet=_BUSINESS_EVIDENCE,
        confidence_base=0.9, confidence_adjusted=0.9, signals=EdgeSignals(),
    )
    source = builder.save_source(db, SearchResult(
        url="https://example.com/a", title="t", snippet="s", provider="brave",
    ), query_used="q")
    builder.add_edge_from_extraction(db, person, edge, 0, source, counterpart)
    db.commit()
    return person


# ---------------------------------------------------------------------------
# 1. the pre-savepoint flush bug
# ---------------------------------------------------------------------------
def test_homonym_note_is_not_left_pending_in_the_session(db):
    """The regression itself. After the guard rejects an identity, the session
    must carry NO dirty ORM state -- that leftover is what the next savepoint
    entry would flush pre-SAVEPOINT, where a lock costs the whole node."""
    person = _person_with_evidence(db, "Prantik Chakraborty")

    assert builder._homonym_conflict(db, person, _ACADEMIC_IDENTITY) is True
    assert not db.dirty, (
        "the advisory metadata write must be flushed inside its own savepoint, "
        "not left pending for an unrelated begin_nested() to flush"
    )


def test_homonym_note_is_still_recorded(db):
    """Flushing it inside a savepoint must not cost the note itself."""
    person = _person_with_evidence(db, "Prantik Chakraborty")

    builder._homonym_conflict(db, person, _ACADEMIC_IDENTITY)

    assert "homonym_rejected" in (person.meta or {})
    assert person.meta["homonym_rejected"]["identity_text"].startswith("Prantik")


def test_resolution_survives_a_lock_on_the_homonym_write(db, monkeypatch):
    """End to end: a transient lock while writing the advisory note must be
    retried, not escalate into a lost node.

    The lock is injected at the CURSOR, not by replacing Session.flush --
    that distinction is the whole point. Patching flush() away means
    SQLAlchemy's own `_flush` never runs, so it never calls
    `transaction.rollback(_capture_exception=True)` and the session is never
    deactivated -- which is the exact state that made the real bug
    unrecoverable. A test that skips it passes with the bug still present
    (confirmed: it did).

    Before the fix, this raised PendingRollbackError out of
    get_or_create_person: the pre-savepoint flush deactivated the session, so
    the caller's retry loop re-entered begin_nested() and died instantly,
    five times over, before giving up and dropping the node.
    """
    _person_with_evidence(db, "Prantik Chakraborty")
    dialect = db.get_bind().dialect
    real_do_execute = dialect.do_execute
    state = {"fired": False}

    def flaky_do_execute(cursor, statement, parameters, context=None):
        if not state["fired"] and statement.lstrip().upper().startswith("UPDATE PEOPLE"):
            state["fired"] = True
            # raised from do_execute so SQLAlchemy routes it through
            # _handle_dbapi_exception and wraps it with .orig set — the shape
            # _is_locked actually inspects, and the exact frame the live
            # failure came from
            raise sqlite3.OperationalError("database is locked")
        return real_do_execute(cursor, statement, parameters, context)

    monkeypatch.setattr(dialect, "do_execute", flaky_do_execute)
    monkeypatch.setattr(builder, "_deadlock_backoff", lambda attempt: None)

    resolved = builder.get_or_create_person(
        db, "Prantik Chakraborty", identity_text=_ACADEMIC_IDENTITY)

    assert state["fired"], "the UPDATE under test never ran — setup is wrong"
    assert resolved is not None, "a transient lock must not drop the node"


def test_flush_in_savepoint_gives_up_on_a_non_transient_error(db):
    """A real bug must still surface -- the retry is for locks, not for
    swallowing genuine failures."""
    person = _person_with_evidence(db, "Prantik Chakraborty")

    def boom():
        raise builder.OperationalError("...", {}, Exception("disk I/O error"))

    with pytest.raises(builder.OperationalError):
        builder.flush_in_savepoint(db, boom)
    assert person is not None


# ---------------------------------------------------------------------------
# 2. database URL resolution
# ---------------------------------------------------------------------------
def test_postgres_scheme_is_rewritten_for_sqlalchemy():
    """Supabase/Render/Heroku still hand out `postgres://`, which SQLAlchemy
    1.4+ refuses with NoSuchModuleError. A pasted dashboard string must work."""
    assert config._normalize_db_url("postgres://u:p@host:5432/db") == \
        "postgresql://u:p@host:5432/db"


def test_normalize_leaves_other_urls_alone():
    for url in ("postgresql://u@h/db", "sqlite:///./artemis.db", "sqlite://"):
        assert config._normalize_db_url(url) == url


def test_database_url_is_used_when_artemis_db_url_is_absent(monkeypatch):
    """Render/Supabase populate DATABASE_URL automatically — attaching the
    database should be all a deploy needs."""
    monkeypatch.delenv("ARTEMIS_DB_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@host/db")
    assert config._resolve_db_url() == "postgresql://u:p@host/db"


def test_artemis_db_url_wins_over_database_url(monkeypatch):
    """A local override must not require unsetting the platform's variable."""
    monkeypatch.setenv("ARTEMIS_DB_URL", "sqlite:///./local.db")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host/db")
    assert config._resolve_db_url() == "sqlite:///./local.db"


def test_sqlite_is_the_default_when_nothing_is_set(monkeypatch):
    monkeypatch.delenv("ARTEMIS_DB_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert config._resolve_db_url() == "sqlite:///./artemis.db"


# ---------------------------------------------------------------------------
# 3. dialect-agnostic schema introspection
# ---------------------------------------------------------------------------
def test_columns_reads_a_real_table(db):
    cols = db_module._columns(db.get_bind(), "people")
    assert {"id", "norm_name", "metadata"} <= cols


def test_columns_is_empty_for_a_missing_table(db):
    """Must return empty rather than raise — _add_columns relies on this to
    skip tables a given database doesn't have."""
    assert db_module._columns(db.get_bind(), "no_such_table") == set()


def test_tables_lists_the_schema(db):
    names = db_module._tables(db.get_bind())
    assert {"people", "organizations", "relationship_edges"} <= names


def test_add_columns_applies_a_missing_column(db):
    """The additive migration path, exercised for real. On SQLite this used to
    work via PRAGMA; the Inspector version must behave identically here AND be
    the same code path Postgres now takes."""
    bind = db.get_bind()
    with bind.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE widgets (id TEXT)")

    db_module._add_columns(bind, [("widgets", "colour", "TEXT")])

    assert "colour" in db_module._columns(bind, "widgets")


def test_add_columns_is_idempotent(db):
    bind = db.get_bind()
    with bind.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE widgets (id TEXT)")

    db_module._add_columns(bind, [("widgets", "colour", "TEXT")])
    db_module._add_columns(bind, [("widgets", "colour", "TEXT")])  # no-op

    assert "colour" in db_module._columns(bind, "widgets")


def test_add_columns_ignores_a_table_that_does_not_exist(db):
    """A database that predates a table shouldn't turn a migration into a
    hard boot failure."""
    db_module._add_columns(db.get_bind(), [("absent_table", "x", "TEXT")])


def test_migration_column_types_are_portable(monkeypatch):
    """Every column type spelled in the migration lists must be accepted by
    BOTH backends. One spelling has to serve SQLite and Postgres now that they
    share this code path, so keep the vocabulary boring on purpose."""
    recorded = []
    monkeypatch.setattr(db_module, "_add_columns",
                        lambda bind, cols: recorded.extend(cols))

    db_module._migrate(None)
    db_module._migrate_boards(None)

    portable = {"TEXT", "INTEGER", "JSON"}
    assert recorded, "the migration lists should not be empty"
    for _table, _col, coltype in recorded:
        base = coltype.split()[0].upper()
        assert base in portable, f"{coltype!r} is not portable across backends"


# ---------------------------------------------------------------------------
# 4. destructive paths must not fire against a SHARED graph
# ---------------------------------------------------------------------------
def test_reset_is_refused_on_a_shared_graph(db, monkeypatch):
    """The footgun this guard exists for: `python -m app.cli "Some Name"`
    resets by default (--keep is opt-IN), so on a team database the most
    ordinary command there is would delete every collaborator's graph."""
    monkeypatch.setattr(builder.config, "IS_POSTGRES", True)
    _person_with_evidence(db, "Prantik Chakraborty")

    with pytest.raises(builder.SharedGraphResetError):
        builder.reset_public_graph(db)

    assert db.execute(select(Person)).scalars().all(), "nothing may be deleted"


def test_reset_still_works_on_a_private_sqlite_graph(db, monkeypatch):
    """Unchanged behaviour for the local-file case it was written for."""
    monkeypatch.setattr(builder.config, "IS_POSTGRES", False)
    _person_with_evidence(db, "Prantik Chakraborty")

    builder.reset_public_graph(db)

    assert not db.execute(select(Person)).scalars().all()


def test_force_overrides_the_shared_guard(db, monkeypatch):
    """An explicit, deliberate wipe is still available."""
    monkeypatch.setattr(builder.config, "IS_POSTGRES", True)
    _person_with_evidence(db, "Prantik Chakraborty")

    builder.reset_public_graph(db, force=True)

    assert not db.execute(select(Person)).scalars().all()


def test_org_discovery_scratch_cleanup_spares_a_shared_graph(db, monkeypatch):
    """org_discovery wipes the public graph as routine cleanup. On a shared
    database that must degrade to leaving data behind, not destroying it."""
    from app.network import org_discovery

    monkeypatch.setattr(builder.config, "IS_POSTGRES", True)
    _person_with_evidence(db, "Prantik Chakraborty")

    org_discovery._clear_scratch_graph(db)  # must not raise, must not delete

    assert db.execute(select(Person)).scalars().all(), (
        "a collaborator's graph must survive someone else's add-org-network")
