"""SQLAlchemy engine / session wiring.

Two engines:
  - default engine (``config.DB_URL``) — ONE shared global discovery graph
    used by both the CLI and the HTTP API (see main.py's get_db).
  - boards engine (``config.BOARDS_DB_URL``) — the Boards/Pages UI workspace,
    owner-scoped by X-Graph-Id (see safe_graph_id). Its own SQLite file
    locally; the same database as the graph on Postgres (see config).

Both backends are supported and the schema is identical on each — models use
only portable column types, so ``create_all`` builds it on either. Postgres is
what a deployment should use: expansion runs several concurrent writers and
SQLite serializes them behind one write lock. SQLite remains the zero-setup
default for local dev and the test suite.

All SQLite engines run in WAL mode with a busy timeout so readers don't block
the writer and brief write contention retries instead of erroring.
"""
import re

from sqlalchemy import create_engine, event, inspect
from sqlalchemy.orm import declarative_base, sessionmaker

from . import config

Base = declarative_base()


def _tune_sqlite(dbapi_conn, _record) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    # 30s, not 5s: the 5s ceiling was demonstrably reachable -- four concurrent
    # expansion workers (EXPAND_NODE_CONCURRENCY) plus a bulk prune could still
    # be contending past it, and a lock that surfaces mid-flush can cost a whole
    # node (see graph.builder.flush_in_savepoint). Waiting longer is strictly
    # better than losing discovered data; nothing here is latency-sensitive
    # enough to prefer failing fast. Postgres doesn't need this at all -- it
    # has row-level locking, not one lock for the whole file.
    cur.execute("PRAGMA busy_timeout=30000")
    cur.close()


def _make_engine(url: str):
    """Engine tuned for whichever backend `url` names.

    The Postgres settings target a MANAGED database reached over the network
    (Supabase/Render/Neon), where the two failure modes are idle connections
    being cut server-side and a connection cap shared with every other client:

      - pool_pre_ping   validates a pooled connection before handing it out,
                        so a connection the provider dropped while idle costs
                        one silent reconnect instead of surfacing as a random
                        OperationalError mid-build.
      - pool_recycle    retires connections well before typical idle timeouts
                        rather than discovering they're dead.
      - pool size       bounded on purpose: expansion opens a Session per
                        concurrent node (EXPAND_NODE_CONCURRENCY) on BOTH
                        sides of a /connect, plus the API's own request
                        sessions. Managed tiers cap total connections quite
                        low, so this leaves headroom instead of exhausting it.
    """
    if url.startswith("sqlite"):
        eng = create_engine(url, connect_args={"check_same_thread": False},
                            future=True)
        event.listen(eng, "connect", _tune_sqlite)
        return eng
    return create_engine(
        url,
        future=True,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_size=config.DB_POOL_SIZE,
        max_overflow=config.DB_MAX_OVERFLOW,
        pool_timeout=30,
    )


engine = _make_engine(config.DB_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)

# Boards/pages get their OWN engine + file (config.BOARDS_DB_URL). They carry
# no FK relationship to the discovery-graph tables, and physically isolating
# them means a board's frequent autosave writes can never hit "database is
# locked" behind a long /connect or /targets/search transaction on the main DB.
boards_engine = _make_engine(config.BOARDS_DB_URL)
BoardsSessionLocal = sessionmaker(bind=boards_engine, autoflush=False, expire_on_commit=False, future=True)


def _board_tables():
    from . import models
    return [models.Board.__table__, models.BoardPage.__table__]


def init_db(bind=None) -> None:
    """Create all non-board tables (idempotent) on the given bind (default
    engine if None). Boards live on their own engine — see init_boards_db()."""
    from . import models  # noqa: F401  (register mappers)

    target = bind or engine
    _drop_legacy_boards_tables(target)
    board_tables = set(_board_tables())
    other_tables = [t for t in Base.metadata.sorted_tables if t not in board_tables]
    Base.metadata.create_all(bind=target, tables=other_tables)
    _migrate(target)


def init_boards_db() -> None:
    """Create the boards/pages tables on their own dedicated engine."""
    from . import models  # noqa: F401
    Base.metadata.create_all(bind=boards_engine, tables=_board_tables())
    try:
        cols = _columns(boards_engine, "boards")
        if cols and "mode" in cols:
            # legacy single-canvas shape — drop only tables that actually
            # exist, in either order, so a partial legacy schema (e.g. no
            # board_pages yet) can't abort this block halfway and leave
            # the incompatible `boards` table behind.
            existing = _tables(boards_engine)
            with boards_engine.begin() as conn:
                for t in ("board_pages", "boards"):
                    if t in existing:
                        conn.exec_driver_sql(f"DROP TABLE {t}")
        Base.metadata.create_all(bind=boards_engine, tables=_board_tables())
    except Exception:
        pass  # table absent or unreadable — safe to ignore
    _migrate_boards(boards_engine)


def _tables(bind) -> set:
    """Table names on `bind`, via SQLAlchemy's Inspector.

    Inspector, not `SELECT name FROM sqlite_master`: that catalog exists only
    on SQLite, so the Postgres path used to raise straight into an
    `except: pass` and silently skip every migration below. Inspector asks
    each dialect in its own terms and returns the same answer for both.
    """
    return set(inspect(bind).get_table_names())


def _columns(bind, table: str) -> set:
    """Column names on `table`, or an empty set if the table doesn't exist.
    Dialect-agnostic replacement for `PRAGMA table_info` — see _tables."""
    insp = inspect(bind)
    if not insp.has_table(table):
        return set()
    return {c["name"] for c in insp.get_columns(table)}


def _drop_legacy_boards_tables(bind) -> None:
    """Boards used to live in the main DB (single-canvas shape, then a
    multi-page shape). Both are now relocated to BOARDS_DB_URL — drop any
    leftover boards/board_pages tables here so the main DB doesn't carry
    dead, confusing tables. Nothing of value is lost: these were always
    pre-launch test rows, never real user data.

    No-op when the graph and boards share one database (the Postgres default,
    see config.BOARDS_DB_URL) — there, `boards` is the LIVE table, not a
    leftover, and dropping it would delete the user's actual boards.
    """
    if config.BOARDS_DB_URL == config.DB_URL:
        return
    try:
        existing = _tables(bind)
        with bind.begin() as conn:
            for t in ("board_pages", "boards"):
                if t in existing:
                    conn.exec_driver_sql(f"DROP TABLE {t}")
    except Exception:
        pass  # absent or unreadable — safe to ignore


def _add_columns(bind, add_columns) -> None:
    """Tiny additive migrations for existing DBs (create_all won't ALTER an
    existing table). Each guarded so it's a no-op when already applied.

    `coltype` is written in portable SQL (TEXT/INTEGER/JSON) — all three are
    accepted by SQLite and Postgres alike, so one spelling serves both.
    """
    for table, col, coltype in add_columns:
        try:
            if col in _columns(bind, table):
                continue
            with bind.begin() as conn:
                conn.exec_driver_sql(
                    f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
        except Exception:
            pass  # table absent or column already present — safe to ignore


def _migrate(bind) -> None:
    _add_columns(bind, [
        ("people", "wikidata_qid", "TEXT"),
        ("people", "processed", "INTEGER DEFAULT 0"),
        ("local_profiles", "connected_on", "TEXT"),
        ("local_profiles", "reach_profile", "JSON"),
        ("local_profiles", "owner_norm", "TEXT"),
        ("enrichment_tasks", "kind", "TEXT DEFAULT 'contact'"),
        ("enrichment_tasks", "silo_weights", "JSON"),
        ("relationship_edges", "verified_status", "TEXT"),
        ("relationship_edges", "verified_at", "TEXT"),
        ("relationship_edges", "verified_reason", "TEXT"),
    ])


def _migrate_boards(bind) -> None:
    _add_columns(bind, [
        ("boards", "target_name", "TEXT"),
        ("boards", "target_org", "TEXT"),
        ("boards", "status", "TEXT DEFAULT 'active'"),
    ])


def get_db():
    """FastAPI dependency yielding a session on the DEFAULT engine (CLI/local)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_boards_db():
    """FastAPI dependency yielding a session on the dedicated boards engine."""
    db = BoardsSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Sanitizes the X-Graph-Id header into a safe filename-stem-shaped owner id
# for scoping Boards (see main.py's _owner_id) — NOT used for the discovery
# graph, which is one shared engine for everyone (see module docstring).
_GRAPH_ID_RE = re.compile(r"[^a-zA-Z0-9_-]")


def safe_graph_id(graph_id: str) -> str:
    """Sanitize a client-supplied graph id into a safe filename stem."""
    gid = _GRAPH_ID_RE.sub("", graph_id or "")[:64]
    return gid or "default"
