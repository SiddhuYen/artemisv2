"""SQLAlchemy engine / session wiring.

Two engines:
  - default engine (``config.DB_URL``) — ONE shared global discovery graph
    used by both the CLI and the HTTP API (see main.py's get_db).
  - boards engine (``config.BOARDS_DB_URL``) — a separate file for the
    Boards/Pages UI workspace, owner-scoped by X-Graph-Id (see safe_graph_id).

All SQLite engines run in WAL mode with a busy timeout so readers don't block
the writer and brief write contention retries instead of erroring.
"""
import re

from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker

from . import config

Base = declarative_base()


def _tune_sqlite(dbapi_conn, _record) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA busy_timeout=5000")
    cur.close()


def _make_engine(url: str):
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    eng = create_engine(url, connect_args=connect_args, future=True)
    if url.startswith("sqlite"):
        event.listen(eng, "connect", _tune_sqlite)
    return eng


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
        with boards_engine.begin() as conn:
            cols = {r[1] for r in conn.exec_driver_sql(
                "PRAGMA table_info(boards)").fetchall()}
            if cols and "mode" in cols:
                # legacy single-canvas shape — drop only tables that actually
                # exist, in either order, so a partial legacy schema (e.g. no
                # board_pages yet) can't abort this block halfway and leave
                # the incompatible `boards` table behind.
                existing = {r[0] for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
                for t in ("board_pages", "boards"):
                    if t in existing:
                        conn.exec_driver_sql(f"DROP TABLE {t}")
        Base.metadata.create_all(bind=boards_engine, tables=_board_tables())
    except Exception:
        pass  # non-SQLite or table absent — safe to ignore
    _migrate_boards(boards_engine)


def _drop_legacy_boards_tables(bind) -> None:
    """Boards used to live in the main DB (single-canvas shape, then a
    multi-page shape). Both are now relocated to BOARDS_DB_URL — drop any
    leftover boards/board_pages tables here so the main DB doesn't carry
    dead, confusing tables. Nothing of value is lost: these were always
    pre-launch test rows, never real user data."""
    try:
        with bind.begin() as conn:
            existing = {r[0] for r in conn.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            for t in ("board_pages", "boards"):
                if t in existing:
                    conn.exec_driver_sql(f"DROP TABLE {t}")
    except Exception:
        pass  # non-SQLite or absent — safe to ignore


def _add_columns(bind, add_columns) -> None:
    """Tiny additive migrations for existing SQLite DBs (create_all won't ALTER
    an existing table). Each guarded so it's a no-op when already applied."""
    with bind.begin() as conn:
        for table, col, coltype in add_columns:
            try:
                cols = {r[1] for r in conn.exec_driver_sql(
                    f"PRAGMA table_info({table})").fetchall()}
                if col not in cols:
                    conn.exec_driver_sql(
                        f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
            except Exception:
                pass  # non-SQLite or already present — safe to ignore


def _migrate(bind) -> None:
    _add_columns(bind, [
        ("people", "wikidata_qid", "TEXT"),
        ("people", "processed", "INTEGER DEFAULT 0"),
        ("local_profiles", "connected_on", "TEXT"),
        ("enrichment_tasks", "kind", "TEXT DEFAULT 'contact'"),
        ("enrichment_tasks", "silo_weights", "JSON"),
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
