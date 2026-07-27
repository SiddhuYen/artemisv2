"""SQLAlchemy engine / session wiring.

Two modes:
  - default engine (``config.DB_URL``) — used by the CLI (a single local graph).
  - per-graph engines (``graph_session(graph_id)``) — used by the HTTP API so
    each browser session gets an ISOLATED SQLite file under ``config.GRAPH_DIR``.
    Concurrent beta testers can't clobber each other's public graph.

All SQLite engines run in WAL mode with a busy timeout so readers don't block
the writer and brief write contention retries instead of erroring.
"""
import os
import re
import threading

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
                conn.exec_driver_sql("DROP TABLE board_pages")
                conn.exec_driver_sql("DROP TABLE boards")
        Base.metadata.create_all(bind=boards_engine, tables=_board_tables())
    except Exception:
        pass  # non-SQLite or table absent — safe to ignore


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


def _migrate(bind) -> None:
    """Tiny additive migrations for existing SQLite DBs (create_all won't ALTER
    an existing table). Each guarded so it's a no-op when already applied."""
    add_columns = [
        ("people", "wikidata_qid", "TEXT"),
        ("people", "processed", "INTEGER DEFAULT 0"),
        ("local_profiles", "connected_on", "TEXT"),
    ]
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


# --- per-graph (per-session) engines for the HTTP API ----------------------
_GRAPH_ID_RE = re.compile(r"[^a-zA-Z0-9_-]")
_graph_makers: dict = {}
_graph_lock = threading.Lock()


def safe_graph_id(graph_id: str) -> str:
    """Sanitize a client-supplied graph id into a safe filename stem."""
    gid = _GRAPH_ID_RE.sub("", graph_id or "")[:64]
    return gid or "default"


def graph_session(graph_id: str):
    """Return a Session bound to an isolated per-graph SQLite file.

    Engines are created once per graph id and cached; tables are created on
    first use. Callers own the session and must close it.
    """
    gid = safe_graph_id(graph_id)
    with _graph_lock:
        maker = _graph_makers.get(gid)
        if maker is None:
            os.makedirs(config.GRAPH_DIR, exist_ok=True)
            url = f"sqlite:///{os.path.join(config.GRAPH_DIR, gid + '.db')}"
            eng = _make_engine(url)
            init_db(bind=eng)
            maker = sessionmaker(bind=eng, autoflush=False,
                                 expire_on_commit=False, future=True)
            _graph_makers[gid] = maker
    return maker()
