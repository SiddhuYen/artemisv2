"""Shared pytest fixtures.

Each test gets its own isolated in-memory SQLite database, so tests never
touch the real artemis.db files on disk and never see each other's rows.

The env override below is separate and just as load-bearing: it pins the
APP'S OWN module-level engine (app.db.engine, built at import time from
config.DB_URL) to a throwaway file, before `app` is imported for the first
time. Without it, any test that starts the FastAPI app -- TestClient's
context manager fires the startup event, which calls init_db() and
init_boards_db() -- runs schema creation against whatever `.env` configures.
Once that became a real Postgres, the suite was quietly doing create_all and
migration round-trips against the DEPLOYED database on every such test: not
just wrong, but slow enough to be obvious (~10s per test setup, a 27x
increase for the whole run). Tests must not depend on, or write to, whatever
database happens to be configured.

setdefault, not assignment: `ARTEMIS_DB_URL=... pytest` still works for
deliberately exercising the suite against another backend.
"""
import os
import tempfile

_TEST_DB_DIR = tempfile.mkdtemp(prefix="artemis-tests-")
os.environ.setdefault("ARTEMIS_DB_URL", f"sqlite:///{_TEST_DB_DIR}/test.db")
os.environ.setdefault("ARTEMIS_BOARDS_DB_URL", f"sqlite:///{_TEST_DB_DIR}/boards.db")

import pytest  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.db import Base  # noqa: E402
import app.models  # noqa: F401,E402  (register mappers on Base before create_all)


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False},
        poolclass=StaticPool, future=True,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False,
                           expire_on_commit=False, future=True)()
    try:
        yield session
    finally:
        session.close()
