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

The provider cache is pinned for the same reason and was missed for longer,
because until something read a number back out of it nothing made the leak
visible. It is a separate sqlite file (config.CACHE_DB) holding search and page
responses plus persistent counters, and any test that exercised a cached path
wrote into the operator's REAL one. Harmless-looking while the only readers
were cache hits; not harmless once Claude token accounting started keeping
month-to-date counters there, at which point a suite run added phantom calls
and phantom dollars to what /status reports as spend.

setdefault, not assignment: `ARTEMIS_DB_URL=... pytest` still works for
deliberately exercising the suite against another backend.
"""
import os
import tempfile

_TEST_DB_DIR = tempfile.mkdtemp(prefix="artemis-tests-")
os.environ.setdefault("ARTEMIS_DB_URL", f"sqlite:///{_TEST_DB_DIR}/test.db")
os.environ.setdefault("ARTEMIS_BOARDS_DB_URL", f"sqlite:///{_TEST_DB_DIR}/boards.db")
os.environ.setdefault("ARTEMIS_CACHE_DB", f"{_TEST_DB_DIR}/cache.db")

import pytest  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.db import Base  # noqa: E402
import app.models  # noqa: F401,E402  (register mappers on Base before create_all)


@pytest.fixture(autouse=True)
def _no_live_claude(monkeypatch):
    """No test may reach the real Anthropic API.

    This was not hypothetical. hop_verify runs inside connect_people and calls
    Claude whenever a credential resolves, so on a developer machine with a key
    in .env the suite was making live, billed calls -- and test_route_hop_context
    passed only because the operator's real provider cache happened to hold a
    verdict for its synthetic "Ada End"/"Bo End" pair. Pinning the cache to a
    temp dir (above) emptied that cache and the test started failing, which is
    the honest outcome: it had been asserting Claude's opinion of a made-up edge,
    fetched over the network, in a unit test.

    Tests that want Claude behavior stub the specific stage they care about
    (relation_classifier.classify, _get_client, hop_verify.*) and are unaffected
    -- those patches sit above this one.
    """
    from app import config
    from app.extraction import claude_client

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "", raising=False)
    monkeypatch.setattr(claude_client, "_get_client", lambda: None)
    claude_client.reset_availability_cache()
    yield
    claude_client.reset_availability_cache()


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
