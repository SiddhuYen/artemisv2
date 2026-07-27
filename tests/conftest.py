"""Shared pytest fixtures.

Each test gets its own isolated in-memory SQLite database, so tests never
touch the real artemis.db files on disk and never see each other's rows.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base
import app.models  # noqa: F401  (register mappers on Base before create_all)


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
