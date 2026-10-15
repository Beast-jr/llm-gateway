import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()


class Base(DeclarativeBase):
    """Parent class for all database tables (calls today, users later)."""


def get_database_url() -> str:
    url = os.getenv("DATABASE_URL", "")
    # Neon gives "postgresql://..."; this tells SQLAlchemy to use the psycopg (v3) driver.
    # psycopg understands Neon's "?sslmode=require" as-is, so no other changes are needed.
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


_SessionLocal: sessionmaker | None = None
_engine = None


def get_session() -> Session:
    # Connect only when first needed, so importing this file (e.g. in tests or CI)
    # doesn't require a real database.
    global _SessionLocal, _engine
    if _SessionLocal is None:
        # No pool_pre_ping: it costs a full network round trip before EVERY query (~250ms to Neon
        # from India). Instead, pool_recycle swaps connections out before Neon can drop them (it
        # suspends idle compute after 5 min). The pool keeps connections open and reuses them.
        _engine = create_engine(
            get_database_url(),
            pool_recycle=240,
            # TCP keepalive: send a tiny "still here" packet when idle, so Neon/routers don't drop us.
            connect_args={"keepalives": 1, "keepalives_idle": 30, "keepalives_interval": 10, "keepalives_count": 3},
        )
        _SessionLocal = sessionmaker(bind=_engine)
    return _SessionLocal()


def warm_up() -> None:
    """Open one real connection now (new connections take ~3s), so it is ready in the pool."""
    with get_session() as session:
        session.execute(text("SELECT 1"))


def retry_once_if_dropped(fn):
    """Run fn(); if the connection turns out to be dead, throw away the pooled connections and try
    once more with a fresh one. Callers run this in a thread (asyncio.to_thread), and a thread
    isn't cancelled by a timeout, so a slow reconnect still finishes for the next request."""
    try:
        return fn()
    except OperationalError:  # "server closed the connection unexpectedly" and friends
        if _engine is not None:
            _engine.dispose()  # forget the dead connection(s); the retry opens a new one
        return fn()
