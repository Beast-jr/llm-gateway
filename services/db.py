import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
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


def get_session() -> Session:
    # Connect only when first needed, so importing this file (e.g. in tests or CI)
    # doesn't require a real database.
    global _SessionLocal
    if _SessionLocal is None:
        # pool_pre_ping: Neon closes idle connections, so check a connection is alive before using it.
        engine = create_engine(get_database_url(), pool_pre_ping=True)
        _SessionLocal = sessionmaker(bind=engine)
    return _SessionLocal()
