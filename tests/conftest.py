import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import api.main
import models.call  # noqa: F401  (registers the calls table)
import services.call_log
from services.db import Base

FAKE_RESULT = {
    "answer": "RAG means Retrieval-Augmented Generation.",
    "model": "openai/gpt-oss-20b",
    "provider": "groq",
    "prompt_tokens": 5,
    "completion_tokens": 7,
    "latency_ms": 12.3,
}


@pytest.fixture(autouse=True)
def test_db(monkeypatch):
    """Every test gets a fresh, empty in-memory SQLite database.

    autouse=True means this runs for EVERY test, so no test can ever touch the real Neon DB.
    """
    engine = create_engine(
        "sqlite://",  # in memory: disappears when the test ends
        connect_args={"check_same_thread": False},  # saving happens in a background thread
        poolclass=StaticPool,  # all sessions share one connection, so they see the same data
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)
    monkeypatch.setattr(services.call_log, "get_session", TestSession)
    yield TestSession
    engine.dispose()


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the real Groq call with one that instantly returns FAKE_RESULT."""
    async def fake_ask_llm(question):
        return dict(FAKE_RESULT)

    monkeypatch.setattr(api.main, "ask_llm", fake_ask_llm)
    return FAKE_RESULT
