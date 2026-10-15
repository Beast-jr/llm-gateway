import fakeredis
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import config
import models.call  # noqa: F401  (registers the tables)
import models.semantic_cache  # noqa: F401
import services.cache
import services.call_log
import services.embeddings
import services.gateway
import services.semantic_cache
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
    monkeypatch.setattr(services.semantic_cache, "get_session", TestSession)
    yield TestSession
    engine.dispose()


@pytest.fixture(autouse=True)
def fake_redis(monkeypatch):
    """Every test gets a fresh, empty fake Redis that lives in memory, never the real Upstash."""
    server = fakeredis.FakeServer()
    client = fakeredis.FakeAsyncRedis(server=server, decode_responses=True)
    monkeypatch.setattr(services.cache, "_get_redis", lambda: client)
    return server  # tests can set server.connected = False to simulate Redis being down


FAKE_VECTOR = [1.0] + [0.0] * (config.EMBEDDING_DIM - 1)  # a fixed, already-normalized vector


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    """Every test uses a fake embedder, so the real model is never downloaded or loaded."""
    async def fake_embed(text):
        return list(FAKE_VECTOR)

    monkeypatch.setattr(services.embeddings, "embed", fake_embed)


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the real Groq call with a fake one. fake_llm.calls counts how often it ran."""
    class FakeLLM:
        calls = 0
        result = FAKE_RESULT

    async def fake_ask_llm(question):
        FakeLLM.calls += 1
        return dict(FAKE_RESULT)

    monkeypatch.setattr(services.gateway, "ask_llm", fake_ask_llm)
    return FakeLLM
