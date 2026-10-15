import fakeredis
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import config
import models.call  # noqa: F401  (registers the tables)
import models.semantic_cache  # noqa: F401
import models.user  # noqa: F401
import services.auth
import services.breaker
import services.cache
import services.call_log
import services.embeddings
import services.gateway
import services.semantic_cache
import services.stats
from api.main import app
from services.auth import AuthUser, current_user
from services.db import Base
from services.llm import LLMError

# The autouse fixtures below replace these with fakes. A few tests need the REAL functions
# (to test their own logic with fake inputs), so we keep a reference before any fixture runs.
REAL_GET_REDIS = services.cache._get_redis
REAL_EMBED = services.embeddings.embed
REAL_FIND_CLOSEST = services.semantic_cache.find_closest

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
    monkeypatch.setattr(services.auth, "get_session", TestSession)
    monkeypatch.setattr(services.stats, "get_session", TestSession)
    yield TestSession
    engine.dispose()


@pytest.fixture(autouse=True)
def default_user():
    """Most tests are not about login, so /chat sees a ready-made user. tests/test_auth.py removes this."""
    services.auth._cache.clear()
    app.dependency_overrides[current_user] = lambda: AuthUser(1, "tester", 10000, True)
    yield
    app.dependency_overrides.pop(current_user, None)
    services.auth._cache.clear()


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


@pytest.fixture(autouse=True)
def fake_lookup(monkeypatch):
    """SQLite has no pgvector, so the semantic lookup is faked. Default: nothing similar found.

    Tests set fake_lookup.match to a dict like {"answer": "...", "similarity": 0.95}.
    fake_lookup.calls counts how often it ran.
    """
    class FakeLookup:
        match = None
        calls = 0
        last_user_id = None  # whose rows the lookup was limited to

    def fake_find_closest(vector, model, user_id=None):
        FakeLookup.calls += 1
        FakeLookup.last_user_id = user_id
        return FakeLookup.match

    FakeLookup.calls = 0
    FakeLookup.match = None
    monkeypatch.setattr(services.semantic_cache, "find_closest", fake_find_closest)
    return FakeLookup


FAKE_GEMINI_RESULT = {
    "answer": "Gemini says: RAG means Retrieval-Augmented Generation.",
    "model": "gemini-2.5-flash-lite",
    "provider": "gemini",
    "prompt_tokens": 6,
    "completion_tokens": 8,
    "latency_ms": 20.0,
    "truncated": False,
}


@pytest.fixture(autouse=True)
def fake_gemini(monkeypatch):
    """Gemini never makes a real call in tests. By default it is 'down'; a test sets fake_gemini.working = True."""
    class FakeGemini:
        calls = 0
        working = False

    async def fake_ask_gemini(question):
        FakeGemini.calls += 1
        if not FakeGemini.working:
            raise LLMError("Gemini is not available in this test", "gemini_error_503")
        return dict(FAKE_GEMINI_RESULT)

    monkeypatch.setattr(services.gateway, "ask_gemini", fake_ask_gemini)
    return FakeGemini


@pytest.fixture(autouse=True)
def breaker_clock(monkeypatch):
    """Every test starts with a closed circuit breaker and a clock the test can move by hand."""
    class BreakerClock:
        now = 1000.0

    monkeypatch.setattr(services.breaker, "_now", lambda: BreakerClock.now)
    services.breaker.groq_breaker.record_success()
    yield BreakerClock
    services.breaker.groq_breaker.record_success()


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
