import pytest
from fastapi.testclient import TestClient

import config
import services.embeddings
import services.semantic_cache
from api.main import app
from models.call import Call
from models.semantic_cache import SemanticCacheEntry
from tests.conftest import FAKE_VECTOR

client = TestClient(app)


def test_miss_saves_to_semantic_cache(fake_llm, test_db):
    res = client.post("/chat", json={"question": "What is RAG?"})
    assert res.status_code == 200

    with test_db() as session:
        rows = session.query(SemanticCacheEntry).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.question == "What is RAG?"
    assert row.answer == fake_llm.result["answer"]
    assert row.model == "openai/gpt-oss-20b"
    assert row.prompt_tokens == 5 and row.completion_tokens == 7
    assert len(row.embedding) == config.EMBEDDING_DIM
    assert list(row.embedding) == pytest.approx(FAKE_VECTOR)


def test_exact_cache_hit_is_not_saved_again(fake_llm, test_db):
    client.post("/chat", json={"question": "What is RAG?"})
    second = client.post("/chat", json={"question": "What is RAG?"}).json()
    assert second["cache_hit"] is True

    with test_db() as session:
        assert session.query(SemanticCacheEntry).count() == 1


def test_embedding_failure_still_returns_200(fake_llm, test_db, monkeypatch):
    async def broken_embed(text):
        raise RuntimeError("model failed to load")

    monkeypatch.setattr(services.embeddings, "embed", broken_embed)
    res = client.post("/chat", json={"question": "What is RAG?"})

    assert res.status_code == 200
    with test_db() as session:
        assert session.query(SemanticCacheEntry).count() == 0
        assert session.query(Call).count() == 1  # the normal call log still worked


def test_db_failure_still_returns_200(fake_llm, monkeypatch):
    def broken_session():
        raise ConnectionError("database is down")

    monkeypatch.setattr(services.semantic_cache, "get_session", broken_session)
    res = client.post("/chat", json={"question": "What is RAG?"})

    assert res.status_code == 200
    assert res.json()["answer"] == fake_llm.result["answer"]
