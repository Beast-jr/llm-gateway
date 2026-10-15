import time

from fastapi.testclient import TestClient

import config
import services.embeddings
from api.main import app
from models.call import Call
from models.semantic_cache import SemanticCacheEntry

client = TestClient(app)
SAVED = {"answer": "RAG = retrieval-augmented generation.", "similarity": 0.94}


def test_semantic_hit_returns_saved_answer(fake_llm, fake_lookup, test_db):
    fake_lookup.match = dict(SAVED)
    body = client.post("/chat", json={"question": "Explain RAG to me"}).json()

    assert body["answer"] == SAVED["answer"]
    assert body["cache_hit"] is True
    assert body["cache_type"] == "semantic"
    assert body["similarity"] == 0.94
    assert body["prompt_tokens"] == 0 and body["completion_tokens"] == 0
    assert fake_llm.calls == 0  # Groq was never asked

    with test_db() as session:
        row = session.query(Call).one()
    assert (row.cache_hit, row.cache_type, row.similarity, row.cost_usd) == (True, "semantic", 0.94, 0)
    with test_db() as session:
        assert session.query(SemanticCacheEntry).count() == 0  # a hit is not saved again


def test_below_cutoff_calls_groq(fake_llm, fake_lookup):
    fake_lookup.match = {"answer": "something else", "similarity": config.SEMANTIC_CUTOFF - 0.01}
    body = client.post("/chat", json={"question": "Explain RAG to me"}).json()

    assert body["cache_hit"] is False
    assert body["cache_type"] is None and body["similarity"] is None
    assert body["answer"] == fake_llm.result["answer"]
    assert fake_llm.calls == 1


def test_exactly_at_cutoff_is_a_hit(fake_llm, fake_lookup):
    fake_lookup.match = {"answer": "saved", "similarity": config.SEMANTIC_CUTOFF}
    assert client.post("/chat", json={"question": "Hi there"}).json()["cache_type"] == "semantic"


def test_lookup_error_still_returns_200(fake_llm, monkeypatch):
    def broken(vector, model):
        raise ConnectionError("database is down")

    monkeypatch.setattr("services.semantic_cache.find_closest", broken)
    res = client.post("/chat", json={"question": "What is RAG?"})

    assert res.status_code == 200
    assert res.json()["cache_hit"] is False
    assert fake_llm.calls == 1


def test_lookup_timeout_still_returns_200(fake_llm, monkeypatch):
    monkeypatch.setattr(config, "SEMANTIC_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr("services.semantic_cache.find_closest", lambda v, m: time.sleep(0.5))
    res = client.post("/chat", json={"question": "What is RAG?"})

    assert res.status_code == 200
    assert res.json()["cache_hit"] is False
    assert fake_llm.calls == 1


def test_embedding_error_still_returns_200(fake_llm, monkeypatch):
    async def broken_embed(text):
        raise RuntimeError("model failed to load")

    monkeypatch.setattr(services.embeddings, "embed", broken_embed)
    res = client.post("/chat", json={"question": "What is RAG?"})

    assert res.status_code == 200
    assert fake_llm.calls == 1


def test_exact_hit_skips_semantic_step(fake_llm, fake_lookup, monkeypatch):
    client.post("/chat", json={"question": "What is RAG?"})  # miss: fills the exact cache

    embed_calls = []
    real_embed = services.embeddings.embed

    async def counting_embed(text):
        embed_calls.append(text)
        return await real_embed(text)

    monkeypatch.setattr(services.embeddings, "embed", counting_embed)
    lookups_before = fake_lookup.calls
    body = client.post("/chat", json={"question": "What is RAG?"}).json()

    assert body["cache_type"] == "exact"
    assert body["similarity"] is None
    assert embed_calls == []
    assert fake_lookup.calls == lookups_before


def test_miss_embeds_only_once(fake_llm, test_db, monkeypatch):
    embed_calls = []
    real_embed = services.embeddings.embed

    async def counting_embed(text):
        embed_calls.append(text)
        return await real_embed(text)

    monkeypatch.setattr(services.embeddings, "embed", counting_embed)
    client.post("/chat", json={"question": "What is RAG?"})

    assert len(embed_calls) == 1  # used for the lookup AND the background save
    with test_db() as session:
        assert session.query(SemanticCacheEntry).count() == 1


def test_semantic_hit_is_stored_for_exact_lookup(fake_llm, fake_lookup):
    fake_lookup.match = dict(SAVED)
    client.post("/chat", json={"question": "Explain RAG to me"})  # semantic hit

    fake_lookup.match = None  # even with the semantic cache empty now...
    body = client.post("/chat", json={"question": "Explain RAG to me"}).json()

    assert body["cache_type"] == "exact"  # ...Redis answers it
    assert body["answer"] == SAVED["answer"]
    assert fake_llm.calls == 0


def test_new_columns_logged_for_exact_and_miss(fake_llm, test_db):
    client.post("/chat", json={"question": "What is RAG?"})
    client.post("/chat", json={"question": "What is RAG?"})

    with test_db() as session:
        rows = session.query(Call).order_by(Call.id).all()
    assert [(r.cache_type, r.similarity) for r in rows] == [(None, None), ("exact", None)]
