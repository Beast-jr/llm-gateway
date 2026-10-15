import asyncio

from fastapi.testclient import TestClient

import config
import services.cache
from api.main import app
from models.call import Call
from services.cache import make_key, normalize

client = TestClient(app)


def test_miss_then_hit(fake_llm, test_db):
    first = client.post("/chat", json={"question": "What is RAG?"}).json()
    second = client.post("/chat", json={"question": "What is RAG?"}).json()

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["answer"] == first["answer"]
    assert second["prompt_tokens"] == 0 and second["completion_tokens"] == 0
    assert fake_llm.calls == 1  # Groq was only asked once

    # Both calls are logged; the hit costs nothing.
    with test_db() as session:
        rows = session.query(Call).order_by(Call.id).all()
    assert [r.cache_hit for r in rows] == [False, True]
    assert rows[0].cost_usd > 0
    assert rows[1].cost_usd == 0


def test_normalization_makes_same_key():
    assert normalize("  What   is RAG? ") == "what is rag?"
    assert make_key("What is RAG?", "m") == make_key("  what   is rag?  ", "m")
    assert make_key("What is RAG?", "m") != make_key("What is RAG?", "other-model")
    assert make_key("What is RAG?", "m") != make_key("What is RAG", "m")  # punctuation still counts


def test_differently_typed_question_is_a_hit(fake_llm):
    client.post("/chat", json={"question": "What is RAG?"})
    res = client.post("/chat", json={"question": "  what IS   rag?  "}).json()

    assert res["cache_hit"] is True
    assert fake_llm.calls == 1


def test_redis_down_still_returns_200(fake_llm, fake_redis):
    fake_redis.connected = False  # every Redis command now raises ConnectionError

    for _ in range(2):
        res = client.post("/chat", json={"question": "What is RAG?"})
        assert res.status_code == 200
        assert res.json()["cache_hit"] is False

    assert fake_llm.calls == 2  # no cache, so Groq answered both times


def test_slow_redis_is_skipped(fake_llm, monkeypatch):
    class SlowRedis:
        async def get(self, key):
            await asyncio.sleep(1)

        async def set(self, *args, **kwargs):
            await asyncio.sleep(1)

    monkeypatch.setattr(config, "REDIS_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(services.cache, "_get_redis", lambda: SlowRedis())
    res = client.post("/chat", json={"question": "What is RAG?"})

    assert res.status_code == 200
    assert res.json()["cache_hit"] is False
    assert fake_llm.calls == 1


def test_answer_saved_with_ttl(fake_llm, fake_redis):
    client.post("/chat", json={"question": "What is RAG?"})

    async def ttl():
        r = services.cache._get_redis()
        return await r.ttl(make_key("What is RAG?", "openai/gpt-oss-20b", "user:1"))  # the test user is #1

    assert 0 < asyncio.run(ttl()) <= config.CACHE_TTL_SECONDS
