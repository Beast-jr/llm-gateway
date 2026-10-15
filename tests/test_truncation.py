import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import config
import services.cache
import services.llm
from api.main import app
from models.call import Call
from models.semantic_cache import SemanticCacheEntry
from services.cache import make_key
from tests.conftest import FAKE_RESULT

client = TestClient(app)


@pytest.fixture
def cut_off_llm(monkeypatch):
    """A fake Groq whose answers stop mid-sentence (finish_reason 'length'). cut_off_llm.calls counts requests."""
    class CutOff:
        calls = 0

    async def fake_ask_llm(question):
        CutOff.calls += 1
        return {**FAKE_RESULT, "answer": "First you install the packages, then you", "completion_tokens": 2048,
                "truncated": True}

    monkeypatch.setattr("services.gateway.ask_llm", fake_ask_llm)
    return CutOff


def test_truncated_answer_is_flagged(cut_off_llm):
    body = client.post("/chat", json={"question": "How do I install Docker on Ubuntu?"}).json()
    assert body["truncated"] is True
    assert body["answer"].endswith("then you")


def test_normal_answer_is_not_flagged(fake_llm):
    assert client.post("/chat", json={"question": "What is RAG?"}).json()["truncated"] is False


def test_truncated_answer_is_never_cached_exactly(cut_off_llm):
    client.post("/chat", json={"question": "How do I install Docker on Ubuntu?"})
    second = client.post("/chat", json={"question": "How do I install Docker on Ubuntu?"}).json()

    assert second["cache_hit"] is False  # asked Groq again instead of serving half an answer
    assert cut_off_llm.calls == 2
    redis = services.cache._get_redis()
    key = make_key("How do I install Docker on Ubuntu?", config_model(), "user:1")
    assert asyncio.run(redis.get(key)) is None


def test_truncated_answer_is_never_saved_semantically(cut_off_llm, test_db):
    client.post("/chat", json={"question": "How do I install Docker on Ubuntu?"})
    with test_db() as session:
        assert session.query(SemanticCacheEntry).count() == 0


def test_truncated_answer_is_still_logged_and_still_costs_budget(cut_off_llm, test_db):
    body = client.post("/chat", json={"question": "How do I install Docker on Ubuntu?"}).json()
    assert body["tokens_used_today"] == 5 + 2048  # you paid for those tokens, so they count
    with test_db() as session:
        assert session.query(Call).one().completion_tokens == 2048


def test_complete_answer_is_cached_as_before(fake_llm, test_db):
    client.post("/chat", json={"question": "What is RAG?"})
    assert client.post("/chat", json={"question": "What is RAG?"}).json()["cache_type"] == "exact"
    with test_db() as session:
        assert session.query(SemanticCacheEntry).count() == 1


# ---------- the real ask_llm: finish_reason and the output limit ----------

def fake_groq(finish_reason, captured):
    async def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="some text"), finish_reason=finish_reason)],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4),
            model="openai/gpt-oss-20b",
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@pytest.mark.parametrize("finish_reason, expected", [("length", True), ("stop", False)])
def test_ask_llm_reads_finish_reason(monkeypatch, finish_reason, expected):
    monkeypatch.setattr(services.llm, "_get_client", lambda: fake_groq(finish_reason, {}))
    result = asyncio.run(services.llm.ask_llm("hi"))
    assert result["truncated"] is expected


def test_ask_llm_sends_the_output_limit_from_config(monkeypatch):
    captured = {}
    monkeypatch.setattr(services.llm, "_get_client", lambda: fake_groq("stop", captured))
    asyncio.run(services.llm.ask_llm("hi"))
    assert captured["max_completion_tokens"] == config.MAX_OUTPUT_TOKENS
    assert config.MAX_OUTPUT_TOKENS >= 4096


def config_model():
    return services.llm.get_model()
