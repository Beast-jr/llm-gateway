import pytest
from fastapi.testclient import TestClient

import services.call_log
import services.gateway
from api.main import app
from models.call import Call
from services.call_log import calculate_cost
from services.llm import LLMError

client = TestClient(app)


def test_successful_chat_saves_one_row(fake_llm, test_db):
    res = client.post("/chat", json={"question": "What is RAG?"})
    assert res.status_code == 200

    with test_db() as session:
        rows = session.query(Call).all()

    assert len(rows) == 1
    row = rows[0]
    assert row.question == "What is RAG?"
    assert row.answer == fake_llm.result["answer"]
    assert row.model == "openai/gpt-oss-20b"
    assert row.provider == "groq"
    assert row.prompt_tokens == 5
    assert row.completion_tokens == 7
    assert row.latency_ms == 12.3
    # 5 input tokens at $0.075/M + 7 output tokens at $0.30/M
    assert row.cost_usd == pytest.approx((5 * 0.075 + 7 * 0.30) / 1_000_000)
    assert row.cache_hit is False
    assert row.created_at is not None


def test_db_failure_still_returns_200(fake_llm, monkeypatch):
    def broken_session():
        raise ConnectionError("database is down")

    monkeypatch.setattr(services.call_log, "get_session", broken_session)
    res = client.post("/chat", json={"question": "What is RAG?"})

    assert res.status_code == 200
    assert res.json()["answer"] == fake_llm.result["answer"]


def test_failed_llm_call_saves_nothing(monkeypatch, test_db):
    # A 502 means there's no answer, so nothing should be logged.
    async def failing_ask_llm(question):
        raise LLMError("Groq request failed")

    monkeypatch.setattr(services.gateway, "ask_llm", failing_ask_llm)
    assert client.post("/chat", json={"question": "Hi"}).status_code == 502

    with test_db() as session:
        assert session.query(Call).count() == 0


def test_unknown_model_costs_zero():
    assert calculate_cost("some-unknown-model", 100, 100) == 0.0
